//! DQSDv2 — Deterministic Spectral Arbitration Kernel
//! Core equation: Z_k += dt*(Sum_j(Z_k*S_j - Z_j*S_k)*kappa_kj - lambda*Z_k)
//! Energy law: d||Z||^2/dt = -2*lambda*||Z||^2 (kappa antisymmetric)
//! License: MIT — Author: Daniel J. Dillberg

#![cfg_attr(not(feature = "std"), no_std)]

pub const RMAX: usize = 16;
pub const KILL_K: u8 = 3;

// ── Fixed-point trait ───────────────────────────────────────
pub trait Fp: Copy + Clone + Send + Sync + 'static {
    fn zero() -> Self;
    fn add(self, r: Self) -> Self;
    fn sub(self, r: Self) -> Self;
    fn mul(self, r: Self) -> Self;
    fn from_f64(v: f64) -> Self;
    fn to_f64(self) -> f64;
}

// ── Q16.16 (i32) — WASM / embedded ─────────────────────────
#[derive(Clone, Copy)]
pub struct Q16(pub i32);
impl Fp for Q16 {
    fn zero() -> Self { Q16(0) }
    fn add(self, r: Self) -> Self { Q16(self.0.saturating_add(r.0)) }
    fn sub(self, r: Self) -> Self { Q16(self.0.saturating_sub(r.0)) }
    fn mul(self, r: Self) -> Self { Q16(((self.0 as i64 * r.0 as i64) >> 16) as i32) }
    fn from_f64(v: f64) -> Self {
        let v = if v > 32000.0 { 32000.0 } else if v < -32000.0 { -32000.0 } else { v };
        Q16((v * 65536.0) as i32)
    }
    fn to_f64(self) -> f64 { self.0 as f64 / 65536.0 }
}

// ── Q31.32 (i64) — PC / gaming / RF ────────────────────────
#[derive(Clone, Copy)]
pub struct Q31(pub i64);
impl Fp for Q31 {
    fn zero() -> Self { Q31(0) }
    fn add(self, r: Self) -> Self { Q31(self.0.saturating_add(r.0)) }
    fn sub(self, r: Self) -> Self { Q31(self.0.saturating_sub(r.0)) }
    fn mul(self, r: Self) -> Self { Q31(((self.0 as i128 * r.0 as i128) >> 32) as i64) }
    fn from_f64(v: f64) -> Self {
        let v = if v > 2e9 { 2e9 } else if v < -2e9 { -2e9 } else { v };
        Q31((v * (1u64 << 32) as f64) as i64)
    }
    fn to_f64(self) -> f64 { self.0 as f64 / (1u64 << 32) as f64 }
}

// ── Q64.64 (i128) — archival / deep-space ───────────────────
#[derive(Clone, Copy)]
pub struct Q64(pub i128);
impl Fp for Q64 {
    fn zero() -> Self { Q64(0) }
    fn add(self, r: Self) -> Self { Q64(self.0.saturating_add(r.0)) }
    fn sub(self, r: Self) -> Self { Q64(self.0.saturating_sub(r.0)) }
    fn mul(self, r: Self) -> Self {
        let lim: i128 = 1i128 << 96;
        let a = if self.0 > lim { lim } else if self.0 < -lim { -lim } else { self.0 };
        let b = if r.0 > lim { lim } else if r.0 < -lim { -lim } else { r.0 };
        Q64(a.saturating_mul(b) >> 64)
    }
    fn from_f64(v: f64) -> Self {
        let v = if v > 1e18 { 1e18 } else if v < -1e18 { -1e18 } else { v };
        Q64((v * (1u128 << 64) as f64) as i128)
    }
    fn to_f64(self) -> f64 { self.0 as f64 / (1u128 << 64) as f64 }
}

// ── Ghost classification (diagnostic only) ──────────────────
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Ghost {
    Nominal = 0, Collapse = 1, Diffuse = 2, Echo = 3,
    Burst = 4, Trap = 5, Vacuum = 6,
}

// ── Frame output (ABI-stable) ───────────────────────────────
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Frame {
    pub id: u64,
    pub energy: f64,
    pub h: f64,
    pub stress: f64,
    pub entropy: f64,
    pub omega_norm: f64,
    pub ghost: u8,
    pub contained: u8,
    pub hash: u64,
    _pad: [u8; 6],
}

impl Default for Frame {
    fn default() -> Self {
        Self {
            id: 0, energy: 0.0, h: 0.0, stress: 0.0, entropy: 0.0,
            omega_norm: 0.0, ghost: 0, contained: 0, hash: 0, _pad: [0; 6],
        }
    }
}

// ── Math (no libm — all approximated) ───────────────────────

fn sin_approx(x: f64) -> f64 {
    let p = core::f64::consts::PI;
    let tp = 2.0 * p;
    // manual floor: x - floor(x/tp)*tp
    let n = x / tp;
    let n = n - (if n >= 0.0 { n as i64 } else { n as i64 - 1 }) as f64;
    let x = n * tp;
    let x = if x > p { x - tp } else { x };
    let ax = if x < 0.0 { -x } else { x };
    16.0 * x * (p - ax) / (5.0 * p * p - 4.0 * ax * (p - ax))
}

fn ln_approx(x: f64) -> f64 {
    if x <= 0.0 { return -40.0; }
    let b = x.to_bits() as i64;
    let e = ((b >> 52) & 0x7ff) - 1023;
    let f = f64::from_bits(((b & 0x000f_ffff_ffff_ffff) | 0x3ff0_0000_0000_0000) as u64);
    (e as f64 + (f - 1.0) * (2.0 - 0.333 * (f - 1.0))) * 0.693_147_180_559_945_3
}

fn fnv1a_f64(vals: &[f64]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    let mut i = 0;
    while i < vals.len() {
        h ^= vals[i].to_bits();
        h = h.wrapping_mul(0x100000001b3);
        i += 1;
    }
    h
}

fn sqrt_approx(v: f64) -> f64 {
    if v <= 0.0 { return 0.0; }
    // Newton-Raphson: 3 iterations from bit-hack seed
    let bits = v.to_bits();
    let seed = f64::from_bits((bits >> 1) + (1023u64 << 51));
    let mut x = seed;
    x = 0.5 * (x + v / x);
    x = 0.5 * (x + v / x);
    x = 0.5 * (x + v / x);
    x
}

// ── Nonlinear operator config ───────────────────────────────
#[derive(Clone, Copy)]
pub struct NlCfg {
    pub vajra: bool,  pub vajra_a: f64,
    pub rose: bool,   pub rose_d: f64,  pub rose_k: f64,
    pub dini: bool,   pub dini_r: f64,
}

impl Default for NlCfg {
    fn default() -> Self {
        Self {
            vajra: false, vajra_a: 0.01,
            rose: false, rose_d: 0.01, rose_k: 4.0,
            dini: false, dini_r: 0.0625,
        }
    }
}

// ── Core state ──────────────────────────────────────────────
pub struct Core<T: Fp> {
    pub z: [T; RMAX],
    pub s: [T; RMAX],
    pub omega: [T; RMAX],
    pub kappa: [T; RMAX * RMAX],
    lam: T, dt: T, al: T, omal: T, od: T,
    pub r: usize,
    pub frame: u64,
    pub alive: u8,
    cf: u8,
    pub nl: NlCfg,
    theta: f64,
}

impl<T: Fp> Core<T> {
    pub fn new(r: usize, lam: f64, dt: f64, al: f64, nl: NlCfg) -> Self {
        let r = if r > RMAX { RMAX } else { r };
        let mut z = [T::zero(); RMAX];
        let mut kappa = [T::zero(); RMAX * RMAX];

        let mut k = 0;
        while k < r {
            z[k] = T::from_f64(0.01 * (k as f64 + 1.0));
            k += 1;
        }

        let mut i = 0;
        while i < r {
            let mut j = 0;
            while j < r {
                kappa[i * RMAX + j] = T::from_f64(
                    sin_approx((i as f64) * 1.37 - (j as f64) * 1.73)
                );
                j += 1;
            }
            i += 1;
        }

        Self {
            z, s: [T::zero(); RMAX], omega: [T::zero(); RMAX], kappa,
            lam: T::from_f64(lam), dt: T::from_f64(dt),
            al: T::from_f64(al), omal: T::from_f64(1.0 - al),
            od: T::from_f64(0.999),
            r, frame: 0, alive: 1, cf: 0, nl, theta: 0.0,
        }
    }

    pub fn step(&mut self, u_max: f64) -> Frame {
        let r = self.r;
        let dtf = self.dt.to_f64();

        // 1. CONTAINMENT
        let mut e2: f64 = 0.0;
        let mut k = 0;
        while k < r { let v = self.z[k].to_f64(); e2 += v * v; k += 1; }

        if e2 > u_max * u_max || e2 != e2 {
            self.cf += 1;
        } else {
            self.cf = 0;
        }

        let killed = self.cf >= KILL_K;
        if killed {
            k = 0;
            while k < r { self.z[k] = T::from_f64(1e-6); k += 1; }
            self.s = [T::zero(); RMAX];
            self.omega = [T::zero(); RMAX];
            self.alive = 1;
            self.cf = 0;
        }

        // 2. LIE BRACKET (indexed antisymmetric — NOT scalar commutator)
        let mut zn = [T::zero(); RMAX];
        k = 0;
        while k < r {
            let mut tq = T::zero();
            let mut j = 0;
            while j < r {
                if j != k {
                    let br = self.z[k].mul(self.s[j]).sub(self.z[j].mul(self.s[k]));
                    tq = tq.add(br.mul(self.kappa[k * RMAX + j]));
                }
                j += 1;
            }
            zn[k] = self.z[k].add(self.dt.mul(tq.sub(self.lam.mul(self.z[k]))));
            k += 1;
        }
        self.z = zn;

        // 3. EMA
        if self.cf == 0 {
            k = 0;
            while k < r {
                self.s[k] = self.al.mul(self.s[k]).add(self.omal.mul(self.z[k]));
                k += 1;
            }
        }

        // 4. NONLINEAR (optional)
        self.theta += dtf;
        if self.nl.vajra {
            let f = T::from_f64(1.0 - self.nl.vajra_a * dtf);
            k = 0; while k < r { self.z[k] = self.z[k].mul(f); k += 1; }
        }
        if self.nl.rose {
            let d = self.nl.rose_d * dtf;
            k = 0;
            while k < r {
                let zk = self.z[k].to_f64();
                let tgt = sin_approx(self.nl.rose_k * (self.theta + k as f64 * 0.1));
                self.z[k] = T::from_f64(zk + d * (tgt - zk));
                k += 1;
            }
        }
        if self.nl.dini {
            let f = T::from_f64(1.0 - self.nl.dini_r * dtf);
            k = 0; while k < r { self.z[k] = self.z[k].mul(f); k += 1; }
        }

        // 5. OMEGA
        k = 0;
        while k < r {
            self.omega[k] = self.omega[k]
                .add(self.z[k].mul(self.omal).mul(self.dt))
                .mul(self.od);
            k += 1;
        }

        // 6. DIAGNOSTICS
        let en = {
            let s = sqrt_approx(e2);
            if s < 1e-15 { 1e-15 } else { s }
        };

        let sn = {
            let mut acc = 0.0;
            k = 0; while k < r { let v = self.s[k].to_f64(); acc += v * v; k += 1; }
            sqrt_approx(acc)
        };

        let on = {
            let mut acc = 0.0;
            k = 0; while k < r { let v = self.omega[k].to_f64(); acc += v * v; k += 1; }
            sqrt_approx(acc)
        };

        let stress = sn / en;
        let h = en;

        let entropy = {
            let t = e2 + 1e-15;
            let mut ent = 0.0;
            k = 0;
            while k < r {
                let v = self.z[k].to_f64();
                let p = (v * v) / t;
                if p > 1e-15 { ent -= p * ln_approx(p); }
                k += 1;
            }
            ent
        };

        let or_val = on / en;
        let ghost = if killed { Ghost::Vacuum }
            else if stress > 1.5 { Ghost::Burst }
            else if en < 1e-10 && entropy < 0.1 { Ghost::Collapse }
            else if entropy > 2.0 { Ghost::Diffuse }
            else if entropy < 0.3 && stress < 0.1 { Ghost::Echo }
            else if or_val > 1.0 { Ghost::Trap }
            else { Ghost::Nominal };

        // 7. FRAME + HASH
        self.frame += 1;
        let mut hv = [0.0f64; 32];
        k = 0; while k < r { hv[k] = self.z[k].to_f64(); k += 1; }
        while k < 2 * r { hv[k] = self.s[k - r].to_f64(); k += 1; }
        let hash = fnv1a_f64(&hv[..2 * r]);

        Frame {
            id: self.frame, energy: en, h, stress, entropy, omega_norm: on,
            ghost: ghost as u8, contained: killed as u8, hash, _pad: [0; 6],
        }
    }
}

// ── C ABI ───────────────────────────────────────────────────
pub type AbiCore = Core<Q31>;

#[no_mangle]
pub extern "C" fn dvsm_init(r: u32, lam: f64, dt: f64, al: f64) -> *mut AbiCore {
    #[cfg(feature = "std")]
    {
        let c = Box::new(AbiCore::new(r as usize, lam, dt, al, NlCfg::default()));
        Box::into_raw(c)
    }
    #[cfg(not(feature = "std"))]
    { core::ptr::null_mut() }
}

#[no_mangle]
pub unsafe extern "C" fn dvsm_step(c: *mut AbiCore, u: f64, o: *mut Frame) -> i32 {
    let c = match c.as_mut() { Some(c) => c, None => return -1 };
    let f = c.step(u);
    if let Some(o) = o.as_mut() { *o = f; }
    0
}

#[no_mangle]
pub unsafe extern "C" fn dvsm_free(c: *mut AbiCore) {
    #[cfg(feature = "std")]
    if !c.is_null() { drop(Box::from_raw(c)); }
}
