use dqsdv2::*;
use std::io::Write;

fn main() {
    println!("DQSDv2 - Deterministic Spectral Arbitration Kernel");
    println!("Author: Daniel J. Dillberg\n");

    run::<Q16>("Q16.16", 4,  0.005,  NlCfg::default());
    run::<Q31>("Q31.32", 8,  0.001,  NlCfg { vajra: true, ..Default::default() });
    run::<Q64>("Q64.64", 16, 0.0001, NlCfg::default());
}

fn run<T: Fp>(name: &str, r: usize, lam: f64, nl: NlCfg) {
    let mut core = Core::<T>::new(r, lam, 1.0 / 240.0, 0.98, nl);

    let path = format!("{}.bin", name.replace('.', "_"));
    let mut file = std::fs::File::create(&path).expect("cannot create output file");

    let start = std::time::Instant::now();
    let n: u64 = 100_000;
    let mut last = Frame::default();

    for _ in 0..n {
        last = core.step(100.0);
        let _ = file.write_all(&last.id.to_le_bytes());
        let _ = file.write_all(&last.energy.to_le_bytes());
        let _ = file.write_all(&last.h.to_le_bytes());
        let _ = file.write_all(&[last.ghost, last.contained]);
        let _ = file.write_all(&last.hash.to_le_bytes());
    }

    let us = start.elapsed().as_micros() as f64 / n as f64;
    println!(
        "  {} R={:2} | {:.1} us/frame | E={:.12} S={:.4} G={} | #{:016X} | -> {}",
        name, r, us, last.energy, last.stress, last.ghost, last.hash, path
    );
}
