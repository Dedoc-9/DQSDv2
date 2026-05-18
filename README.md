# DQSDv2

**Deterministic Spectral Arbitration Kernel**

```
Z_k += dt * (Sum_j (Z_k*S_j - Z_j*S_k) * kappa_kj - lambda*Z_k)
d||Z||^2/dt = -2*lambda*||Z||^2
```

## Build & Run

```
cd DQSDv2
cargo build --release
cargo run --release
```

## Output

Three binary telemetry files: Q16_16.bin, Q31_32.bin, Q64_64.bin

Author: Daniel J. Dillberg — License: MIT
