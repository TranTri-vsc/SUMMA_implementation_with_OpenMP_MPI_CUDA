# Distributed SUMMA (MPI) with CPU (OpenMP) and GPU (CUDA)

Efficient, multi-node matrix multiplication using the **SUMMA** algorithm. This project implements a 2‑D process grid with MPI broadcasts, a CPU path parallelized via OpenMP, and a GPU path using CUDA. It reports both **end‑to‑end** and **compute‑only** performance and includes a verification routine.

---

## Highlights
- **Algorithm:** SUMMA — broadcast A‑panels along process rows and B‑panels down process columns; accumulate local DGEMM into C.
- **CPU kernel:** OpenMP‑parallel local DGEMM (loop tiling + `collapse`), static scheduling.
- **GPU kernel:** CUDA DGEMM path guarded by a `--gpu` runtime flag.
- **Verification:** Gather distributed C to rank 0 and compare against a reference with tolerance `1e-5`.
- **Metrics:** We report both **end‑to‑end** (MPI + kernel) and **compute‑only** GFLOP/s using `t_max` across ranks.

---

## Results (example: N = 5000)
- **CPU scaling (e.g., P = 2×2 → 4×4 → 8×8):**
  - Significant speedups as P increases; the gap between compute‑only and end‑to‑end widens due to communication overheads.
- **GPU vs CPU at the same P:**
  - GPU widely outperforms CPU on compute‑only metrics; end‑to‑end speedups are moderated by host broadcasts, H2D copies, and kernel launch/synchronization overheads.

Exact numbers, figures, and discussion are in the report.

---

## How it works
1. **Process grid:** Build a √P × √P MPI Cartesian grid; each rank owns a tile of A, B, and C.
2. **Panel broadcasts:** For each k‑block, broadcast A‑panel along the row and B‑panel down the column.
3. **Local compute:** Accumulate `C += A_panel * B_panel` using either:
   - **CPU path:** OpenMP local DGEMM.
   - **GPU path:** CUDA DGEMM (enabled via `--gpu`).
4. **Timing & GFLOPS:** Measure per‑rank end‑to‑end and compute‑only times, reduce by `MAX`, and compute global GFLOP/s using `2*N^3`.
5. **Verification:** Gather C to rank 0, compute max |C − C_ref|, and pass under a set tolerance.

---

## Build

### Prerequisites
- C++17 compiler with OpenMP
- MPI implementation (e.g., MPICH/OpenMPI)
- (Optional) CUDA toolkit for GPU builds

### CPU build
```bash
make cpu
```

### GPU build
```bash
make gpu      # or compile with nvcc and define -DENABLE_CUDA if needed
```

> If you run with `--gpu` without a CUDA build, the program will abort with a clear message.

---

## Run

### Local (examples)
```bash
# 4 MPI ranks on one node (2x2 grid), CPU path, N=5000, verify result
mpirun -n 4 ./summa --N 5000 --verify

# 4 MPI ranks on one node, GPU path
mpirun -n 4 ./summa --N 5000 --gpu --verify
```

### Slurm (examples)
```bash
# CPU (2x2)
srun -N 1 -n 4 -c 8 --cpu-bind=cores ./summa --N 5000

# Scale up (e.g., 4x4)
srun -N 2 -n 16 -c 8 --cpu-bind=cores ./summa --N 5000

# GPU (ensure one GPU per rank)
srun -N 4 -n 16 --gpus-per-task=1 ./summa --N 5000 --gpu
```

---

## Command‑line options
```text
--N <int>     Global matrix dimension (default 512)
--gpu         Use CUDA path (requires GPU build)
--verify      Gather and check C vs. reference on rank 0
```

At startup, options are echoed as: `Opts: N=<N> mode=<cpu|gpu> verify=<0|1>`.

---

## Repo map (key files)
- `main.cpp` — CLI/dispatch and MPI init/finalize.
- `summa_cpu.cpp` — SUMMA loop, MPI broadcasts, timing/GFLOPS, verification hook.
- `local_dgemm_cpu.cpp` — OpenMP local DGEMM.
- `verify.cpp` — Gather C and numeric tolerance check.
- `HW3_Report.pdf` — Runs, screenshots, and analysis.

If your repo also includes GPU sources (e.g., `summa_gpu.cu`, `local_dgemm_gpu.cu`) or Slurm scripts (`run_cpu.sbatch`, `run_gpu.sbatch`, `cpu_verify.sbatch`), list them here for quick navigation.

---

## Reproducing results
We use one methodology:
- **End‑to‑end** timing around the full SUMMA loop (includes MPI and compute).
- **Compute‑only** timing around the local DGEMM / CUDA event region.
- GFLOP/s computed with `t_max` across ranks to reflect the slowest rank.

Details and exact configurations are in the report.

---

## Notes & Learnings
- As P increases, communication and launch overheads start to dominate the gap between compute‑only and end‑to‑end throughput.
- On GPUs, host broadcasts, H2D copies, and kernel launch/sync costs are crucial to track; overlapping communication and computation is a key next step.

---

## Extra notes (for running on NERSC Clusters):
```bash
To compile on Perlmutter: 
module load PrgEnv-gnu
module load cudatoolkit
```
