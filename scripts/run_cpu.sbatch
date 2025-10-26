#!/bin/bash
#SBATCH -A m5101                # Replace with your NERSC project account
#SBATCH -C cpu                  # CPU partition
#SBATCH -q debug                # regular queue (or debug for short test)
#SBATCH -t 00:30:00             # Walltime
#SBATCH -N 4                    # Number of nodes
#SBATCH -n 64                   # MPI ranks (must be a perfect square)
#SBATCH -c 8                    # CPU cores per rank
#SBATCH -J summa-cpu-N5000      # Job name
#SBATCH -o output/%x-%j.out        # Standard output and error log

mkdir -p output

module load PrgEnv-gnu
make clean && make cpu

export OMP_NUM_THREADS=8
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

N=${N:-5000}

echo "Result of CPU when N is ${N}, Number of thread is ${OMP_NUM_THREADS}"

echo " *** For P is 4 (2x2) ***"
srun -N 1 -n 4  -c 8 --cpu-bind=cores ./summa --N ${N} --verify

echo " *** For P is 16 (4x4) ***"
srun -N 1 -n 16 -c 8 --cpu-bind=cores ./summa --N ${N} --verify

echo " *** For P is 64 (8x8) ***"
srun -N 4 -n 64 -c 8 --cpu-bind=cores ./summa --N ${N} --verify
