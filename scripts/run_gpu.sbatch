#!/bin/bash
#SBATCH -A m5101                # Replace with your NERSC project account
#SBATCH -C gpu                  # GPU partition
#SBATCH -q debug                # regular queue (or debug for short test)
#SBATCH -t 00:30:00             # Walltime
#SBATCH -N 4                    # Number of nodes
#SBATCH -n 16                   # MPI ranks (must be a perfect square)
#SBATCH -c 8                    # CPU cores per rank
#SBATCH --gpus-per-task=1       # 1 GPU per rank
#SBATCH -J summa-gpu-N5000      # Job name
#SBATCH -o output/%x-%j.out        # Standard output and error log

mkdir -p output

module load PrgEnv-gnu
module load cudatoolkit
make clean && make gpu

export OMP_NUM_THREADS=8
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

N=${N:-5000}

echo "Result of GPU when N is ${N}, Number of thread is ${OMP_NUM_THREADS} and 1 GPU per rank"

echo " *** For P is 4 (2x2) and on 1 GPU node (4 GPUs)***"
srun -N 1 -n 4  -c 8 --gpus-per-task=1 --gpu-bind=single:1 ./summa --N ${N} --gpu --verify

echo " *** For P is 16 (4x4) and on 4 GPU nodes (16 GPUs)***"
srun -N 4 -n 16 -c 8 --gpus-per-task=1 --gpu-bind=single:1 ./summa --N ${N} --gpu --verify
