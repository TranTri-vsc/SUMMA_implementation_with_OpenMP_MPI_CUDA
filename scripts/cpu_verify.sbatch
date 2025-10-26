#!/bin/bash
#SBATCH -A m5101                # Replace with your NERSC project account
#SBATCH -C cpu                  # CPU partition
#SBATCH -q debug                # debug queue for short test
#SBATCH -t 00:05:00             # Walltime
#SBATCH -N 1                    # just 1 node for verification
#SBATCH -n 4                    # 4 MPI ranks
#SBATCH -c 8                    # 8 cores for each rank
#SBATCH -J cpu-verification     # Job name
#SBATCH -o output/%x-%j.out       # Standard output and error log

mkdir -p output

module load PrgEnv-gnu
make clean && make cpu

export OMP_NUM_THREADS=8
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

N=${N:-1000} # smaller size for quick verification

echo "CPU version verification result"
echo " *** N is ${N}, P is 4 (2x2) Number of thread is ${OMP_NUM_THREADS} *** "
srun -n 4 -c 8 --cpu-bind=cores ./summa --N ${N} --verify