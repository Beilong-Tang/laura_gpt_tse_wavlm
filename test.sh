#!/bin/bash
#SBATCH -J train_test
#SBATCH -n 32
#SBATCH -N 1
#SBATCH --gres=dcu:4
#SBATCH -p kshdnormal02
#SBATCH -o log.out
#SBATCH -e log.err

export ROCBLAS_TENSILE_LIBPATH=/public/software/compiler/rocm/dtk-23.10/lib/rocblas/library_dcu2

source ~/anaconda3/etc/profile.d/conda.sh
conda activate bltang_new

module purge
module load compiler/devtoolset/7.3.1
module load mpi/hpcx/2.7.4/gcc-7.3.1
module load compiler/rocm/dtk-23.10


python -u test.py