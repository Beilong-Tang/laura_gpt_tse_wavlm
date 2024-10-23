#!/bin/bash
#SBATCH -J train_laura_se
#SBATCH -p kshdnormal02
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=dcu:4
#SBATCH -o log.out
#SBATCH -e log.err

# export MIOPEN_FIND_MODE=3
# export HSA_FORCE_FINE_GRAIN_PRICE=1
# export NCCL_IB_HCA=mlx5_0
# export NCCL_SOCKET_IFNAME=ib0

export ROCBLAS_TENSILE_LIBPATH=/public/software/compiler/rocm/dtk-23.10/lib/rocblas/library_dcu2

source ~/anaconda3/etc/profile.d/conda.sh
conda activate bltang_new

module purge
module load compiler/devtoolset/7.3.1
module load mpi/hpcx/2.7.4/gcc-7.3.1
module load compiler/rocm/dtk-23.10


python -u train.py --config exp/se_wavlm/config/conf.yaml