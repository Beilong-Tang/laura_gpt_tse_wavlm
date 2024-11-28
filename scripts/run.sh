#!/bin/bash
#SBATCH -J dump_libri2mix_data_to_wavlm
#SBATCH -n 32
#SBATCH -N 1
#SBATCH --gres=dcu:4
#SBATCH -p kshdnormal
#SBATCH -o log.out
#SBATCH -e log.err

export MIOPEN_FIND_MODE=3
export HSA_FORCE_FINE_GRAIN_PRICE=1
export NCCL_IB_HCA=mlx5_0
export NCCL_SOCKET_IFNAME=ib0

# export ROCBLAS_TENSILE_LIBPATH=/public/software/compiler/rocm/dtk-23.10/lib/rocblas/library_dcu2

source ~/anaconda3/etc/profile.d/conda.sh
conda activate bltang_new

module purge
module load compiler/devtoolset/7.3.1
module load mpi/hpcx/2.7.4/gcc-7.3.1
module load compiler/rocm/dtk-23.04.1

base_path=/public/home/qinxy/bltang/data/LibriMix/Libri2Mix/Libri2Mix
list_path=$base_path/wav16k/min/list
output_path=$base_path/wav16k/min/wavlm_6
num_proc=8
gpus="cuda:0 cuda:1 cuda:2 cuda:3"
wavlm_ckpt=/public/home/qinxy/bltang/wavlm_large/wavlm_large_new/WavLM-Large.pt

mode=("train" "test" "dev")

### Generate Libri2mix Training Data
for data in "${mode[@]}"; do
    echo "Processing $data"
    python -u exp/laura_gpt_se/scripts/dump_data_to_wavlm.py --clean_scp $list_path/$data/s1.scp \
        --mix_scp $list_path/$data/mix.scp \
        --output_dir $output_path/$data \
        --wavlm_ckpt $wavlm_ckpt \
        --num_proc $num_proc \
        --gpus $gpus
done



