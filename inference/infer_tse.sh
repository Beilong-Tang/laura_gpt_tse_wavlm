#!/bin/bash
#SBATCH -J laur_gpt_se_inference
#SBATCH -n 32
#SBATCH -N 1
#SBATCH --gres=dcu:4
#SBATCH -p kshdnormal
#SBATCH -o inference/log_infer.out
#SBATCH -e inference/log_infer.err

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

# python -u inference/infer.py --config_file exp/tse_wavlm/config/conf.yaml --model_file /DKUdata/tangbl/laura_gpt_tse/exp/tse_wavlm/ckpt/conf/best.pth --output_dir /DKUdata/tangbl/laura_gpt_tse/output/conf_no_ref --default_config inference/infer_no_ref_decoder.yaml --device cuda:3


# Infer with LJSpeech K 1024
python -u inference/infer.py --config_file exp/tse_wavlm/config/conf_ljspeech.yaml --model_file /DKUdata/tangbl/laura_gpt_tse/exp/tse_wavlm/ckpt/conf_ljspeech/best.pth --output_dir /DKUdata/tangbl/laura_gpt_tse/output/conf_ljspeech --default_config inference/infer_no_ref_decoder.yaml --gpus cuda:0 cuda:1 cudda:4 cuda:5

## Infer with auxilary audio
# python -u inference/infer.py --config_file exp/tse_wavlm/config/conf.yaml --model_file /public/home/qinxy/bltang/laura_gpt_tse/exp/tse_wavlm/ckpt/laura_gpt_tse_ljspeech.pt --output_dir /public/home/qinxy/bltang/laura_gpt_tse/output/conf --default_config inference/infer.yaml --device cuda:0