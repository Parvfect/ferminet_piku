#!/bin/bash

#SBATCH --job-name=muon_silicon_q_t_relaxed
#SBATCH --output=t_relaxed.out
#SBATCH --nodes=8
#SBATCH --gres=gpu:4
#SBATCH --time=1-00:00:00
hostname
cd ~
source ~/miniforge3/bin/activate
conda activate ferminet-piku

pwd
cd ferminet_remote
cd ferminet_piku

export NVIDIA_TF32_OVERRIDE=0
export JAX_DEFAULT_MATMUL_PRECISION=highest
export JAX_ENABLE_X64=0

PORT=1345
IP_ADDR=$(ifconfig 2> /dev/null | awk '$1 == "inet" {print $2}' | head -n 2 | tail -n 1)

srun --nodes=8 \
     --gres=gpu:4 \
     --export=ALL \
     python -u ferminet/configs/silicon/t_relaxed.py --server_addr="$IP_ADDR:$PORT"
