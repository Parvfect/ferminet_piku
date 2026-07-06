#!/bin/bash

#SBATCH --job-name=muon_d_qpp_widebi_verify
#SBATCH --output=muon_wide_burnin_verify.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=0-01:00:00
hostname
cd ~
source /home/u6em/parvfect.u6em/miniforge3/bin/activate
conda activate ferminet-piku

pwd
cd ferminet_piku

export NVIDIA_TF32_OVERRIDE=0
export JAX_DEFAULT_MATMUL_PRECISION=highest
export JAX_ENABLE_X64=0

PORT=1345
IP_ADDR=$(ifconfig 2> /dev/null | awk '$1 == "inet" {print $2}' | head -n 2 | tail -n 1)

srun --nodes=1 \
     --gres=gpu:4 \
     --export=ALL \
     python -u ferminet/configs/diamond/bc_relaxed/wide_burnin_verify.py --server_addr="$IP_ADDR:$PORT"
