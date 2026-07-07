#!/bin/bash

#SBATCH --job-name=muon_d_qpp_bc_seeded_inf_oldenv
#SBATCH --output=bc_seeded_inf_oldenv.out
#SBATCH --nodes=8
#SBATCH --gres=gpu:4
#SBATCH --time=1-00:00:00
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

srun --nodes=8 \
     --gres=gpu:4 \
     --export=ALL \
     python -u ferminet/configs/diamond/bc_relaxed/inference_bc_seeded.py --server_addr="$IP_ADDR:$PORT"
