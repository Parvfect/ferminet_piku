#!/bin/bash

#SBATCH --job-name=methyl_psiformer
#SBATCH --output=methyl_psiformer.out
#SBATCH --gpus=1
#SBATCH --time=20:00:00         # Hours:Mins:Secs

hostname
cd ~
source ~/miniforge3/bin/activate
conda activate ferminet_af
nvidia-smi --list-gpus
pwd
cd ferminet_af
python /home/u6em/parvfect.u6em/miniforge3/envs/jax_cuda/bin/ferminet \
	--config ferminet/configs/methyl_radical.py \
	--config.log.save_path /projects/u6em/parv/methyl \
	--config.log.save_frequency 2000 \
	--config.network.network_type psiformer \
