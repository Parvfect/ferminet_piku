#!/bin/bash

#SBATCH --job-name=muonioum_com_1000_ferminet
#SBATCH --output=muonioum_com_1000_ferminet.out
#SBATCH --gpus=1
#SBATCH --time=10:00:00         # Hours:Mins:Secs

hostname
cd ~
source ~/miniforge3/bin/activate
conda activate ferminet_af
nvidia-smi --list-gpus
pwd
cd ferminet_af
python /home/u6em/parvfect.u6em/miniforge3/envs/jax_cuda/bin/ferminet \
	--config ferminet/configs/muonioum.py \
	--config.log.save_path /projects/u6em/parv/muonioum_com3_ferminet_1000 \
	--config.log.save_frequency 2000 \
	--config.network.psiformer.num_layers 2 \ 
