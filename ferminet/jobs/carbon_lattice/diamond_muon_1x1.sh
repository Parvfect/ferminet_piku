#!/bin/bash

#SBATCH --job-name=diamon_muon_1x1
#SBATCH --output=diamond_muon_1x1_nopp.out
#SBATCH --gpus=1
#SBATCH --time=20:00:00         # Hours:Mins:Secs

hostname
cd ~
source ~/miniforge3/bin/activate
conda activate ferminet-piku
nvidia-smi --list-gpus
pwd
cd ferminet

ferminet --config ferminet/configs/diamond/carbon_1x1_muon.py \
	--config.log.save_path /projects/u6em/parv/diamond/1x1/


