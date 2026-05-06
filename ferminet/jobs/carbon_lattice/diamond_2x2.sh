#!/bin/bash

#SBATCH --job-name=diamond_2x2
#SBATCH --output=diamond_2x2.out
#SBATCH --gpus=1
#SBATCH --time=20:00:00         # Hours:Mins:Secs

hostname
cd ~
source ~/miniforge3/bin/activate
conda activate ferminet-piku
nvidia-smi --list-gpus
pwd
cd ferminet

ferminet --config ferminet/configs/carbon_2x2.py \
	--config.log.save_path /projects/u6em/parv/carbon_lattice/2x2/non_muon

