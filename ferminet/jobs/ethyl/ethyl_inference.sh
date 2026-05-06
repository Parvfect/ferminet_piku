#!/bin/bash

#SBATCH --job-name=ethyl_inference
#SBATCH --output=ethyl_inference_5e3.out
#SBATCH --gpus=1
#SBATCH --time=4:00:00         # Hours:Mins:Secs

hostname
cd ~
source ~/miniforge3/bin/activate
conda activate ferminet_af
nvidia-smi --list-gpus
pwd
cd ferminet_af/ferminet
python ethyl.py --ckpt /projects/u6em/parv/ethyl_radical/qmcjax_ckpt_096000.npz --n_iterations 50000
