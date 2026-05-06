#!/bin/bash

#SBATCH --job-name=methyl_inference
#SBATCH --output=methyl_inferences/methyl_inference_4.out
#SBATCH --gpus=1
#SBATCH --time=4:00:00         # Hours:Mins:Secs

hostname
cd ~
source ~/miniforge3/bin/activate
conda activate ferminet_af
nvidia-smi --list-gpus
pwd
cd ferminet_af/ferminet
python methyl.py --ckpt /projects/u6em/parv/methyl_radical/qmcjax_ckpt_344000.npz --n_iterations 20000
