#!/bin/bash
#SBATCH --job-name=resnet18_nc_drouet
#SBATCH --partition=ENSTA-l40s
#SBATCH --output=resnet18_nc_%j.log
#SBATCH --error=resnet18_nc_%j.err
#SBATCH --time=20:00:00
#SBATCH --nodelist=ensta-l40s01.r2.enst.fr
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16

# Load environment
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate resnet

# Navigate to project directory
cd ~/gianni

# Run inference
srun python run_nc.py

echo "End of NC readings!"