#!/bin/bash
#SBATCH --job-name=resnet18_inference_drouet
#SBATCH --partition=ENSTA-l40s
#SBATCH --output=resnet18_inference_%j.log
#SBATCH --error=resnet18_inference_%j.err
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
srun python inference.py

echo "End of inference!"