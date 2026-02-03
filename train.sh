#!/bin/bash
#SBATCH --job-name=resnet18_cifar100_drouet
#SBATCH --partition=ENSTA-l40s
#SBATCH --output=resnet18_cifar100_%j.log
#SBATCH --error=resnet18_cifar100_%j.err
#SBATCH --time=07:00:00
#SBATCH --nodelist=ensta-l40s01.r2.enst.fr

# Load and initialize conda
source ~/.bashrc
module load conda
eval "$(conda shell.bash hook)"

# Activate conda environment
conda activate resnet

# Navigate to project directory
cd ~/gianni

# Create outputs directory if it doesn't exist
mkdir -p outputs

# Run training on scratch model
echo "Starting ResNet18 training on CIFAR-100 (scratch model)..."
srun python train.py -m scratch

# Optional: Run training on torchvision model
# echo "Starting ResNet18 training on CIFAR-100 (torchvision model)..."
# python train.py -m torchvision

echo "Training complete!"
