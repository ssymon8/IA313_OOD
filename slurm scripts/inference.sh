#!/bin/bash
#SBATCH --job-name=resnet18_cifar100_drouet
#SBATCH --partition=ENSTA-l40s
#SBATCH --output=resnet18_cifar100_%j.log
#SBATCH --error=resnet18_cifar100_%j.err
#SBATCH --time=20:00:00
#SBATCH --nodelist=ensta-l40s01.r2.enst.fr
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16

# Load environment
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate resnet

# Aller dans ton dossier
cd ~/gianni

echo "Début de l'évaluation OOD..."
echo "Job ID: $SLURM_JOB_ID"
echo "GPU utilisé: $CUDA_VISIBLE_DEVICES"

# Lancement du script python
# srun permet de propager les signaux SLURM correctement
srun python inference.py

echo "Évaluation terminée."