# CSC_5IA23 Project: Out-of-Distribution Detection and Neural Collapse

This repository contains the complete implementation of an anomaly detection pipeline (Out-of-Distribution or OOD) and an analysis of the Neural Collapse phenomenon. The project centers around a ResNet-18 model trained from scratch on CIFAR-100, evaluated against an OOD dataset (SVHN).

The main framework used is PyTorch.

## Table of Contents
1. [Repository Architecture](#repository-architecture)
2. [Installation and Prerequisites](#installation-and-prerequisites)
3. [Model Training](#model-training)
4. [OOD Scores: Baselines, Mahalanobis, and ViM](#ood-scores)
5. [Neural Collapse (NC) Analysis](#neural-collapse-analysis)
6. [NECO Score](#neco-score)

---

## Repository Architecture

The project is modular, separating training, feature extraction, mathematical score computations, and visualizations:

* `train.py` / `training_utils.py`: Scripts dedicated to training the ResNet-18 on CIFAR-100 (terminal phase pushed to 200 epochs).
* `resnet18.py` / `resnet18_torchvision.py`: Network architecture definition.
* `inference.py`: Main script for evaluating AUROC metrics across various OOD scores.
* `utils.py`: Utility functions, including baseline scores (MSP, Max Logit, Energy Score).
* `mahalanobis_utils.py`: Implementation of the Mahalanobis score (Single-Layer and Multibranch/Feature Ensemble).
* `vim_utils.py`: Implementation of the ViM (Virtual-logit Matching) method, extracting the Principal Space via SVD on the classifier weights.
* `nc_utils.py`: Tools for mathematical extraction and visualization of Neural Collapse properties (NC1 to NC5) in the latent space.
* `neco_utils.py`: Implementation of the NECO score, exploiting the rigid geometry of Neural Collapse for OOD detection.
* `slurm_scripts/`: Launch scripts for cluster execution (GPU management, e.g., L40S).

---

## Pre-existing repositories

For some part of the project we took the inspiration for the implementation from the following repositories:

* For the initial Resnet Architecture and the utilities to inspect the training evolution: https://github.com/hubert10/ResNet18_from_Scratch_using_PyTorch
* For the Mahalanobis Detection method implementation: https://github.com/HarryAnthony/Mahalanobis-OOD-detection

## Installation and Prerequisites

Ensure you have a Python environment with a CUDA-compatible graphics card.

```bash
git clone [https://github.com/ssymon8/IA313_OOD.git](https://github.com/ssymon8/IA313_OOD.git)
cd IA313_OOD
pip install -r requirements.txt