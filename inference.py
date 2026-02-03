import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from utils import msp_score, max_logit_score, energy_score
from mahalanobis_utils import mahalanobis_parameters, mahalanobis_score, multibranch_mahalanobis_parameters, multibranch_mahalanobis_score
from resnet18 import ResNet, BasicBlock

#inference parameters
batch_size = 512
num_classes = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "./checkpoints/resnet_scratch_epoch_200_ckpt.pth"

#model loading
def load_model(MODEL_PATH, device):
    model = ResNet(img_channels=3, num_layers=18, block=BasicBlock, num_classes=num_classes)

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model

#data loading
def load_data(batch_size=batch_size):
    stats = ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*stats),
    ])

    # Train loader without data augmentation for Mahalanobis parameters
    train_dataset = datasets.CIFAR100(root='data', train=True, download=True, transform=test_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 1. In-Distribution (CIFAR-100 Test)
    id_dataset = datasets.CIFAR100(root='data', train=False, download=True, transform=test_transform)
    id_loader = DataLoader(id_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 2. Out-of-Distribution (SVHN Test)
    ood_dataset = datasets.SVHN(root='data', split='test', download=True, transform=test_transform)
    ood_loader = DataLoader(ood_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, id_loader, ood_loader

#score computation
def compute_ood_scores(model, loader, score_fn):
    scores = []

    print(f"Computing OOD scores for {score_fn.__name__}...")
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            outputs = model(images)
            batch_scores = score_fn(outputs)
            scores.append(batch_scores.cpu().numpy())
    
    return np.concatenate(scores)

if __name__ == "__main__":
    model = load_model(MODEL_PATH, device)
    train_loader, id_loader, ood_loader = load_data(batch_size)
    
    # compute and save mahalanobis parameters
    if os.path.exists('mahalanobis_stats.pth'):
            print("loading mahalanobis stats...")
            stats = torch.load('mahalanobis_stats.pth')
            class_means = stats['class_means']
            precision = stats['precision']
    else:
        # computation and saving if not existing
        class_means, precision = mahalanobis_parameters(model, train_loader, device)
        torch.save({
            'class_means': class_means,
            'precision': precision
        }, 'mahalanobis_stats.pth')

    if os.path.exists('multibranch_mahalanobis_stats.pth'):
            print("loading multibranch mahalanobis stats...")
            multibranch_stats = torch.load('multibranch_mahalanobis_stats.pth')
            multibranch_confidence = multibranch_stats['confidence']
    else:
         multibranch_confidence = multibranch_mahalanobis_parameters(model, train_loader, device)
         torch.save({'confidence': multibranch_confidence}, 'multibranch_mahalanobis_stats.pth')
    
    # ------compute scores------

    #MSP Scores
    id_scores_msp = compute_ood_scores(model, id_loader, msp_score)
    ood_scores_msp = compute_ood_scores(model, ood_loader, msp_score)

    #Max Logit Scores
    id_scores_maxlogit = compute_ood_scores(model, id_loader, max_logit_score)
    ood_scores_maxlogit = compute_ood_scores(model, ood_loader, max_logit_score)

    #Energy Scores
    id_scores_energy = compute_ood_scores(model, id_loader, energy_score)
    ood_scores_energy = compute_ood_scores(model, ood_loader, energy_score)

    #Mahalanobis Scores
    id_scores_mahalanobis = mahalanobis_score(model, id_loader, class_means, precision, device).cpu().numpy()
    ood_scores_mahalanobis = mahalanobis_score(model, ood_loader, class_means, precision, device).cpu().numpy()

    #Multibranch Mahalanobis Scores
    id_scores_multibranch_mahalanobis = multibranch_mahalanobis_score(model, id_loader, multibranch_confidence, device).cpu().numpy()
    ood_scores_multibranch_mahalanobis = multibranch_mahalanobis_score(model, ood_loader, multibranch_confidence, device).cpu().numpy()


    #Other OOD metrics to be added
    #
    #
    #

    def evaluate_auroc(id_scores, ood_scores, name):
        # Label 1 for ID, 0 for OOD
        y_true = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
        y_scores = np.concatenate([id_scores, ood_scores])
        
        auroc = roc_auc_score(y_true, y_scores)
        print(f"Métrique {name} - AUROC: {auroc:.4f}")
        return auroc
    
    print("--Results--:")

    # Evaluate AUROC for MSP
    evaluate_auroc(id_scores_msp, ood_scores_msp, "-MSP-")
    # Evaluate AUROC for Max Logit
    evaluate_auroc(id_scores_maxlogit, ood_scores_maxlogit, "-Max Logit-")
    # Evaluate AUROC for Energy
    evaluate_auroc(id_scores_energy, ood_scores_energy, "-Energy-")
    #Evaluate AUROC for Mahalanobis
    evaluate_auroc(id_scores_mahalanobis, ood_scores_mahalanobis, "-Mahalanobis-")
    #Evaluate AUROC for Multibranch Mahalanobis
    evaluate_auroc(id_scores_multibranch_mahalanobis, ood_scores_multibranch_mahalanobis, "-Multibranch Mahalanobis-")



