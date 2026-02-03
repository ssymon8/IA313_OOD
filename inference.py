import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from utils import msp_score, max_logit_score


#inference parameters
batch_size = 512
num_classes = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "./checkpoints/resnet_scratch_epoch_200_ckpt.pth"

#model loading
from resnet18 import ResNet, BasicBlock

model = ResNet(img_channels=3, num_layers=18, block=BasicBlock, num_classes=num_classes)

checkpoint = torch.load(MODEL_PATH, map_location=device)
if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)

model.to(device)
model.eval()

#data loading
def load_data(batch_size=batch_size):
    stats = ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*stats),
    ])

    # 1. In-Distribution (CIFAR-100 Test)
    id_dataset = datasets.CIFAR100(root='data', train=False, download=True, transform=test_transform)
    id_loader = DataLoader(id_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 2. Out-of-Distribution (SVHN Test)
    ood_dataset = datasets.SVHN(root='data', split='test', download=True, transform=test_transform)
    ood_loader = DataLoader(ood_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return id_loader, ood_loader

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
    