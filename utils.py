import matplotlib.pyplot as plt
import os
import torch
import numpy as np
from tqdm import tqdm

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

plt.style.use("ggplot")


def get_data(batch_size=64):
    #Data augmentation for the training set and normalization
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    #Normalization for the validation set
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    # CIFAR100 training dataset.
    dataset_train = datasets.CIFAR100(
        root="data",
        train=True,
        download=True,
        transform=train_transform,
    )

    # CIFAR100 validation dataset.
    dataset_valid = datasets.CIFAR100(
        root="data",
        train=False,
        download=True,
        transform=valid_transform,
    )

    # Create data loaders.
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    valid_loader = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    return train_loader, valid_loader


def save_plots(train_acc, valid_acc, train_loss, valid_loss, name=None):
    """
    Function to save the loss and accuracy plots to disk.
    """
    # Accuracy plots.
    plt.figure(figsize=(10, 7))
    plt.plot(train_acc, color="tab:blue", linestyle="-", label="train accuracy")
    plt.plot(valid_acc, color="tab:red", linestyle="-", label="validataion accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join("outputs", name + "_accuracy.png"))

    # Loss plots.
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color="tab:blue", linestyle="-", label="train loss")
    plt.plot(valid_loss, color="tab:red", linestyle="-", label="validataion loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join("outputs", name + "_loss.png"))

#-----Baseline OOD scoring functions-----#

def msp_score(outputs):
    """
    Function to compute the Maximum Softmax Probability (MSP) scores.
    """
    softmax_outputs = torch.softmax(outputs, dim=1)
    msp, _ = torch.max(softmax_outputs, dim=1)
    return msp

def max_logit_score(outputs):
    """
    Function to compute the Maximum Logit scores.
    """
    max_logit, _ = torch.max(outputs, dim=1)
    return max_logit


#-----Mahalanobis distance scoring functions-----#

def mahalanobis_parameters(model, train_loader, device):
    model.eval()
       
    # Hook to extract features from the avgpool layer
    features_list = []
    all_labels = []
    def hook(module, input, output):
        features_list.append(output.flatten(1).detach().cpu())
    
    handle = model.avgpool.register_forward_hook(hook)
    
    print("Mahalanobis stats computing (Train Set)...")
    with torch.no_grad():
        for images, labels in tqdm(train_loader):
            images = images.to(device)
            _ = model(images)
            all_labels.append(labels)
    handle.remove()
    
    # Concatenation
    features = torch.cat(features_list) # [50000, 512]
    labels = torch.cat(all_labels)     # [50000]
    
    # class means calculation
    class_means = []
    num_classes = 100
    
    for c in range(num_classes):
        class_features = features[labels == c]
        mean = torch.mean(class_features, dim=0)
        class_means.append(mean)
        
    class_means = torch.stack(class_means)
    
    # Tied Covariance Matrix
    # centering the features around their class means
    # X_centered = X - Mu_{y}
    centered_features = []
    for c in range(num_classes):
        class_features = features[labels == c]
        mean = class_means[c]
        centered_features.append(class_features - mean)
        
    centered_features = torch.cat(centered_features)
    
    # empiricalk cov
    # shape: [512, 512]
    cov = torch.matmul(centered_features.t(), centered_features) / (len(features) - 1)
    
    # matrix inversion
    precision = torch.linalg.pinv(cov, hermitian=True)
    
    return class_means, precision

def mahalanobis_score(model, loader, class_means, precision, device):
    model.eval()
    
    class_means = class_means.to(device)
    precision = precision.to(device)

    scores = []
    
    # Hook to extract features from the avgpool layer
    batch_features = []
    def hook(module, input, output):
        batch_features.append(output.flatten(1))
    
    handle = model.avgpool.register_forward_hook(hook)
    
    print("Computing Mahalanobis scores...")
    with torch.no_grad():
        for images, _ in tqdm(loader):
            images = images.to(device)
            batch_features= []

            _ = model(images)
            
            # Extract features
            features = batch_features[0]  # [batch_size, feature_dim]
            
            # Compute Mahalanobis distance to each class mean
            batch_scores = []
            for c in range(100):
                #centering
                delta = features - class_means[c]
                # Mahalanobis distance
                term1 = torch.matmul(delta, precision)
                dist_c = torch.sum(term1 * delta, dim=1)
                batch_scores.append(dist_c)
            
            dists_all_classes = torch.stack(batch_scores, dim=1)  # [batch_size, num_classes]

            #we take the min of the distances
            min_dists, _ = torch.min(dists_all_classes, dim=1)
            scores.append(-min_dists.cpu())  # negative distance as score

    handle.remove()
    return torch.cat(scores)