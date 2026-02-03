import matplotlib.pyplot as plt
import os
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms


#-----Mahalanobis distance scoring functions-----#

def extract_features(model, loader, device, layer_name='avgpool'):
    """
    Extract features from a specified layer of the model for all samples in the loader.
    """
    
    model.eval()
    features_list = []
    labels_list = []
    
    # Hook to extract features from the specified layer
    batch_features = []
    def hook(module, input, output):
        if output.dim() > 2:
            #if 4D tensor (layer 3 for our ResNet18), apply adaptive avg pooling
            output = F.adaptive_avg_pool2d(output, 1)
        batch_features.append(output.flatten(1).detach().cpu())
    
    #we hook the target layer
    handle = getattr(model, layer_name).register_forward_hook(hook)
    
    print(f"Extracting features from layer: {layer_name}...")
    with torch.no_grad():
        for images, _ in tqdm(loader):
            images = images.to(device)
            batch_features = []

            _ = model(images)

            # extract features
            features = batch_features[0]  # [batch_size, feature_dim]
            features_list.append(features)
    
    handle.remove() #hook cleanup

    # Concatenation
    features = torch.cat(features_list)  # [num_samples, feature_dim]
    labels = torch.cat(labels_list)      # [num_samples]
    return features, labels

def get_class_means(features, labels, num_classes=100):
    """
    Compute class means from features and labels.
    """
    class_means = []
    for c in range(num_classes):
        class_features = features[labels == c]
        mean = torch.mean(class_features, dim=0)
        class_means.append(mean)
    class_means = torch.stack(class_means)
    return class_means

def estimate_inv_covariance(features, labels, class_means):
    """
    Estimate the tied inverse covariance matrix from features and class means.
    """
    #centering the features around their class means
    centered_features = []
    for c in range(len(class_means)):
        class_features = features[labels == c]
        mean = class_means[c]
        centered_features.append(class_features - mean)
    
    centered_features = torch.cat(centered_features)

    # empirical covariance
    cov = torch.matmul(centered_features.t(), centered_features) / (len(features) - 1)

    precision = torch.linalg.pinv(cov, hermitian=True)
    return precision

def mahalanobis_parameters(model, train_loader, device, layer_name='avgpool'):

    """
    Single layer Mahalanobis parameters computation.
    Computes class means and tied covariance matrix inverse on the designated layer 
    of the NN.
    """
    # feature extraction
    features, labels = extract_features(model, train_loader, device, layer_name)

    # class means...
    class_means = get_class_means(features, labels, num_classes=100)

    # precision matrix...
    precision = estimate_inv_covariance(features, labels, class_means)

    return class_means, precision


def multibranch_mahalanobis_parameters(model, train_loader, device):
    """
    Multi-layer Mahalanobis parameters computation.
    Computes class means and tied covariance matrix inverse on multiple layers 
    of the NN.
    Returns a list of (class_means, precision) tuples for each layer.
    """

    confidence = []

    layers = ['layer1', 'layer2', 'layer3', 'layer4', 'avgpool']

    for layer_name in layers:
        class_means, precision = mahalanobis_parameters(
            model, train_loader, device, layer_name
        )
        confidence.append((class_means, precision))
    
    return confidence

def mahalanobis_score(model, loader, class_means, precision, device, layer_name='avgpool'):
    model.eval()
    
    class_means = class_means.to(device)
    precision = precision.to(device)

    scores = []
    
    # Hook to extract features
    batch_features = []
    def hook(module, input, output):
        if output.dim() > 2:
            #if 4D tensor (layer 3 for our ResNet18), apply adaptive avg pooling
            output = F.adaptive_avg_pool2d(output, 1)
        batch_features.append(output.flatten(1))

    target_layer = getattr(model, layer_name)    
    handle = target_layer.register_forward_hook(hook)
    
    print("Computing Mahalanobis scores...")
    with torch.no_grad():
        for images, _ in loader:
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

def multibranch_mahalanobis_score(model, loader, confidence, device, alpha=None):
    """
    Computes Mahalanobis scores from multiple layers and linearly combines them.
    alpha: list of weights for each layer. If None, uniform weights are used.
    Returns the combined scores.
    """
    model.eval()
    
    layers = ['layer1', 'layer2', 'layer3', 'layer4', 'avgpool']
    
    # if no alpha provided, use uniform weights
    if alpha is None:
        alpha = [1.0/len(layers)] * len(layers)

    total_scores = None
    
    print("Computing Multibranch Mahalanobis scores...")
    for i, layer_name in enumerate(tqdm(layers)):
        
        # i-th layer parameters
        stats = confidence[i]
        class_means = stats[0]
        precision = stats[1]
        
        # i-th layer scores
        layer_score = mahalanobis_score(
            model, loader, class_means, precision, device, layer_name=layer_name
        )
        
        if total_scores is None:
            total_scores = alpha[i] * layer_score
        else:
            total_scores += alpha[i] * layer_score
            
    return total_scores