import torch
import torch.nn as nn
import numpy as np
from torch.linalg import pinv, eigh
from tqdm import tqdm
import torch.nn.functional as F

def extract_features_and_logits(model, loader, device, layer_name='avgpool'):
    """
    Extract features before last layer and logits
    """
    model.eval()
    features_list = []
    logits_list = []
    
    # Hook 
    batch_features = []
    def hook(module, input, output):
        if output.dim() > 2:
            output = F.adaptive_avg_pool2d(output, 1)
        batch_features.append(output.flatten(1).detach())
    
    handle = getattr(model, layer_name).register_forward_hook(hook)
    
    print(f"Extracting features/logits for Vim from : {layer_name}")
    
    with torch.no_grad():
        for images, _ in tqdm(loader, desc="Extraction ViM"):
            images = images.to(device)
            batch_features = []
            
            # Forward
            logits = model(images)
            
            features_list.append(batch_features[0].cpu())
            logits_list.append(logits.detach().cpu())
            
    handle.remove()
    
    return torch.cat(features_list), torch.cat(logits_list)

def vim_parameters(model, train_loader, device, layer_name='avgpool'):
    """
    Compute ViM params:
    - u is the virtual origin
    - NS is the residual space
    - alpha is the balancing factor
    """

    features, logits = extract_features_and_logits(model, train_loader, device, layer_name)
    
    print("Computing ViM parameters...")
    
    #Computing on the device
    features = features.to(device).double()
    logits = logits.to(device).double()
    
    if hasattr(model, 'fc'):
        fc = model.fc

    W = fc.weight.detach().to(device).double() # [C, D]
    b = fc.bias.detach().to(device).double()   # [C]
    
    # 3. computing the virtual origin
    Winv = pinv(W)
    u = -torch.matmul(Winv, b) # [D]
    
    # 4. Centering the features around u
    X_centered = features - u
    
    # 5. computing the different spaces
    U, S, Vh = torch.linalg.svd(W, full_matrices=True)

    num_classes = W.shape[0]
    V= Vh.t() # [D, D]
    NS = V[:, num_classes:] # [D, D-C]
    
    vlogits = torch.logsumexp(logits, dim=1)
    
    projected = torch.matmul(X_centered, NS)
    residual_norm = torch.norm(projected, dim=1)
    
    alpha = torch.mean(vlogits) / torch.mean(residual_norm)
    
    print(f"calibrated vim, alpha = {alpha.item():.4f}")
    
    #returning to cpu and float32 for inference and savign
    return {
        'u': u.float().cpu(),
        'NS': NS.float().cpu(),
        'alpha': alpha.float().cpu()
    }

def vim_score(model, loader, vim_stats, device, layer_name='avgpool'):
    """
    Score = Energy - alpha * Residual_Norm
    """
    model.eval()
    
    # load stats
    u = vim_stats['u'].to(device)
    NS = vim_stats['NS'].to(device)
    alpha = vim_stats['alpha'].to(device)
    
    scores = []
    
    # Hook
    batch_features = []
    def hook(module, input, output):
        if output.dim() > 2:
            output = F.adaptive_avg_pool2d(output, 1)
        batch_features.append(output.flatten(1).detach())
    
    handle = getattr(model, layer_name).register_forward_hook(hook)
    
    print("Calcul des scores ViM...")
    with torch.no_grad():
        for images, _ in tqdm(loader):
            images = images.to(device)
            batch_features = []
            
            # Forward for logits
            logits = model(images)
            features = batch_features[0] # [B, 512]
            
            energy = torch.logsumexp(logits, dim=1)
            
            # Residual score
            x_centered = features - u
            # Projection on NS
            projected = torch.matmul(x_centered, NS)
            # Norm
            res_norm = torch.norm(projected, dim=1)
            
            score = energy - alpha * res_norm
            
            scores.append(score.cpu())
            
    handle.remove()
    return torch.cat(scores)