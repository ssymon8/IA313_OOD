import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm

def compute_nc_metrics(features, labels, weights, biases=None, num_classes=100):
    """
    Computes NC1 to NC5 metrics based on the extracted features, labels, and last layer weights.
    """
    D = features.shape[1]
    
    # 1. Global mean and class means
    mu_G = torch.mean(features, dim=0) # [D]
    
    mu_c = torch.zeros(num_classes, D).to(features.device)
    within_class_vars = []
    
    for c in range(num_classes):
        mask = (labels == c)
        class_features = features[mask]
        
        if len(class_features) > 0:
            mu_c[c] = torch.mean(class_features, dim=0)
            # NC1 : intra class variance (mean of L2 norms squared)
            variance = torch.mean(torch.norm(class_features - mu_c[c], dim=1)**2)
            within_class_vars.append(variance.item())
        else:
            within_class_vars.append(0.0)
            
    # Centering of the means
    M_centered = mu_c - mu_G # [C, D]
    
    # 2. NC2 : ETF Geometry (Equiangular Tight Frame)
    # similarity matrix between class means (cosine similarity)
    M_norm = F.normalize(M_centered, p=2, dim=1)
    cosine_matrix_nc2 = torch.matmul(M_norm, M_norm.t()) # [C, C]
    
    # 3. NC3 : weights alignment vs class means
    # Cosinus between W_c et M_c
    W_norm = F.normalize(weights, p=2, dim=1)
    # take the diagonal of the matrix product
    cosines_nc3 = torch.sum(W_norm * M_norm, dim=1) # [C]
    
    metrics = {
        'within_class_vars': within_class_vars,
        'cosine_matrix_nc2': cosine_matrix_nc2.cpu().numpy(),
        'cosines_nc3': cosines_nc3.cpu().numpy(),
        'theoretical_angle': -1.0 / (num_classes - 1)
    }
    
    if biases is not None:
        metrics['biases'] = biases.cpu().numpy()
        
    return metrics

def plot_nc_visualizations(metrics, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    
    # --- Plot NC1 : Intra-class variance---
    plt.figure(figsize=(10, 5))
    plt.bar(range(100), metrics['within_class_vars'], color='tab:blue')
    plt.title("NC1: Variance Intra-Classe par Classe")
    plt.xlabel("Classe")
    plt.ylabel("Variance (Norme L2 au carré)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, "nc1_within_class_variance.png"))
    plt.close()

    # --- Plot NC2 : similarity matrices ---
    plt.figure(figsize=(8, 7))
    sns.heatmap(metrics['cosine_matrix_nc2'], cmap='coolwarm', center=0, 
                vmin=-0.1, vmax=1.0, cbar_kws={'label': 'Cosine Similarity'})
    plt.title(f"NC2: ETF Geometry\n(Théorie Hors-Diagonale = {metrics['theoretical_angle']:.4f})")
    plt.xlabel("Classe i")
    plt.ylabel("Classe j")
    plt.savefig(os.path.join(output_dir, "nc2_class_mean_cosine.png"))
    plt.close()

    # --- Plot NC3 : Weight alignement / means ---
    plt.figure(figsize=(10, 5))
    plt.bar(range(100), metrics['cosines_nc3'], color='tab:green')
    plt.axhline(y=1.0, color='r', linestyle='--', label='Alignement Parfait (1.0)')
    plt.title("NC3: Similarité Cosinus entre Poids (W) et Moyennes (M)")
    plt.xlabel("Classe")
    plt.ylabel("Cosine Similarity")
    plt.ylim(0, 1.1)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "nc3_weight_mean_alignment.png"))
    plt.close()
    
    # --- Plot NC5 : bias norms ---
    if 'biases' in metrics:
        plt.figure(figsize=(10, 5))
        plt.bar(range(100), metrics['biases'], color='tab:purple')
        plt.title("NC5: Valeur des Biais du Classifieur")
        plt.xlabel("Classe")
        plt.ylabel("Valeur du Biais")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(output_dir, "nc5_biases.png"))
        plt.close()
        
    print(f"Visualisations NC sauvegardées dans le dossier '{output_dir}/'.")