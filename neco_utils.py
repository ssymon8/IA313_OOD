import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score


# ------------------------------------------------------------
# Feature extraction (same logic as ViM)
# ------------------------------------------------------------

def extract_features(model, loader, device, layer_name='avgpool'):
    """
    Extract penultimate features from model.
    """
    model.eval()
    features_list = []
    labels_list = []

    batch_features = []

    def hook(module, input, output):
        if output.dim() > 2:
            output = F.adaptive_avg_pool2d(output, 1)
        batch_features.append(output.flatten(1).detach())

    handle = getattr(model, layer_name).register_forward_hook(hook)

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Extracting features (NECO)"):
            images = images.to(device)
            batch_features = []

            _ = model(images)

            features_list.append(batch_features[0].cpu())
            labels_list.append(labels.cpu())

    handle.remove()

    return torch.cat(features_list), torch.cat(labels_list)


# ------------------------------------------------------------
# Compute class means (Neural Collapse)
# ------------------------------------------------------------

def compute_class_means(features, labels, num_classes=100):
    """
    Compute class means μ_k
    """
    class_means = []

    for k in range(num_classes):
        class_features = features[labels == k]
        mean_k = class_features.mean(dim=0)
        class_means.append(mean_k)

    return torch.stack(class_means)  # [C, D]


# ------------------------------------------------------------
# NECO score
# ------------------------------------------------------------

def neco_score(features, class_means):
    """
    S_NECO(x) = min_k ||f(x) - μ_k||^2
    """
    # features: [N, D]
    # class_means: [C, D]

    dists = torch.cdist(features, class_means) ** 2  # [N, C]
    min_dist, _ = torch.min(dists, dim=1)

    return min_dist


# ------------------------------------------------------------
# Full NECO pipeline
# ------------------------------------------------------------

def compute_neco_scores(model, train_loader, id_loader, ood_loader, device):
    """
    Full NECO pipeline:
    - compute class means from train set
    - compute NECO score for ID and OOD
    - compute AUROC
    """

    print("\n===== Computing NECO parameters =====")

    train_features, train_labels = extract_features(model, train_loader, device)
    class_means = compute_class_means(train_features, train_labels)

    print("Computing ID NECO scores...")
    id_features, _ = extract_features(model, id_loader, device)
    id_scores = neco_score(id_features, class_means)

    print("Computing OOD NECO scores...")
    ood_features, _ = extract_features(model, ood_loader, device)
    ood_scores = neco_score(ood_features, class_means)

    # Convert to numpy
    id_scores = id_scores.numpy()
    ood_scores = ood_scores.numpy()

    # For AUROC:
    # ID label = 1, OOD label = 0
    y_true = np.concatenate([
        np.ones_like(id_scores),
        np.zeros_like(ood_scores)
    ])
    y_scores = np.concatenate([id_scores, ood_scores])

    # Important:
    # Large distance = more OOD
    # So we invert sign for AUROC consistency
    auroc = roc_auc_score(y_true, -y_scores)

    print(f"NECO AUROC: {auroc:.4f}")

    return id_scores, ood_scores, auroc
