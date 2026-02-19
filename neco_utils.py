import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


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
# Compute class means
# ------------------------------------------------------------

def compute_class_means(features, labels, num_classes=100):
    """
    Compute μ_k for each class
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
    Neural Collapse OOD score.

    We return NEGATIVE squared distance so that:
        - High score = ID
        - Low score = OOD

    S_NECO(x) = - min_k ||f(x) - μ_k||^2
    """

    # Compute squared Euclidean distance
    dists = torch.cdist(features, class_means) ** 2  # [N, C]
    min_dist, _ = torch.min(dists, dim=1)

    return -min_dist  # IMPORTANT: negative for AUROC consistency


# ------------------------------------------------------------
# Full NECO pipeline
# ------------------------------------------------------------

def compute_neco_scores(model, train_loader, id_loader, ood_loader, device):
    """
    Compute NECO OOD scores.
    """

    print("\n===== Computing NECO parameters =====")

    # 1️⃣ Compute class means on train set
    train_features, train_labels = extract_features(model, train_loader, device)
    class_means = compute_class_means(train_features, train_labels)

    # Move to device once
    class_means = class_means.to(device)

    # 2️⃣ ID scores
    print("Computing ID NECO scores...")
    id_features, _ = extract_features(model, id_loader, device)
    id_features = id_features.to(device)
    id_scores = neco_score(id_features, class_means).cpu().numpy()

    # 3️⃣ OOD scores
    print("Computing OOD NECO scores...")
    ood_features, _ = extract_features(model, ood_loader, device)
    ood_features = ood_features.to(device)
    ood_scores = neco_score(ood_features, class_means).cpu().numpy()

    return id_scores, ood_scores
