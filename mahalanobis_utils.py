


#-----Mahalanobis distance scoring functions-----#

def mahalanobis_parameters(model, train_loader, device):

    """
    Single layer Mahalanobis parameters computation.
    Computes class means and tied covariance matrix inverse on the penultimate layer 
    of the NN.
    """
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


def multibranch_mahalanobis_parameters(model, train_loader, device):
    """
    Multi-layer Mahalanobis parameters computation.
    Computes class means and tied covariance matrix inverse on multiple layers 
    of the NN.
    Returns a list of (class_means, precision) tuples for each layer.
    """
    # To be implemented if needed
    pass

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