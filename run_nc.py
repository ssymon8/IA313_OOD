from mahalanobis_utils import extract_features
from nc_utils import compute_nc_metrics, plot_nc_visualizations

# 1. Extract features and labels for NC analysis

print("Extraction of features and labels for NC analysis...")
features, labels = extract_features(model, train_loader, device, layer_name='avgpool')

# 2. Get the weights and biases of the last layer
W = model.fc.weight.detach()
b = model.fc.bias.detach()

# 3. compute NC metrics
metrics = compute_nc_metrics(
    features.to(device), 
    labels.to(device), 
    W.to(device),
    biases=b.to(device),
    num_classes=100
)

plot_nc_visualizations(metrics)