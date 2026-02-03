import matplotlib.pyplot as plt
import os

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
