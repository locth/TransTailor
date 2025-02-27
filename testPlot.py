import torch
import torchvision
import torchvision.transforms as transforms
from Pruner import Pruner
import argparse
import os
import logging
import time
import matplotlib.pyplot as plt

def LoadModel(device):
    # Load the VGG16 model
    model = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)

    # Replace the last layer of the model with a new layer that matches the number of classes in CIFAR10
    num_classes = 10
    model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, num_classes)

    model = model.to(device)

    return model

def LoadData(numWorker, batchSize):
    # Define the data transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    data_path = os.path.join(".", "data")

    # Load the CIFAR10 dataset
    full_dataset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)

    # Split into Train, Validate, and Test datasets
    train_size = int(0.6 * len(full_dataset))
    validate_size = int(0.2 * len(full_dataset))
    test_size = len(full_dataset) - train_size - validate_size

    train_dataset, validate_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, validate_size, test_size])

    kwargs = {'num_workers': numWorker, 'pin_memory': True} if device == 'cuda' else {}
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchSize, shuffle=True, **kwargs)
    validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=batchSize, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batchSize, shuffle=False, **kwargs)

    return train_loader, validate_loader, test_loader

if __name__ == "__main__":
    # LOAD ARGUMENTS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    FIG_PATH = os.path.join("fig.png")

    # LOAD DATASET
    train_loader, validate_loader, test_loader = LoadData(8, 32)

    model = LoadModel(device)

    # INIT PRUNING SCHEME
    pruner = Pruner(model, train_loader, validate_loader, test_loader, device)
    pruner.LoadState("checkpoint/pruner/TA5_IA10_DROP5_checkpoint_1675.pkl")

    pruner.PlotLosses(pruner.train_losses, pruner.val_losses, FIG_PATH)