# src/data_loader.py

import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

def get_data_loaders(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert image to PyTorch Tensor
        transforms.Normalize((0.1307,), (0.3081,))  # Normalize to mean/std
    ])

    train_dataset = MNIST(root='data/', train=True, transform=transform, download=True)
    test_dataset = MNIST(root='data/', train=False, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
