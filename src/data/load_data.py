import torch
from torchvision import datasets
from src.data.preprocess import get_transform

def load_data(batch_size=64):
    transform = get_transform()

    train_data = datasets.MNIST(root='data/raw', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root='data/raw', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader