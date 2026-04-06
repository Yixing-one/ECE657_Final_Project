import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_cifar10(data_dir='./data'):
    # Create directory
    os.makedirs(data_dir, exist_ok=True)
    
    # (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010) are normalization values for cifar
    # Transforms
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    val_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=val_transform
    )
    
    return train_dataset, val_dataset

def get_loaders(batch_size=128, num_workers=2, data_dir='./data'):
    train_dataset, val_dataset = get_cifar10(data_dir)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader
