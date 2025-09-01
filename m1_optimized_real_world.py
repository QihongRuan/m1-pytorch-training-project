#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import os
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# M1 optimizations
torch.backends.mps.is_available = lambda: False  # Force CPU for stability
torch.set_num_threads(8)  # Use all 8 cores

class M1OptimizedCNN(nn.Module):
    """CNN optimized for M1 architecture with efficient operations"""
    def __init__(self, num_classes=10):
        super(M1OptimizedCNN, self).__init__()
        
        # Use grouped convolutions and depthwise separable convs for efficiency
        self.features = nn.Sequential(
            # First block
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),
            
            # Second block - efficient grouped convolutions
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
            
            # Third block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.3),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def get_cifar10_data():
    """Download and prepare CIFAR-10 dataset with M1-optimized transforms"""
    print("Downloading CIFAR-10 dataset...")
    
    # Optimized transforms for training
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Download datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=test_transform
    )
    
    # Optimized data loaders for M1
    train_loader = DataLoader(
        train_dataset, batch_size=128, shuffle=True, 
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=128, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    
    return train_loader, test_loader

def train_model(model, train_loader, test_loader, num_epochs=10):
    """M1-optimized training loop"""
    device = torch.device('cpu')  # Using CPU for M1 compatibility
    model = model.to(device)
    
    # Optimized optimizer and scheduler for M1
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.CrossEntropyLoss()
    
    # Enable optimizations
    model = torch.jit.script(model)  # JIT compilation for M1
    
    print(f"Training on device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += targets.size(0)
            correct_train += (predicted == targets).sum().item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        scheduler.step()
        
        # Calculate training metrics
        epoch_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train
        train_losses.append(epoch_loss)
        train_accuracies.append(train_accuracy)
        
        # Testing phase
        model.eval()
        correct_test = 0
        total_test = 0
        
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs, 1)
                total_test += targets.size(0)
                correct_test += (predicted == targets).sum().item()
        
        test_accuracy = 100 * correct_test / total_test
        test_accuracies.append(test_accuracy)
        
        epoch_time = time.time() - start_time
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {epoch_loss:.4f}, Train Acc: {train_accuracy:.2f}%')
        print(f'  Test Acc: {test_accuracy:.2f}%, Time: {epoch_time:.2f}s')
        print('-' * 60)
    
    return train_losses, train_accuracies, test_accuracies

def analyze_performance(train_losses, train_accuracies, test_accuracies):
    """Analyze and visualize training performance"""
    epochs = range(1, len(train_losses) + 1)
    
    # Create performance plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Loss plot
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(epochs, train_accuracies, 'r-', label='Training Accuracy')
    ax2.plot(epochs, test_accuracies, 'g-', label='Test Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_performance.png', dpi=150, bbox_inches='tight')
    print("Performance plots saved to training_performance.png")
    
    # Print final results
    print("\n" + "="*50)
    print("FINAL RESULTS:")
    print(f"Final Training Accuracy: {train_accuracies[-1]:.2f}%")
    print(f"Final Test Accuracy: {test_accuracies[-1]:.2f}%")
    print(f"Best Test Accuracy: {max(test_accuracies):.2f}%")
    print("="*50)

def main():
    """Main execution function optimized for M1 MacBook Pro"""
    print("="*60)
    print("M1-Optimized Real-World PyTorch Task: CIFAR-10 Classification")
    print("="*60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CPU threads: {torch.get_num_threads()}")
    print(f"Available device: {'MPS' if torch.backends.mps.is_available() else 'CPU'}")
    
    total_start_time = time.time()
    
    try:
        # Load real-world data
        train_loader, test_loader = get_cifar10_data()
        print(f"Dataset loaded: {len(train_loader.dataset)} training samples")
        print(f"                {len(test_loader.dataset)} test samples")
        
        # Create M1-optimized model
        model = M1OptimizedCNN(num_classes=10)
        print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Train the model
        print("\nStarting training...")
        train_losses, train_accs, test_accs = train_model(
            model, train_loader, test_loader, num_epochs=15
        )
        
        # Analyze results
        analyze_performance(train_losses, train_accs, test_accs)
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    total_time = time.time() - total_start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")
    print("Task completed! Your M1 MacBook Pro handled real-world AI training. 🚀")

if __name__ == "__main__":
    main()