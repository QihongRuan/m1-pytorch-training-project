#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
import time
import threading
import multiprocessing
import numpy as np

class LargeNetwork(nn.Module):
    def __init__(self):
        super(LargeNetwork, self).__init__()
        self.layers = nn.ModuleList([
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        ])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def cpu_intensive_training():
    """Run CPU-intensive PyTorch training"""
    print("Starting CPU-intensive training...")
    
    # Create large model
    model = LargeNetwork()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Generate large random dataset
    batch_size = 256
    
    for epoch in range(50):  # Run for 50 epochs
        start_time = time.time()
        
        for batch in range(100):  # 100 batches per epoch
            # Generate random data
            data = torch.randn(batch_size, 2048)
            targets = torch.randint(0, 10, (batch_size,))
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            if batch % 20 == 0:
                print(f"Epoch {epoch+1}, Batch {batch+1}, Loss: {loss.item():.4f}")
        
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1} completed in {epoch_time:.2f} seconds")

def matrix_operations_stress():
    """Stress test with large matrix operations"""
    print("Starting matrix operations stress test...")
    
    for i in range(1000):
        # Create large matrices
        a = torch.randn(2048, 2048)
        b = torch.randn(2048, 2048)
        
        # Matrix multiplication
        c = torch.mm(a, b)
        
        # Element-wise operations
        d = torch.sin(c) + torch.cos(c)
        e = torch.exp(torch.tanh(d))
        
        # Eigenvalue decomposition (very CPU intensive)
        if i % 50 == 0:
            try:
                eigenvals = torch.linalg.eigvals(a[:512, :512])  # Smaller matrix for eig
                print(f"Matrix operation {i+1} completed, max eigenvalue: {eigenvals.abs().max():.4f}")
            except:
                print(f"Matrix operation {i+1} completed")

def parallel_worker(worker_id):
    """Worker function for parallel processing"""
    print(f"Worker {worker_id} starting...")
    
    for i in range(200):
        # Heavy computations
        x = torch.randn(1024, 1024)
        y = torch.randn(1024, 1024)
        
        # Multiple operations to stress CPU
        z = torch.mm(x, y)
        z = torch.fft.fft2(z)
        z = torch.abs(z)
        z = torch.log(z + 1e-8)
        
        if i % 50 == 0:
            print(f"Worker {worker_id}: iteration {i+1}")

def run_stress_test():
    """Main stress test function"""
    print("=" * 50)
    print("PyTorch MacBook Pro Stress Test Starting...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CPU count: {multiprocessing.cpu_count()}")
    print("=" * 50)
    
    start_time = time.time()
    
    # Run different stress tests in sequence
    try:
        # 1. Neural network training
        cpu_intensive_training()
        
        print("\n" + "=" * 30)
        print("Moving to matrix operations...")
        print("=" * 30 + "\n")
        
        # 2. Matrix operations
        matrix_operations_stress()
        
        print("\n" + "=" * 30)
        print("Starting parallel processing...")
        print("=" * 30 + "\n")
        
        # 3. Parallel processing to use all cores
        num_workers = multiprocessing.cpu_count()
        threads = []
        
        for i in range(num_workers):
            thread = threading.Thread(target=parallel_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
            
    except KeyboardInterrupt:
        print("\nStopping stress test...")
    
    total_time = time.time() - start_time
    print("\n" + "=" * 50)
    print(f"Stress test completed in {total_time:.2f} seconds")
    print("Your MacBook Pro should be nice and warm now! 🔥")
    print("=" * 50)

if __name__ == "__main__":
    run_stress_test()