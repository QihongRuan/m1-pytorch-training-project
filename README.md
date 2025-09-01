# M1 MacBook Pro PyTorch Training Project

## 🚀 Project Overview

This project demonstrates **M1-optimized PyTorch training** on a MacBook Pro, featuring real-world CIFAR-10 image classification with comprehensive monitoring and visualization tools.

### 📊 Key Achievements
- **Real-world AI Training**: 50,000 CIFAR-10 images (10 categories)
- **M1 Optimization**: Leverages all 8 CPU cores (4 performance + 4 efficiency)
- **1.28M Parameter CNN**: Custom neural network architecture
- **Real-time Monitoring**: Live CPU, memory, and training metrics
- **Comprehensive Logging**: Training progress visualization

## 🔧 System Specifications
- **Device**: MacBook Pro (Late 2020)
- **Chip**: Apple M1
- **CPU**: 8 cores (4 performance + 4 efficiency)
- **Memory**: 8 GB unified memory
- **OS**: macOS (Darwin 24.5.0)
- **PyTorch Version**: 2.2.2

## 📁 Project Structure

```
m1-pytorch-training-project/
├── m1_optimized_real_world.py     # Main training script (M1 optimized)
├── pytorch_stress_test.py         # CPU stress testing script
├── monitor_training.py            # Advanced training monitor
├── quick_monitor.py               # Real-time system monitor
├── requirements.txt               # Python dependencies
└── README.md                     # This documentation
```

## 🧠 Neural Network Architecture

**Custom M1-Optimized CNN:**
- **Input**: 32x32x3 RGB images (CIFAR-10)
- **Architecture**: 
  - 3 Convolutional blocks with BatchNorm + ReLU + Dropout
  - Channel progression: 3 → 64 → 128 → 256
  - Adaptive Global Average Pooling
  - Fully connected layers: 256 → 512 → 10 (classes)
- **Parameters**: 1,283,914 trainable parameters
- **Optimizer**: AdamW with Cosine Annealing scheduler
- **Regularization**: Dropout (0.1-0.5) + L2 weight decay

## 📈 Training Performance

### M1 Optimization Features:
- **JIT Compilation**: TorchScript for faster execution
- **Multi-core Utilization**: 8-core CPU optimization
- **Efficient Data Loading**: 4 worker processes with persistent workers
- **Memory Optimization**: Pin memory + batch normalization
- **Gradient Clipping**: Stability improvements

### Real-time Monitoring:
- **Training Metrics**: Loss, accuracy per epoch/batch
- **System Performance**: CPU usage, memory consumption
- **Progress Tracking**: ETA estimation and completion percentage
- **Visualization**: Matplotlib-based real-time plots

## 🎯 Dataset: CIFAR-10

**Categories (10 classes):**
1. Airplane ✈️
2. Automobile 🚗
3. Bird 🐦
4. Cat 🐱
5. Deer 🦌
6. Dog 🐶
7. Frog 🐸
8. Horse 🐴
9. Ship 🚢
10. Truck 🚚

**Dataset Stats:**
- **Training**: 50,000 images
- **Testing**: 10,000 images
- **Image Size**: 32×32 pixels, RGB
- **Download Size**: ~170MB

## ⚡ Performance Benchmarks

**Training Performance:**
- **Epochs**: 15 total
- **Batch Size**: 128 (optimized for M1 memory bandwidth)
- **Training Time**: ~15-18 minutes
- **CPU Utilization**: 100%+ (multi-core)
- **Memory Usage**: ~1.1GB peak
- **Accuracy**: 70-80% typical CIFAR-10 performance

## 🔍 Real-time Monitoring Features

### System Monitoring:
- **Process CPU Usage**: Per-process M1 core utilization
- **Memory Tracking**: RAM usage and allocation
- **Multi-process Handling**: Main + 4 data loader workers
- **Real-time Updates**: 5-second refresh intervals

### Training Visualization:
- **Loss Curves**: Real-time training loss progression
- **Accuracy Plots**: Training vs. validation accuracy
- **System Metrics**: CPU and memory usage over time
- **Progress Estimation**: ETA calculations and completion bars

## 🛠️ Usage Instructions

### 1. Install Dependencies
```bash
pip install torch torchvision matplotlib psutil
```

### 2. Run M1-Optimized Training
```bash
python m1_optimized_real_world.py
```

### 3. Start Real-time Monitor (in separate terminal)
```bash
python quick_monitor.py
```

### 4. Run Stress Test (optional)
```bash
python pytorch_stress_test.py
```

## 📊 Expected Outputs

1. **training_performance.png** - Training/validation curves
2. **Console logs** - Epoch progress, loss, accuracy
3. **Real-time monitoring** - Live system performance
4. **Model checkpoints** - Trained network weights

## 🔬 Technical Details

### M1-Specific Optimizations:
- **CPU-only execution**: Optimized for M1 architecture
- **Thread allocation**: 8 threads matching M1 cores
- **Memory efficiency**: Unified memory architecture utilization
- **Thermal management**: Automatic throttling prevention

### Data Augmentation:
- **Random horizontal flip**: 50% probability
- **Random rotation**: ±10 degrees
- **Color jitter**: Brightness, contrast, saturation variation
- **Normalization**: ImageNet-style mean/std normalization

## 🎯 Key Results

**Typical Performance Metrics:**
- **Final Training Accuracy**: ~85-90%
- **Final Test Accuracy**: ~75-80%
- **Training Loss**: Converges to ~0.3-0.5
- **System Performance**: Stable throughout training
- **M1 Utilization**: Optimal multi-core usage

## 🚀 Future Enhancements

- **Metal Performance Shaders (MPS)**: GPU acceleration when stable
- **Model Quantization**: Post-training optimization
- **Transfer Learning**: Pre-trained model fine-tuning
- **Distributed Training**: Multi-device scaling
- **TensorBoard Integration**: Advanced visualization

## 👨‍💻 Author

**Qihong Ruan** (Tesla-20)
- Email: ambitionyouth95@gmail.com
- GitHub: [@QihongRuan](https://github.com/QihongRuan)

## 📝 License

MIT License - Feel free to use and modify for your projects!

---

**Created with**: Claude Code on M1 MacBook Pro ⚡
**Date**: September 2025
**Purpose**: Demonstrating real-world M1 PyTorch optimization and monitoring