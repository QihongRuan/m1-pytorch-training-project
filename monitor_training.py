#!/usr/bin/env python3
import time
import re
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.dates import DateFormatter
import numpy as np
from datetime import datetime, timedelta
import os
import psutil

class TrainingMonitor:
    def __init__(self):
        self.epochs = []
        self.batches = []
        self.losses = []
        self.train_accuracies = []
        self.test_accuracies = []
        self.times = []
        self.cpu_usage = []
        self.memory_usage = []
        self.start_time = datetime.now()
        
        # Set up the plots
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle('Real-time PyTorch Training Monitor - M1 MacBook Pro', fontsize=16)
        
        # Initialize empty lines
        self.loss_line, = self.ax1.plot([], [], 'b-', linewidth=2, label='Training Loss')
        self.acc_line, = self.ax2.plot([], [], 'g-', linewidth=2, label='Training Accuracy')
        self.test_acc_line, = self.ax2.plot([], [], 'r-', linewidth=2, label='Test Accuracy')
        self.cpu_line, = self.ax3.plot([], [], 'orange', linewidth=2, label='CPU Usage %')
        self.mem_line, = self.ax4.plot([], [], 'purple', linewidth=2, label='Memory Usage %')
        
        # Configure axes
        self.ax1.set_title('Training Loss Progress')
        self.ax1.set_xlabel('Time (minutes)')
        self.ax1.set_ylabel('Loss')
        self.ax1.grid(True, alpha=0.3)
        self.ax1.legend()
        
        self.ax2.set_title('Model Accuracy')
        self.ax2.set_xlabel('Epoch')
        self.ax2.set_ylabel('Accuracy %')
        self.ax2.grid(True, alpha=0.3)
        self.ax2.legend()
        
        self.ax3.set_title('CPU Usage (M1 - 8 cores)')
        self.ax3.set_xlabel('Time (minutes)')
        self.ax3.set_ylabel('CPU %')
        self.ax3.set_ylim(0, 100)
        self.ax3.grid(True, alpha=0.3)
        self.ax3.legend()
        
        self.ax4.set_title('Memory Usage (8GB RAM)')
        self.ax4.set_xlabel('Time (minutes)')
        self.ax4.set_ylabel('Memory %')
        self.ax4.set_ylim(0, 100)
        self.ax4.grid(True, alpha=0.3)
        self.ax4.legend()
        
        plt.tight_layout()
        
    def get_system_stats(self):
        """Get current system resource usage"""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        return cpu_percent, memory.percent
    
    def parse_training_output(self, output_text):
        """Parse training output for metrics"""
        current_time = (datetime.now() - self.start_time).total_seconds() / 60.0
        
        # Get system stats
        cpu_usage, mem_usage = self.get_system_stats()
        self.cpu_usage.append(cpu_usage)
        self.memory_usage.append(mem_usage)
        self.times.append(current_time)
        
        # Parse epoch completion
        epoch_pattern = r"Epoch (\d+)/\d+:"
        train_loss_pattern = r"Train Loss: ([\d.]+)"
        train_acc_pattern = r"Train Acc: ([\d.]+)%"
        test_acc_pattern = r"Test Acc: ([\d.]+)%"
        batch_pattern = r"Epoch (\d+), Batch (\d+), Loss: ([\d.]+)"
        
        lines = output_text.split('\n')
        
        for line in lines:
            # Parse batch updates
            batch_match = re.search(batch_pattern, line)
            if batch_match:
                epoch, batch, loss = batch_match.groups()
                self.epochs.append(int(epoch))
                self.batches.append(int(batch))
                self.losses.append(float(loss))
            
            # Parse epoch summaries
            if "Epoch" in line and "Train Loss:" in line:
                train_loss_match = re.search(train_loss_pattern, line)
                train_acc_match = re.search(train_acc_pattern, line)
                
                if train_loss_match and train_acc_match:
                    loss = float(train_loss_match.group(1))
                    acc = float(train_acc_match.group(1))
                    
                    # Find corresponding test accuracy
                    for next_line in lines[lines.index(line):]:
                        test_acc_match = re.search(test_acc_pattern, next_line)
                        if test_acc_match:
                            test_acc = float(test_acc_match.group(1))
                            self.test_accuracies.append(test_acc)
                            break
    
    def update_plots(self):
        """Update all plots with latest data"""
        if len(self.losses) > 0:
            # Update loss plot
            batch_times = np.linspace(0, self.times[-1] if self.times else 0, len(self.losses))
            self.loss_line.set_data(batch_times, self.losses)
            self.ax1.relim()
            self.ax1.autoscale_view()
        
        if len(self.test_accuracies) > 0:
            # Update accuracy plot
            epochs_range = range(1, len(self.test_accuracies) + 1)
            self.test_acc_line.set_data(epochs_range, self.test_accuracies)
            self.ax2.relim()
            self.ax2.autoscale_view()
        
        if len(self.cpu_usage) > 0:
            # Update CPU usage
            self.cpu_line.set_data(self.times, self.cpu_usage)
            self.ax3.relim()
            self.ax3.autoscale_view()
        
        if len(self.memory_usage) > 0:
            # Update memory usage
            self.mem_line.set_data(self.times, self.memory_usage)
            self.ax4.relim()
            self.ax4.autoscale_view()
        
        # Add current status text
        status_text = f"Runtime: {self.times[-1]:.1f}min | "
        if self.losses:
            status_text += f"Latest Loss: {self.losses[-1]:.4f} | "
        if self.test_accuracies:
            status_text += f"Test Acc: {self.test_accuracies[-1]:.1f}% | "
        if self.cpu_usage:
            status_text += f"CPU: {self.cpu_usage[-1]:.1f}% | "
        if self.memory_usage:
            status_text += f"RAM: {self.memory_usage[-1]:.1f}%"
        
        self.fig.suptitle(f'Real-time Training Monitor - {status_text}', fontsize=14)
        
        return [self.loss_line, self.test_acc_line, self.cpu_line, self.mem_line]
    
    def print_progress_summary(self):
        """Print text-based progress summary"""
        print("\n" + "="*80)
        print("🔥 REAL-TIME TRAINING PROGRESS SUMMARY 🔥")
        print("="*80)
        
        if self.times:
            print(f"⏱️  Runtime: {self.times[-1]:.1f} minutes")
        
        if self.epochs and self.batches:
            print(f"📊 Progress: Epoch {self.epochs[-1]}, Batch {self.batches[-1]}")
        
        if self.losses:
            print(f"📉 Current Loss: {self.losses[-1]:.4f}")
            if len(self.losses) > 10:
                recent_trend = "📈 Rising" if self.losses[-1] > self.losses[-5] else "📉 Falling"
                print(f"📈 Loss Trend: {recent_trend}")
        
        if self.test_accuracies:
            print(f"🎯 Test Accuracy: {self.test_accuracies[-1]:.2f}%")
            if len(self.test_accuracies) > 1:
                best_acc = max(self.test_accuracies)
                print(f"🏆 Best Accuracy: {best_acc:.2f}%")
        
        if self.cpu_usage:
            avg_cpu = np.mean(self.cpu_usage[-10:]) if len(self.cpu_usage) >= 10 else np.mean(self.cpu_usage)
            print(f"🖥️  CPU Usage: {self.cpu_usage[-1]:.1f}% (avg: {avg_cpu:.1f}%)")
        
        if self.memory_usage:
            print(f"🧠 Memory Usage: {self.memory_usage[-1]:.1f}%")
        
        # Performance estimation
        if len(self.epochs) > 0 and self.times:
            epochs_completed = len(set(self.epochs))
            if epochs_completed > 0:
                time_per_epoch = self.times[-1] / epochs_completed
                remaining_epochs = 15 - epochs_completed
                eta_minutes = remaining_epochs * time_per_epoch
                print(f"⏰ ETA: ~{eta_minutes:.1f} minutes remaining ({remaining_epochs} epochs left)")
        
        print("="*80)

def monitor_training():
    """Main monitoring function"""
    monitor = TrainingMonitor()
    
    print("🚀 Starting Real-time Training Monitor for M1 MacBook Pro")
    print("📊 Monitoring: Loss, Accuracy, CPU Usage, Memory Usage")
    print("⌨️  Press Ctrl+C to stop monitoring")
    print("-" * 60)
    
    try:
        update_count = 0
        while True:
            # This would normally read from a log file or API
            # For demo, we'll simulate with system monitoring
            output_text = f"Simulated training output at {datetime.now()}"
            
            # Parse any new training data
            monitor.parse_training_output(output_text)
            
            # Update plots every few iterations
            if update_count % 5 == 0:
                monitor.update_plots()
                plt.pause(0.1)
            
            # Print progress summary every 30 seconds
            if update_count % 30 == 0:
                monitor.print_progress_summary()
            
            update_count += 1
            time.sleep(1)  # Update every second
            
    except KeyboardInterrupt:
        print("\n\n🛑 Monitoring stopped by user")
        print("💾 Saving final plots...")
        plt.savefig('real_time_training_monitor.png', dpi=150, bbox_inches='tight')
        print("📊 Final monitoring plots saved to 'real_time_training_monitor.png'")
        plt.show()

if __name__ == "__main__":
    monitor_training()