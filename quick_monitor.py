#!/usr/bin/env python3
import time
import psutil
import subprocess
import os

def get_process_stats():
    """Get CPU and memory usage of the training process"""
    try:
        result = subprocess.run(['pgrep', '-f', 'm1_optimized_real_world.py'], 
                              capture_output=True, text=True)
        if result.stdout.strip():
            pid = int(result.stdout.strip().split('\n')[0])
            process = psutil.Process(pid)
            cpu_percent = process.cpu_percent(interval=1)
            memory_mb = process.memory_info().rss / 1024 / 1024
            memory_percent = process.memory_percent()
            return pid, cpu_percent, memory_mb, memory_percent, True
    except:
        pass
    return None, 0, 0, 0, False

def monitor_system():
    """Monitor overall system performance"""
    cpu_overall = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    return cpu_overall, memory.percent

start_time = time.time()
print("🚀 Real-time PyTorch Training Monitor")
print("📊 Tracking: Process CPU, Memory, System Performance")
print("=" * 60)

iteration = 0
while True:
    try:
        # Get process stats
        pid, proc_cpu, proc_mem_mb, proc_mem_pct, is_running = get_process_stats()
        
        # Get system stats
        sys_cpu, sys_mem = monitor_system()
        
        # Calculate runtime
        runtime_min = (time.time() - start_time) / 60
        
        # Clear screen and show updated stats
        os.system('clear')
        
        print("🔥 M1 MacBook Pro - PyTorch Training Monitor 🔥")
        print("=" * 60)
        print(f"⏱️  Runtime: {runtime_min:.1f} minutes")
        print(f"📊 Update #{iteration + 1}")
        print()
        
        if is_running:
            print("✅ TRAINING STATUS: ACTIVE")
            print(f"🆔 Process ID: {pid}")
            print(f"🖥️  Process CPU: {proc_cpu:.1f}% (M1 optimized)")
            print(f"🧠 Process Memory: {proc_mem_mb:.1f} MB ({proc_mem_pct:.1f}%)")
        else:
            print("❌ TRAINING STATUS: NOT RUNNING")
        
        print()
        print("📈 SYSTEM PERFORMANCE:")
        print(f"🔧 Overall CPU: {sys_cpu:.1f}%")
        print(f"💾 Overall Memory: {sys_mem:.1f}%")
        
        # Progress estimation
        if runtime_min > 0:
            # Rough estimate: 15 epochs should take ~15-20 minutes
            estimated_total = 18  # minutes
            progress_pct = min((runtime_min / estimated_total) * 100, 100)
            progress_bar = "█" * int(progress_pct / 5) + "░" * (20 - int(progress_pct / 5))
            print()
            print(f"⏰ Estimated Progress: [{progress_bar}] {progress_pct:.1f}%")
            
            if progress_pct < 100:
                eta = estimated_total - runtime_min
                print(f"🎯 ETA: ~{eta:.1f} minutes remaining")
        
        print()
        print("💡 Your M1 MacBook Pro is working hard on real AI training!")
        print("🎯 Training CIFAR-10 image classification (1.28M parameters)")
        print("⌨️  Press Ctrl+C to stop monitoring")
        print("=" * 60)
        
        time.sleep(5)  # Update every 5 seconds
        iteration += 1
        
    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped")
        break