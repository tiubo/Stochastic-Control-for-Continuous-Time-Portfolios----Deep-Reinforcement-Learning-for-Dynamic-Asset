"""
Simple Training Monitor

Check training progress and status.
"""

import os
import sys
from pathlib import Path
import time

def check_log_file(log_path):
    """Check and display last lines of log file."""
    if not os.path.exists(log_path):
        return None

    with open(log_path, 'r') as f:
        lines = f.readlines()
        if lines:
            return lines[-5:]  # Last 5 lines
    return None

def check_models():
    """Check if model files exist."""
    models = {
        'DQN': 'models/dqn_trained.pth',
        'PPO': 'models/ppo/ppo_final.pth',
        'SAC': 'models/sac_trained.pth'
    }

    status = {}
    for name, path in models.items():
        if os.path.exists(path):
            size = os.path.getsize(path) / (1024 * 1024)  # MB
            status[name] = f"✅ Complete ({size:.2f} MB)"
        else:
            status[name] = "⏳ In Progress or Pending"

    return status

def main():
    """Main monitoring function."""
    print("="*70)
    print("RL AGENT TRAINING MONITOR")
    print("="*70)

    # Check logs
    print("\n📊 TRAINING LOGS:")
    print("-"*70)

    logs = {
        'DQN': 'logs/dqn_training.log',
        'PPO': 'logs/ppo_training.log',
        'SAC': 'logs/sac_training.log'
    }

    for name, log_path in logs.items():
        print(f"\n{name}:")
        lines = check_log_file(log_path)
        if lines:
            for line in lines:
                print(f"  {line.rstrip()}")
        else:
            print("  ⏳ Not started yet")

    # Check models
    print("\n\n📦 MODEL STATUS:")
    print("-"*70)

    model_status = check_models()
    for name, status in model_status.items():
        print(f"{name:10s}: {status}")

    print("\n" + "="*70)
    print("Tip: Run 'python scripts/monitor_training.py' to check progress")
    print("="*70)

if __name__ == "__main__":
    main()
