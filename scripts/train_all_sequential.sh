#!/bin/bash
# Sequential Training Script for All RL Agents
# This script trains DQN, PPO, and SAC one after another

echo "================================================================================"
echo "SEQUENTIAL RL AGENT TRAINING"
echo "================================================================================"
echo "Start time: $(date)"
echo ""

# Create logs directory
mkdir -p logs
mkdir -p models

# 1. Train DQN
echo "================================================================================"
echo "STEP 1/3: Training DQN Agent (1000 episodes)"
echo "================================================================================"
python scripts/train_dqn.py \
  --data data/processed/dataset_with_regimes.csv \
  --episodes 1000 \
  --save models/dqn_trained.pth \
  --device cpu \
  2>&1 | tee logs/dqn_training.log

if [ $? -eq 0 ]; then
    echo "✅ DQN training completed successfully!"
else
    echo "❌ DQN training failed. Check logs/dqn_training.log"
    exit 1
fi

echo ""
echo "================================================================================"
echo "STEP 2/3: Training PPO Agent (500K timesteps)"
echo "================================================================================"
python scripts/train_ppo_optimized.py \
  --n-envs 8 \
  --total-timesteps 500000 \
  --learning-rate 3e-4 \
  --output-dir models/ppo \
  2>&1 | tee logs/ppo_training.log

if [ $? -eq 0 ]; then
    echo "✅ PPO training completed successfully!"
else
    echo "❌ PPO training failed. Check logs/ppo_training.log"
    exit 1
fi

echo ""
echo "================================================================================"
echo "STEP 3/3: Training SAC Agent (500K timesteps)"
echo "================================================================================"
python scripts/train_sac.py \
  --data data/processed/complete_dataset.csv \
  --total-timesteps 500000 \
  --save models/sac_trained.pth \
  2>&1 | tee logs/sac_training.log

if [ $? -eq 0 ]; then
    echo "✅ SAC training completed successfully!"
else
    echo "❌ SAC training failed. Check logs/sac_training.log"
    exit 1
fi

echo ""
echo "================================================================================"
echo "ALL TRAINING COMPLETE!"
echo "================================================================================"
echo "End time: $(date)"
echo ""
echo "Trained models:"
ls -lh models/*.pth models/*/*.pth 2>/dev/null
echo ""
echo "Next steps:"
echo "  1. Run: python scripts/monitor_training.py"
echo "  2. Create RL agent adapters"
echo "  3. Run comprehensive backtests"
echo "================================================================================"
