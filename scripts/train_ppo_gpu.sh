#!/bin/bash
# PPO GPU Training Script
# Optimized for NVIDIA V100 GPU (AWS p3.2xlarge or GCP n1-standard-8)

set -e

echo "===================================================================="
echo "PPO TRAINING ON GPU"
echo "===================================================================="
echo ""

# Verify CUDA availability
echo "[1/5] Verifying CUDA..."
python -c "import torch; assert torch.cuda.is_available(), 'ERROR: CUDA not available!'; print(f'✓ GPU Available: {torch.cuda.get_device_name(0)}')"

# Create directories
echo ""
echo "[2/5] Creating directories..."
mkdir -p models/ppo logs
echo "✓ Directories created"

# Display training parameters
echo ""
echo "[3/5] Training Parameters:"
echo "  - Total Timesteps: 100,000"
echo "  - N-Steps: 2,048"
echo "  - Learning Rate: 3e-4"
echo "  - Device: CUDA"
echo "  - Estimated Time: 1-2 hours on V100"
echo ""

# Start training
echo "[4/5] Starting PPO training..."
echo "  Log file: logs/ppo_training_gpu_$(date +%Y%m%d_%H%M%S).log"
echo "  Progress will be displayed below..."
echo "===================================================================="
echo ""

python scripts/train_ppo_optimized.py \
    --data-path data/processed/dataset_with_regimes.csv \
    --total-timesteps 100000 \
    --n-steps 2048 \
    --learning-rate 3e-4 \
    --output-dir models/ppo \
    --device cuda \
    2>&1 | tee logs/ppo_training_gpu_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "===================================================================="
echo "[5/5] PPO training complete!"
echo "  ✓ Model saved to: models/ppo/ppo_final.pth"
echo "  ✓ Training log: logs/ppo_training_gpu_*.log"
echo "===================================================================="
