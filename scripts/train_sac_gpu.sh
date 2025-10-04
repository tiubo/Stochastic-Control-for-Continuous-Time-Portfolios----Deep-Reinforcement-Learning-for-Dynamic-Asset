#!/bin/bash
# SAC GPU Training Script
# Optimized for NVIDIA V100 GPU (AWS p3.2xlarge or GCP n1-standard-8)

set -e

echo "===================================================================="
echo "SAC TRAINING ON GPU"
echo "===================================================================="
echo ""

# Verify CUDA availability
echo "[1/5] Verifying CUDA..."
python -c "import torch; assert torch.cuda.is_available(), 'ERROR: CUDA not available!'; print(f'✓ GPU Available: {torch.cuda.get_device_name(0)}'); print(f'✓ CUDA Version: {torch.version.cuda}')"

# Create directories
echo ""
echo "[2/5] Creating directories..."
mkdir -p models logs
echo "✓ Directories created"

# Display training parameters
echo ""
echo "[3/5] Training Parameters:"
echo "  - Total Timesteps: 200,000"
echo "  - Evaluation Frequency: 5,000 steps"
echo "  - Save Frequency: 10,000 steps"
echo "  - Device: CUDA"
echo "  - Estimated Time: 2-3 hours on V100"
echo ""

# Start training
echo "[4/5] Starting SAC training..."
echo "  Log file: logs/sac_training_gpu_$(date +%Y%m%d_%H%M%S).log"
echo "  Progress will be displayed below..."
echo "===================================================================="
echo ""

python scripts/train_sac.py \
    --data-path data/processed/dataset_with_regimes.csv \
    --total-timesteps 200000 \
    --eval-freq 5000 \
    --save-freq 10000 \
    --model-save-path models/sac_trained_gpu.pth \
    --device cuda \
    2>&1 | tee logs/sac_training_gpu_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "===================================================================="
echo "[5/5] SAC training complete!"
echo "  ✓ Model saved to: models/sac_trained_gpu.pth"
echo "  ✓ Training log: logs/sac_training_gpu_*.log"
echo "===================================================================="
