# GPU Migration Guide for SAC/PPO Training

Complete guide for migrating training to cloud GPU instances for 100x speedup.

---

## Quick Summary

**Problem**: SAC training on CPU takes ~38 hours (current: 1.8% complete @ 67 it/s)
**Solution**: Migrate to GPU for 2-4 hour completion (100x speedup)
**Cost**: ~$3-10 for complete training (2-4 hours @ $0.90-3.06/hour)

---

## Option 1: AWS EC2 GPU Instance (Recommended)

### Step 1: Launch GPU Instance

```bash
# Instance Type: p3.2xlarge
# - 1x NVIDIA V100 GPU (16GB)
# - 8 vCPUs, 61 GB RAM
# - Cost: ~$3.06/hour (On-Demand) or ~$0.90/hour (Spot)
# - Region: us-east-1 (cheapest)

# Launch via AWS CLI
aws ec2 run-instances \
    --image-id ami-0c55b159cbfafe1f0 \  # Deep Learning AMI (Ubuntu 20.04)
    --instance-type p3.2xlarge \
    --key-name your-key-pair \
    --security-group-ids sg-xxxxxxxxx \
    --subnet-id subnet-xxxxxxxxx \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100}}]' \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=RL-Training-GPU}]'
```

**Or via AWS Console:**
1. Go to EC2 → Launch Instance
2. Choose "Deep Learning AMI GPU PyTorch" (Ubuntu 20.04)
3. Instance type: `p3.2xlarge`
4. Storage: 100 GB
5. Security group: Allow SSH (port 22) from your IP
6. Launch with your key pair

### Step 2: Connect to Instance

```bash
# Get instance public IP from AWS Console or CLI
aws ec2 describe-instances --filters "Name=tag:Name,Values=RL-Training-GPU" \
    --query 'Reservations[*].Instances[*].[PublicIpAddress]' --output text

# SSH into instance
ssh -i ~/path/to/your-key.pem ubuntu@<INSTANCE_PUBLIC_IP>
```

### Step 3: Setup Environment

```bash
# Activate PyTorch environment (pre-installed on Deep Learning AMI)
source activate pytorch

# Verify GPU
nvidia-smi

# Clone repository
git clone https://github.com/mohin-io/Stochastic-Control-for-Continuous-Time-Portfolios--Deep-Reinforcement-Learning-for-Dynamic-Asset.git
cd Stochastic-Control-for-Continuous-Time-Portfolios--Deep-Reinforcement-Learning-for-Dynamic-Asset

# Install dependencies
pip install -r requirements.txt

# Verify CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0)}')"
```

### Step 4: Train SAC on GPU

```bash
# SAC Training (200k timesteps, ~2-3 hours on GPU)
python scripts/train_sac.py \
    --data-path data/processed/dataset_with_regimes.csv \
    --total-timesteps 200000 \
    --eval-freq 10000 \
    --save-freq 20000 \
    --model-save-path models/sac_trained_gpu.pth \
    --device cuda \
    2>&1 | tee logs/sac_training_gpu.log &

# Monitor progress
tail -f logs/sac_training_gpu.log

# Or use tmux for persistent session
tmux new -s training
python scripts/train_sac.py --device cuda ...
# Ctrl+B, D to detach
# tmux attach -t training  (to reattach)
```

### Step 5: Train PPO on GPU

```bash
# PPO Training (100k timesteps, ~1-2 hours on GPU)
python scripts/train_ppo_optimized.py \
    --data-path data/processed/dataset_with_regimes.csv \
    --total-timesteps 100000 \
    --n-steps 2048 \
    --learning-rate 3e-4 \
    --output-dir models/ppo \
    --device cuda \
    2>&1 | tee logs/ppo_training_gpu.log &
```

### Step 6: Download Trained Models

```bash
# From your local machine
scp -i ~/path/to/your-key.pem \
    ubuntu@<INSTANCE_PUBLIC_IP>:/path/to/models/sac_trained_gpu.pth \
    ./models/

scp -i ~/path/to/your-key.pem \
    ubuntu@<INSTANCE_PUBLIC_IP>:/path/to/models/ppo/ppo_final.pth \
    ./models/
```

### Step 7: Terminate Instance

```bash
# IMPORTANT: Terminate instance to stop charges
aws ec2 terminate-instances --instance-ids <INSTANCE_ID>

# Or via Console: EC2 → Instances → Right-click → Terminate
```

---

## Option 2: Google Cloud Platform (GCP)

### Step 1: Create GPU Instance

```bash
# Instance Type: n1-standard-8 with 1x NVIDIA V100
# Cost: ~$2.48/hour (On-Demand) or ~$0.74/hour (Preemptible)

gcloud compute instances create rl-training-gpu \
    --zone=us-central1-a \
    --machine-type=n1-standard-8 \
    --accelerator=type=nvidia-tesla-v100,count=1 \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=100GB \
    --maintenance-policy=TERMINATE \
    --metadata="install-nvidia-driver=True"
```

### Step 2: Connect & Setup

```bash
# SSH
gcloud compute ssh rl-training-gpu --zone=us-central1-a

# Verify GPU
nvidia-smi

# Clone & setup (same as AWS steps 3-5)
git clone https://github.com/mohin-io/...
cd Stochastic-Control-for-Continuous-Time-Portfolios...
pip install -r requirements.txt

# Train
python scripts/train_sac.py --device cuda ...
python scripts/train_ppo_optimized.py --device cuda ...
```

### Step 3: Download & Terminate

```bash
# Download models
gcloud compute scp rl-training-gpu:/path/to/models/* ./models/ \
    --zone=us-central1-a

# Terminate
gcloud compute instances delete rl-training-gpu --zone=us-central1-a
```

---

## Option 3: Google Colab (Free GPU)

### Limitations
- Session timeout after 12 hours
- May disconnect randomly
- T4 GPU (slower than V100)
- No persistent storage

### Setup

1. **Open Colab**: https://colab.research.google.com/
2. **Enable GPU**: Runtime → Change runtime type → GPU → T4
3. **Clone Repository**:

```python
!git clone https://github.com/mohin-io/Stochastic-Control-for-Continuous-Time-Portfolios--Deep-Reinforcement-Learning-for-Dynamic-Asset.git
%cd Stochastic-Control-for-Continuous-Time-Portfolios--Deep-Reinforcement-Learning-for-Dynamic-Asset
!pip install -r requirements.txt
```

4. **Train**:

```python
# SAC Training
!python scripts/train_sac.py \
    --data-path data/processed/dataset_with_regimes.csv \
    --total-timesteps 200000 \
    --device cuda

# Download model from Colab
from google.colab import files
files.download('models/sac_trained.pth')
```

---

## Cost Comparison

| Platform | Instance Type | GPU | Cost/Hour | Est. Total (4h) | Spot/Preemptible |
|----------|---------------|-----|-----------|-----------------|------------------|
| **AWS** | p3.2xlarge | V100 | $3.06 | $12.24 | $0.90/h ($3.60) |
| **GCP** | n1-std-8 + V100 | V100 | $2.48 | $9.92 | $0.74/h ($2.96) |
| **Colab** | Free Tier | T4 | $0 | $0 | N/A |
| **Colab Pro** | Pro Tier | V100/A100 | $10/mo | ~$0 | N/A |

**Recommendation**: GCP Preemptible (~$3 total) or AWS Spot (~$3.60 total)

---

## Performance Expectations

### CPU vs GPU Speedup

| Metric | CPU (Current) | GPU (Expected) | Speedup |
|--------|---------------|----------------|---------|
| **SAC Training** | ~38 hours | 2-3 hours | 12-19x |
| **Iterations/sec** | 67 it/s | 800-1200 it/s | 12-18x |
| **PPO Training** | ~20 hours | 1-2 hours | 10-20x |
| **Total Time** | ~58 hours | ~4 hours | **14.5x** |

### Timeline

```
AWS/GCP GPU Instance (4 hours total):
├── Setup & clone        (15 min)
├── SAC training         (2-3 hours)
├── PPO training         (1-2 hours)
└── Download models      (5 min)
```

---

## Training Scripts (GPU-Optimized)

### `scripts/train_sac_gpu.sh`

```bash
#!/bin/bash
# SAC GPU Training Script

set -e

echo "===================================================================="
echo "SAC TRAINING ON GPU"
echo "===================================================================="

# Verify CUDA
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print(f'Using GPU: {torch.cuda.get_device_name(0)}')"

# Create directories
mkdir -p models logs

# Train SAC
python scripts/train_sac.py \
    --data-path data/processed/dataset_with_regimes.csv \
    --total-timesteps 200000 \
    --eval-freq 5000 \
    --save-freq 10000 \
    --model-save-path models/sac_trained_gpu.pth \
    --device cuda \
    2>&1 | tee logs/sac_training_gpu_$(date +%Y%m%d_%H%M%S).log

echo "SAC training complete!"
echo "Model saved to: models/sac_trained_gpu.pth"
```

### `scripts/train_ppo_gpu.sh`

```bash
#!/bin/bash
# PPO GPU Training Script

set -e

echo "===================================================================="
echo "PPO TRAINING ON GPU"
echo "===================================================================="

# Verify CUDA
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'"

# Train PPO
python scripts/train_ppo_optimized.py \
    --data-path data/processed/dataset_with_regimes.csv \
    --total-timesteps 100000 \
    --n-steps 2048 \
    --learning-rate 3e-4 \
    --output-dir models/ppo \
    --device cuda \
    2>&1 | tee logs/ppo_training_gpu_$(date +%Y%m%d_%H%M%S).log

echo "PPO training complete!"
```

### `scripts/train_all_gpu.sh`

```bash
#!/bin/bash
# Train all algorithms on GPU sequentially

set -e

echo "Training all RL algorithms on GPU..."
echo "Estimated time: 4 hours"

# SAC
echo "[1/2] Training SAC..."
bash scripts/train_sac_gpu.sh

# PPO
echo "[2/2] Training PPO..."
bash scripts/train_ppo_gpu.sh

echo "All training complete!"
echo "Models:"
echo "  - models/sac_trained_gpu.pth"
echo "  - models/ppo/ppo_final.pth"
```

---

## Troubleshooting

### Issue 1: CUDA Out of Memory

```bash
# Reduce batch size in scripts/train_sac.py
# Line ~60: batch_size=64 → batch_size=32
```

### Issue 2: SSH Connection Timeout

```bash
# Use tmux for persistent sessions
tmux new -s training
# Run training commands
# Ctrl+B, D to detach
# tmux attach -t training to reattach
```

### Issue 3: Spot Instance Interruption (AWS)

```bash
# Enable checkpoint saving
# SAC automatically saves every 20k steps (--save-freq 20000)
# To resume: Load checkpoint and continue training
```

### Issue 4: Data Download Failed

```bash
# Data is in repository, but if missing:
cd data/processed
# Ensure dataset_with_regimes.csv exists
ls -lh
```

---

## Monitoring Training

### Real-time Progress

```bash
# Watch GPU utilization
watch -n 1 nvidia-smi

# Monitor training log
tail -f logs/sac_training_gpu.log

# Check training speed
grep "it/s" logs/sac_training_gpu.log | tail -20
```

### TensorBoard (Optional)

```python
# Add to training script:
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/sac_experiment')
writer.add_scalar('Loss/train', loss, global_step)
```

```bash
# Launch TensorBoard
tensorboard --logdir=runs --port=6006 --bind_all
# Access: http://<INSTANCE_IP>:6006
```

---

## Post-Training: Model Comparison

After GPU training completes, run comprehensive comparison:

```bash
# Compare DQN vs SAC vs PPO vs Baselines
python scripts/compare_all_agents.py \
    --dqn-model models/dqn_trained_ep1000.pth \
    --sac-model models/sac_trained_gpu.pth \
    --ppo-model models/ppo/ppo_final.pth \
    --data data/processed/dataset_with_regimes.csv \
    --output-dir simulations/final_comparison

# Generate comparison dashboard
python scripts/tier2_improvements.py
```

---

## Summary Checklist

- [ ] Choose platform (AWS, GCP, or Colab)
- [ ] Launch GPU instance (p3.2xlarge or n1-standard-8 + V100)
- [ ] SSH and verify GPU with `nvidia-smi`
- [ ] Clone repository
- [ ] Install dependencies (`pip install -r requirements.txt`)
- [ ] Train SAC (~2-3 hours): `python scripts/train_sac.py --device cuda`
- [ ] Train PPO (~1-2 hours): `python scripts/train_ppo_optimized.py --device cuda`
- [ ] Download trained models to local machine
- [ ] **Terminate instance** to stop charges
- [ ] Run comparison analysis locally
- [ ] Update SESSION_STATUS.md with results

---

## Estimated Total Cost

**Cheapest Option** (GCP Preemptible): **~$3.00**
- Setup: 15 min
- SAC: 2.5 hours
- PPO: 1.5 hours
- **Total**: 4 hours @ $0.74/hour = **$2.96**

**Most Reliable** (AWS On-Demand): **~$12.00**
- 4 hours @ $3.06/hour = **$12.24**

**Free Option** (Colab): **$0** (with limitations)

---

## Quick Start Command (Copy-Paste)

```bash
# AWS
ssh -i your-key.pem ubuntu@<IP>
source activate pytorch
git clone https://github.com/mohin-io/Stochastic-Control-for-Continuous-Time-Portfolios--Deep-Reinforcement-Learning-for-Dynamic-Asset.git
cd Stochastic-Control-for-Continuous-Time-Portfolios--Deep-Reinforcement-Learning-for-Dynamic-Asset
pip install -r requirements.txt
python scripts/train_sac.py --device cuda 2>&1 | tee logs/sac_gpu.log &
python scripts/train_ppo_optimized.py --device cuda 2>&1 | tee logs/ppo_gpu.log &
```

---

**Need Help?** Check SESSION_STATUS.md for current project status and performance benchmarks.
