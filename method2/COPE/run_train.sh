#!/bin/bash
set -e
source /base/mambaforge/etc/profile.d/conda.sh
conda activate /root/shared-nvme/reid_env
cd /root/shared-nvme/REID/COPE

echo "=== CCVID COPE Training (dummy masks) ==="
echo "Start: $(date)"
echo "Batch: 32 | Instances: 2 | AMP: ON"
echo ""

# Monitor VRAM in background
(
    while true; do
        mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null)
        echo "[VRAM $(date +%H:%M:%S)] $mem"
        sleep 30
    done
) &
MONITOR_PID=$!

# Run training
python train_cope.py --config_file configs/CCVID_train/cope.yml 2>&1
TRAIN_EXIT=$?

kill $MONITOR_PID 2>/dev/null
echo ""
echo "Training finished with exit code: $TRAIN_EXIT at $(date)"
