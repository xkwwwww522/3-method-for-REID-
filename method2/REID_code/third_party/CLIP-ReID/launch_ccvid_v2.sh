#!/bin/bash
# CCVID v2 Training: random frame sampling + anti-overfitting config
set -e

PYTHON=/root/miniconda3/envs/env4leng/bin/python
PROJ=/root/autodl-tmp/ylma/REID/third_party/CLIP-ReID
OUTDIR=/root/autodl-tmp/ylma/REID/output/ccvid_v2
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p $OUTDIR

echo "=== CCVID v2 Training Launch: $TIMESTAMP ===" | tee $OUTDIR/launch.log

# Kill old processes
pkill -9 -f "train_clipreid.py" 2>/dev/null || true
pkill -9 -f "monitor_gpu" 2>/dev/null || true
sleep 2

# Clear GPU
$PYTHON -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

# Launch training (force single GPU to avoid DataParallel issues)
cd $PROJ
export CUDA_VISIBLE_DEVICES=0
nohup $PYTHON -u train_clipreid.py \
    --config_file configs/person/vit_clipreid_ccvid_v2.yml \
    > $OUTDIR/train_${TIMESTAMP}.log 2>&1 &
TRAIN_PID=$!
echo "TRAIN_PID=$TRAIN_PID" | tee $OUTDIR/pids.txt

# Wait for init
sleep 8

if ! kill -0 $TRAIN_PID 2>/dev/null; then
    echo "FATAL: Training failed!" | tee -a $OUTDIR/launch.log
    tail -30 $OUTDIR/train_${TIMESTAMP}.log | tee -a $OUTDIR/launch.log
    exit 1
fi

# GPU monitor
nohup bash -c '
TRAIN_PID='$TRAIN_PID'
LOG='$OUTDIR'/monitor_'$TIMESTAMP'.log
MAX_MEM=0
THRESHOLD=25000
echo "$(date): Monitor $$ watching $TRAIN_PID, threshold=$THRESHOLD MiB" >> $LOG
while kill -0 $TRAIN_PID 2>/dev/null; do
    MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    if [ -n "$MEM" ]; then
        if [ "$MEM" -gt "$MAX_MEM" ]; then MAX_MEM=$MEM; fi
        echo "$(date): GPU=$MEM MiB" >> $LOG
        if [ "$MEM" -gt "$THRESHOLD" ]; then
            echo "$(date): FATAL OOM! $MEM > $THRESHOLD. Killing." >> $LOG
            kill -9 $TRAIN_PID
            exit 1
        fi
    fi
    sleep 30
done
echo "$(date): Train $TRAIN_PID exited. Peak mem: $MAX_MEM MiB" >> $LOG
' > /dev/null 2>&1 &
MON_PID=$!
echo "MONITOR_PID=$MON_PID" >> $OUTDIR/pids.txt

echo "=== Training launched ===" | tee -a $OUTDIR/launch.log
echo "Train PID: $TRAIN_PID" | tee -a $OUTDIR/launch.log
echo "Log: $OUTDIR/train_${TIMESTAMP}.log" | tee -a $OUTDIR/launch.log
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader | tee -a $OUTDIR/launch.log
