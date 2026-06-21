#!/bin/bash
# CCVID Full-Parameter Training Launcher with OOM Guard
set -e

PYTHON=/root/miniconda3/envs/env4leng/bin/python
PROJ=/root/autodl-tmp/ylma/REID/third_party/CLIP-ReID
OUTDIR=/root/autodl-tmp/ylma/REID/output/ccvid_fulltrain
PIDFILE=$OUTDIR/pids.txt
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p $OUTDIR

echo "=== CCVID Full Training Launch: $TIMESTAMP ===" | tee $OUTDIR/launch.log

# Kill old processes
pkill -9 -f "train_clipreid.py" 2>/dev/null || true
pkill -9 -f "monitor_gpu" 2>/dev/null || true
sleep 2

# Clear GPU cache
$PYTHON -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

# Launch training
cd $PROJ
nohup $PYTHON train_clipreid.py \
    --config_file configs/person/vit_clipreid_ccvid_full.yml \
    > $OUTDIR/train_${TIMESTAMP}.log 2>&1 &
TRAIN_PID=$!
echo "TRAIN_PID=$TRAIN_PID" | tee -a $OUTDIR/launch.log
echo $TRAIN_PID > $PIDFILE

# Wait for training to initialize
sleep 5

# Check if training is actually running
if ! kill -0 $TRAIN_PID 2>/dev/null; then
    echo "FATAL: Training failed to start!" | tee -a $OUTDIR/launch.log
    echo "Last log lines:" | tee -a $OUTDIR/launch.log
    tail -30 $OUTDIR/train_${TIMESTAMP}.log | tee -a $OUTDIR/launch.log
    exit 1
fi

# Launch GPU monitor
nohup bash -c '
TRAIN_PID='"$TRAIN_PID"'
LOG='"$OUTDIR"'/monitor_'"$TIMESTAMP"'.log
THRESHOLD=25000
echo "$(date): Monitor PID $$ watching train PID $TRAIN_PID, threshold=$THRESHOLD MiB" >> $LOG
while kill -0 $TRAIN_PID 2>/dev/null; do
    MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    if [ -n "$MEM" ]; then
        echo "$(date): GPU=$MEM MiB" >> $LOG
        if [ "$MEM" -gt "$THRESHOLD" ]; then
            echo "$(date): FATAL OOM! GPU memory $MEM > $THRESHOLD MiB. Killing training PID $TRAIN_PID" >> $LOG
            kill -9 $TRAIN_PID
            exit 1
        fi
    fi
    sleep 30
done
echo "$(date): Training process $TRAIN_PID has exited" >> $LOG
' > /dev/null 2>&1 &
MON_PID=$!
echo "MONITOR_PID=$MON_PID" | tee -a $OUTDIR/launch.log
echo $MON_PID >> $PIDFILE

echo "=== Training launched successfully ===" | tee -a $OUTDIR/launch.log
echo "Train PID: $TRAIN_PID" | tee -a $OUTDIR/launch.log
echo "Monitor PID: $MON_PID" | tee -a $OUTDIR/launch.log
echo "Train log: $OUTDIR/train_${TIMESTAMP}.log" | tee -a $OUTDIR/launch.log
echo "Monitor log: $OUTDIR/monitor_${TIMESTAMP}.log" | tee -a $OUTDIR/launch.log

# Show initial GPU state
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader | tee -a $OUTDIR/launch.log
