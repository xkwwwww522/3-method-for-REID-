#!/bin/bash
# CCVID v3: Clothes-aware triplet training
set -e

PYTHON=/root/miniconda3/envs/env4leng/bin/python
PROJ=/root/autodl-tmp/ylma/REID/third_party/CLIP-ReID
OUTDIR=/root/autodl-tmp/ylma/REID/output/ccvid_v3
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p $OUTDIR

echo "=== CCVID v3 (Clothes-Aware) Launch: $TIMESTAMP ===" | tee $OUTDIR/launch.log

pkill -9 -f "train_clipreid.py" 2>/dev/null || true
pkill -9 -f "monitor_gpu" 2>/dev/null || true
sleep 2

export CUDA_VISIBLE_DEVICES=0
cd $PROJ

nohup $PYTHON -u train_clipreid.py \
    --config_file configs/person/vit_clipreid_ccvid_v3.yml \
    > $OUTDIR/train_${TIMESTAMP}.log 2>&1 &
TRAIN_PID=$!
echo "TRAIN_PID=$TRAIN_PID" | tee $OUTDIR/pids.txt

sleep 8

if ! kill -0 $TRAIN_PID 2>/dev/null; then
    echo "FATAL: Training died!" | tee -a $OUTDIR/launch.log
    tail -40 $OUTDIR/train_${TIMESTAMP}.log | tee -a $OUTDIR/launch.log
    exit 1
fi

# Monitor
nohup bash -c '
PID='$TRAIN_PID'
LOG='$OUTDIR'/monitor_'$TIMESTAMP'.log
MAX=0; THR=25000
echo "$(date): Monitor $$ watching $PID" >> $LOG
while kill -0 $PID 2>/dev/null; do
    M=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ -n "$M" ] && { [ $M -gt $MAX ] && MAX=$M; echo "$(date): GPU=$M MiB" >> $LOG; [ $M -gt $THR ] && { echo "$(date): OOM! Kill." >> $LOG; kill -9 $PID; exit 1; }; }
    sleep 30
done
echo "$(date): Done. Peak=$MAX MiB" >> $LOG
' > /dev/null 2>&1 &
echo "MONITOR_PID=$!" >> $OUTDIR/pids.txt

echo "=== Training launched ===" | tee -a $OUTDIR/launch.log
echo "Log: $OUTDIR/train_${TIMESTAMP}.log" | tee -a $OUTDIR/launch.log
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader | tee -a $OUTDIR/launch.log
