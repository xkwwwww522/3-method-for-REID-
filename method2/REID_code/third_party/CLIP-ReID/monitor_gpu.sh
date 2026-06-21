#!/bin/bash
TRAIN_PID=
LOG=/root/autodl-tmp/ylma/REID/output/ccvid_fulltrain/monitor.log
THRESHOLD=25000
echo "Mon Jun  8 00:50:18     2026: Monitor PID 1484 watching train PID , threshold= MiB" >> 
while kill -0  2>/dev/null; do
  MEM=
  if [ -n "" ]; then
    echo "Mon Jun  8 00:50:19     2026: GPU= MiB" >> 
    if [ "" -gt "" ]; then
      echo "Mon Jun  8 00:50:19     2026: OOM!  > . Killing " >> 
      kill -9 
      exit 1
    fi
  fi
  sleep 30
done
echo "Mon Jun  8 00:50:19     2026: Train  exited" >> 
