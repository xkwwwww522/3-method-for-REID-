"""Analyze CCVID data distribution and test improvements."""
import sys
sys.path.insert(0, '/root/autodl-tmp/ylma/REID/third_party/CLIP-ReID')
import numpy as np
from collections import defaultdict

# Load dataset
from datasets.ccvid import CCVID
ds = CCVID(root='/root/autodl-tmp/ylma/REID/data', verbose=True)

def analyze_split(name, data):
    frames_per_pid = defaultdict(int)
    for _, pid, _, _ in data:
        frames_per_pid[pid] += 1
    counts = sorted(frames_per_pid.values())
    print(f'\n{name}: {len(counts)} unique IDs, {len(data)} total frames')
    print(f'  Frames/ID: min={min(counts)}, max={max(counts)}, mean={np.mean(counts):.1f}, median={np.median(counts):.1f}')
    # Distribution
    bins = [2, 4, 8, 16, 32, 64, 128, 256]
    for b in bins:
        n = sum(1 for c in counts if c <= b)
        print(f'  <= {b:4d} frames: {n:3d} IDs ({n/len(counts)*100:.0f}%)')

analyze_split('Train', ds.train)
analyze_split('Query', ds.query)
analyze_split('Gallery', ds.gallery)

# Check clothes labels distribution
print('\n=== Clothes Distribution ===')
from collections import Counter
clothes_counter = Counter()
with open('/root/autodl-tmp/ylma/REID/data/CCVID_cope/train.txt') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 3:
            clothes_counter[parts[2]] += 1
print(f'Train: {len(clothes_counter)} unique clothes types, {clothes_counter.most_common(5)}')

# Count per-PID clothes changes
pid_clothes = defaultdict(set)
with open('/root/autodl-tmp/ylma/REID/data/CCVID_cope/train.txt') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 3:
            pid_clothes[int(parts[1])].add(parts[2])
multi_clothes = sum(1 for v in pid_clothes.values() if len(v) > 1)
print(f'PIDs with >1 clothes: {multi_clothes}/{len(pid_clothes)}')

print('\nDONE')
