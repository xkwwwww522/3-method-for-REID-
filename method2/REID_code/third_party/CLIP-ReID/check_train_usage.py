"""Check how many frames actually participate in CCVID training."""
import sys, os, glob, re
sys.path.insert(0, '/root/autodl-tmp/ylma/REID/third_party/CLIP-ReID')
import numpy as np
from collections import defaultdict
from datasets.ccvid import CCVID

ds = CCVID(root='/root/autodl-tmp/ylma/REID/data', verbose=False)
data_root = '/root/autodl-tmp/ylma/REID/data/CCVID_cope'

# Count available frames per tracklet
all_files = glob.glob(os.path.join(data_root, '**', '*.jpg'))
file_index = defaultdict(list)
for fp in all_files:
    match = re.match(r'(.+)_\d+\.jpg$', os.path.basename(fp))
    if match:
        file_index[match.group(1)].append(fp)

# Parse train.txt
with open(os.path.join(data_root, 'train.txt')) as f:
    train_lines = f.readlines()

available_frames = []
used_frames = []
for line in train_lines:
    line = line.strip()
    if not line: continue
    parts = line.split()
    if len(parts) < 2: continue
    prefix = parts[0]
    key = prefix.replace('/', '_')
    files = file_index.get(key, [])
    available_frames.append(len(files))

    # What our dataset uses: min(4, n_files), evenly spaced
    n_avail = len(files)
    n_take = min(4, n_avail)
    used_frames.append(n_take)

total_avail = sum(available_frames)
total_used = sum(used_frames)

print('='*60)
print('  CCVID Training Data Usage')
print('='*60)
print('')
print('Train.txt tracklets: %d' % len(train_lines))
print('75 unique IDs')
print('')
print('Frames available per tracklet:')
print('  min=%d, max=%d, mean=%.1f, median=%.1f' % (
    min(available_frames), max(available_frames),
    np.mean(available_frames), np.median(available_frames)))
print('')
print('Frames USED per tracklet (n_take=min(4, n)):')
print('  total used  = %d images (%.1f per tracklet)' % (total_used, np.mean(used_frames)))
print('')
print('='*60)
print('  SUMMARY')
print('='*60)
print('  Total available: %d images' % total_avail)
print('  Total used:      %d images' % total_used)
print('  Usage rate:      %.1f%%' % (total_used / total_avail * 100))
print('')
print('  That means %.1f%% of training images were NEVER seen!' % (100 - total_used/total_avail*100))
print('  %d images thrown away without being used in training' % (total_avail - total_used))
