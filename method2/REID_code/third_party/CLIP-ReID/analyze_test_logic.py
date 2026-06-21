"""Analyze CCVID test structure: tracklets, frames, and what actually gets used."""
import sys, os, glob
sys.path.insert(0, '/root/autodl-tmp/ylma/REID/third_party/CLIP-ReID')
import numpy as np
from collections import defaultdict
from datasets.ccvid import CCVID

ds = CCVID(root='/root/autodl-tmp/ylma/REID/data', verbose=True)

# ========================
# 1. Tracklet-level analysis
# ========================
# ds.query entries are (path, pid, camid, vid)
# Each entry = one frame from one tracklet

# Count how many tracklets per PID in query
q_tracklets = defaultdict(set)
for path, pid, camid, vid in ds.query:
    # Get tracklet prefix from path
    # path like: /root/.../CCVID_cope/gallery/session1_031_01_00001.jpg
    basename = os.path.basename(path)
    # session1_031_01_00001.jpg -> session1_031_01
    parts = basename.rsplit('_', 1)
    if len(parts) == 2:
        prefix = parts[0]
        q_tracklets[pid].add(prefix)

g_tracklets = defaultdict(set)
for path, pid, camid, vid in ds.gallery:
    basename = os.path.basename(path)
    parts = basename.rsplit('_', 1)
    if len(parts) == 2:
        prefix = parts[0]
        g_tracklets[pid].add(prefix)

# Count frames per PID
q_frames = defaultdict(int)
for path, pid, _, _ in ds.query:
    q_frames[pid] += 1
g_frames = defaultdict(int)
for path, pid, _, _ in ds.gallery:
    g_frames[pid] += 1

print('\n' + '='*70)
print('  CCVID Test Structure (Tracklet Level)')
print('='*70)

# Stats
q_tracklet_counts = [len(v) for v in q_tracklets.values()]
g_tracklet_counts = [len(v) for v in g_tracklets.values()]
q_frame_counts = [v for v in q_frames.values()]
g_frame_counts = [v for v in g_frames.values()]

print('\n--- Query ---')
print('Total PIDs: %d' % len(q_tracklets))
print('Total tracklets: %d' % sum(q_tracklet_counts))
print('Tracklets per PID: min=%d, max=%d, mean=%.1f, median=%.1f' % (
    min(q_tracklet_counts), max(q_tracklet_counts),
    np.mean(q_tracklet_counts), np.median(q_tracklet_counts)))
print('Total frames (after 4/p sampling): %d' % sum(q_frame_counts))
print('Frames per PID: min=%d, max=%d, mean=%.1f' % (
    min(q_frame_counts), max(q_frame_counts), np.mean(q_frame_counts)))

print('\n--- Gallery ---')
print('Total PIDs: %d' % len(g_tracklets))
print('Total tracklets: %d' % sum(g_tracklet_counts))
print('Tracklets per PID: min=%d, max=%d, mean=%.1f, median=%.1f' % (
    min(g_tracklet_counts), max(g_tracklet_counts),
    np.mean(g_tracklet_counts), np.median(g_tracklet_counts)))
print('Total frames (after 4/p sampling): %d' % sum(g_frame_counts))
print('Frames per PID: min=%d, max=%d, mean=%.1f' % (
    min(g_frame_counts), max(g_frame_counts), np.mean(g_frame_counts)))

# ========================
# 2. What our "4 frames" evaluation actually uses
# ========================
print('\n' + '='*70)
print('  What "N frames" Actually Means')
print('='*70)

# Example: PID with the most tracklets
max_q_pid = max(q_tracklets, key=lambda p: len(q_tracklets[p]))
print('\nExample: Query PID=%d' % max_q_pid)
print('  Tracklets (%d):' % len(q_tracklets[max_q_pid]))
for t in sorted(q_tracklets[max_q_pid])[:10]:
    print('    %s' % t)
print('  Total frames available: %d' % q_frames[max_q_pid])
print('')
print('  With "2 frames": we sample 2 frames from these %d frames -> average -> 1 feature' % q_frames[max_q_pid])
print('  With "4 frames": we sample 4 frames from these %d frames -> average -> 1 feature' % q_frames[max_q_pid])
print('  With "all (22)": we use all %d frames -> average -> 1 feature' % q_frames[max_q_pid])

# What percentage of frames actually participate?
for n in [2, 4, 8, 16, 0]:
    if n == 0:
        total_used = sum(q_frame_counts) + sum(g_frame_counts)
        label = 'all'
    else:
        total_used = n * len(q_frames) + n * len(g_frames)
        label = str(n)
    total_avail = sum(q_frame_counts) + sum(g_frame_counts)
    pct = total_used / total_avail * 100
    print('\n  %s frames: %d images used / %d available = %.1f%%' % (label, total_used, total_avail, pct))

# ========================
# 3. The effect: how many UNIQUE IMAGES are actually loaded and processed
# ========================
print('\n' + '='*70)
print('  Test Pipeline Breakdown')
print('='*70)
print('''
  1. CCVID._process_split(): reads query.txt (N_tracklets lines)
       For each tracklet: glob finds all frames -> sample 4 evenly spaced frames
       Result: ds.query = N_tracklets × 4 items

  2. load_frames(ds.query, max_n=N):
       Groups all frames by PID
       If PID has M frames across all its tracklets:
         - If M <= N: keep all M
         - If M > N: sample N evenly spaced
       Result: dict[pid] = list of N frame tensors

  3. pid_avg_pool():
       For each PID: extract features for all N frames -> average -> 1 feature vector
       Result: 151 feature vectors (one per query PID)

  4. Matching:
       151 query features × 151 gallery features = distance matrix
       eval_func computes mAP/R1/R5/R10

  KEY INSIGHT:
  - We treat each PID as one "probe" with ONE aggregated feature
  - All tracklets of the same PID are merged into a single feature
  - The evaluation is "person-level", not "tracklet-level"
  - CCVID typically evaluates at tracklet level (each tracklet is a separate query)
''')

print('DONE')
