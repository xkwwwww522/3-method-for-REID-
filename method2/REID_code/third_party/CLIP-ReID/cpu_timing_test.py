"""Quick timing test: pixel averaging on CPU."""
import time, sys, os, glob, re
sys.path.insert(0, '/root/autodl-tmp/ylma/REID/third_party/CLIP-ReID')
from PIL import Image
import torchvision.transforms as T
import torch
from collections import defaultdict

tf = T.Compose([T.Resize((256, 128)), T.ToTensor(), T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])])
root = '/root/autodl-tmp/ylma/REID/data/CCVID_cope'

# Build index
all_files = glob.glob(os.path.join(root, '**', '*.jpg'))
fi = defaultdict(list)
for fp in all_files:
    m = re.match(r'(.+)_\d+\.jpg$', os.path.basename(fp))
    if m: fi[m.group(1)].append(fp)
print('Indexed %d files' % len(all_files))

# Load one tracklet
with open(os.path.join(root, 'query.txt')) as f:
    first_line = f.readline().strip().split()
    key = first_line[0].replace('/', '_')

files = fi.get(key, [])
n_test = min(50, len(files))
print('Testing with %d of %d frames' % (n_test, len(files)))

# Time image loading + transform
t0 = time.time()
frames = [tf(Image.open(f).convert('RGB')) for f in files[:n_test]]
avg = torch.stack(frames, dim=0).mean(dim=0)
t_per_10 = (time.time() - t0) / n_test * 10

print('Time for %d frames: %.1fs (%.0fms/frame)' % (n_test, time.time()-t0, (time.time()-t0)/n_test*1000))

# Estimate
total_query = 116799
total_gallery = 112421
total = total_query + total_gallery
hours_pixel_avg = total * (time.time()-t0) / n_test / 3600
print('\n=== Estimates ===')
print('Pixel averaging 229K images: %.1f hours' % hours_pixel_avg)
print('Then 12 checkpoints x 1908 forward passes on CPU: ~%d hours' % (12 * 1908 * 2 / 3600))
print('Total CPU estimate: %.1f hours' % (hours_pixel_avg + 12 * 1908 * 2 / 3600))
print('\nRECOMMENDATION: Wait for GPU recovery. This is infeasible on CPU.')
