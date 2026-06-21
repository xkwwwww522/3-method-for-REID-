import sys, time, glob, os
sys.path.insert(0, '/root/autodl-tmp/ylma/REID/third_party/CLIP-ReID')
t0 = time.time()
print('Testing CCVID dataset loading...')
from datasets.ccvid import CCVID
ds = CCVID(root='/root/autodl-tmp/ylma/REID/data', verbose=True)
print('Train: %d IDs, %d images' % (ds.num_train_pids, ds.num_train_imgs))
print('Query: %d IDs, %d images' % (ds.num_query_pids, ds.num_query_imgs))
print('Gallery: %d IDs, %d images' % (ds.num_gallery_pids, ds.num_gallery_imgs))
print('Time: %.1fs' % (time.time() - t0))
print('DONE')
