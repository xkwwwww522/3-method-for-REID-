"""Quick CCVID dataset test - writes results to file."""
import sys, time
sys.path.insert(0, '/root/autodl-tmp/ylma/REID/third_party/CLIP-ReID')

with open('/tmp/ccvid_test_result.txt', 'w') as f:
    t0 = time.time()
    f.write('Testing CCVID dataset loading...\n')
    f.flush()

    from datasets.ccvid import CCVID
    ds = CCVID(root='/root/autodl-tmp/ylma/REID/data', verbose=True)

    f.write('Train: %d IDs, %d images, %d cams\n' % (ds.num_train_pids, ds.num_train_imgs, ds.num_train_cams))
    f.write('Query: %d IDs, %d images\n' % (ds.num_query_pids, ds.num_query_imgs))
    f.write('Gallery: %d IDs, %d images\n' % (ds.num_gallery_pids, ds.num_gallery_imgs))
    f.write('Time: %.1fs\n' % (time.time() - t0))
    f.write('DONE\n')
