import sys
sys.path.insert(0, '/root/autodl-tmp/ylma/REID/third_party/CLIP-ReID')
from config import cfg
cfg.merge_from_file('configs/person/vit_clipreid_ccvid_full.yml')
print('type:', type(cfg.DATASETS.ROOT_DIR))
print('repr:', repr(cfg.DATASETS.ROOT_DIR))
print('len:', len(cfg.DATASETS.ROOT_DIR))
print('[0]:', repr(cfg.DATASETS.ROOT_DIR[0]))
