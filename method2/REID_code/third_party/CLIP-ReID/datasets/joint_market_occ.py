# encoding: utf-8
"""Joint dataset combining Market1501 + Occluded_Duke for occlusion-robust training."""
import glob, re, os.path as osp
from .bases import BaseImageDataset
from .market1501 import Market1501
from .occ_duke import OCC_DukeMTMCreID

class JointMarketOccDuke(BaseImageDataset):
    """Combines Market1501 (clean) + Occluded_Duke (real occlusion) for training.
    Uses Market1501 query/gallery for validation (market1501 test set)."""
    dataset_dir = ''

    def __init__(self, root='', verbose=True, **kwargs):
        super().__init__()
        market_root = osp.join(root, '')
        occ_root = osp.join(root, '')

        # Load Market1501
        self.market = Market1501(root=market_root, verbose=False)
        # Load Occluded_Duke
        self.occ_duke = OCC_DukeMTMCreID(root=occ_root, verbose=False)

        # Combine training data: Market IDs 0..750, OccDuke IDs 751..751+occ_count-1
        occ_pid_offset = self.market.num_train_pids
        occ_train = []
        for img_path, pid, camid, trackid in self.occ_duke.train:
            occ_train.append((img_path, occ_pid_offset + pid, camid, trackid))

        self.train = self.market.train + occ_train

        # Use Market1501 query/gallery for validation
        self.query = self.market.query
        self.gallery = self.market.gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids =             self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids =             self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids =             self.get_imagedata_info(self.gallery)

        if verbose:
            print("=> Joint Market1501 + Occluded_Duke loaded")
            self.print_dataset_statistics(self.train, self.query, self.gallery)
            print(f"  (Market1501 IDs: 0-{self.market.num_train_pids-1}, "
                  f"Occluded_Duke IDs: {self.market.num_train_pids}-{self.num_train_pids-1})")
