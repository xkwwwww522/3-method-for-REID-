import glob
import re
import os.path as osp
from .bases import BaseImageDataset

class MOVE(BaseImageDataset):
    """MOVE dataset - test only, no training split.
    After reorganization:
    - query: 100 IDs x 2 images = 200 images
    - gallery (test/): 100 IDs x varying images = 340 images
    """
    dataset_dir = 'MOVE'

    def __init__(self, root='', verbose=True, **kwargs):
        super(MOVE, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'test')

        self._check_before_run()

        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)
        # Use query as placeholder train for dataloader compatibility
        # (MOVE is test-only; these are NOT used for training)
        train = query

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids =             self.get_imagedata_info(self.query)  # use query for num_classes
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids =             self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids =             self.get_imagedata_info(self.gallery)

        if verbose:
            print("=> MOVE dataset loaded (test-only, no train split)")
            self.print_dataset_statistics(train, query, gallery)

    def _check_before_run(self):
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '**', '*.jpg'), recursive=True)
        pattern = re.compile(r'(\d+)C(\d+)')

        dataset = []
        pid_container = set()

        for img_path in img_paths:
            filename = osp.basename(img_path)
            res = pattern.search(filename)
            if res:
                pid, camid = map(int, res.groups())
                if pid == -1:
                    continue
                pid_container.add(pid)
                dataset.append((img_path, pid, camid))
            else:
                print("Warning: file {} naming mismatch, skipped".format(filename))

        pid2label = {pid: label for label, pid in enumerate(sorted(pid_container))}

        final_dataset = []
        for img_path, pid, camid in dataset:
            if relabel:
                pid = pid2label[pid]
            final_dataset.append((img_path, pid, camid, 0))

        return final_dataset
