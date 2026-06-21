import glob, re, os.path as osp
from .bases import BaseImageDataset

class CCVID_sample(BaseImageDataset):
    dataset_dir = "CCVID_sample"

    def __init__(self, root="", verbose=True, **kwargs):
        super(CCVID_sample, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, "train")
        self.query_dir = osp.join(self.dataset_dir, "query")
        self.gallery_dir = osp.join(self.dataset_dir, "gallery")
        query = self._process_dir(self.query_dir)
        gallery = self._process_dir(self.gallery_dir)
        train = self._process_dir(self.train_dir)
        if verbose:
            print("=> CCVID_sample loaded")
            self.print_dataset_statistics(train, query, gallery)
        self.train = train
        self.query = query
        self.gallery = gallery
        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

    def _process_dir(self, dir_path):
        dataset = []
        img_paths = sorted(glob.glob(osp.join(dir_path, "*.jpg")))
        for img_path in img_paths:
            fname = osp.basename(img_path)
            parts = fname.replace(".jpg", "").split("_")
            pid = int(parts[1]) if len(parts) >= 2 else 0
            camid = int(parts[2]) if len(parts) >= 3 else 0
            dataset.append((img_path, pid, camid, 1, None))
        return dataset
