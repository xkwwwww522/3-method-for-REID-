import glob, re, os.path as osp
from .bases import BaseImageDataset

class MOVE(BaseImageDataset):
    dataset_dir = "MOVE"

    def __init__(self, root="", verbose=True, **kwargs):
        super(MOVE, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.query_dir = osp.join(self.dataset_dir, "query")
        self.gallery_dir = osp.join(self.dataset_dir, "test")
        self.train_dir = osp.join(self.dataset_dir, "test")  # placeholder

        query = self._process_dir(self.query_dir)
        gallery = self._process_dir(self.gallery_dir)
        train = [('', 0, 0, 0, None)]  # placeholder

        if verbose:
            print("=> MOVE loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery
        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = 0, 0, 0, 0
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

    def _process_dir(self, dir_path):
        # Each subdir is a person ID, images named {pid}C{cam}T0001F{frame}.jpg
        dataset = []
        pid_dirs = glob.glob(osp.join(dir_path, "*"))
        for pid_dir in sorted(pid_dirs):
            pid_str = osp.basename(pid_dir)
            try:
                pid = int(pid_str)
            except ValueError:
                continue
            img_paths = glob.glob(osp.join(pid_dir, "*.jpg"))
            for img_path in sorted(img_paths):
                # Extract camera from filename: {pid}C{cam}T...
                fname = osp.basename(img_path)
                match = re.search(r"C(\d+)", fname)
                camid = int(match.group(1)) - 1 if match else 0
                dataset.append((img_path, pid, camid, 1, None))
        return dataset
