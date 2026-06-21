import glob, re, os.path as osp
from .bases import BaseImageDataset

class MOVE_ENHANCED(BaseImageDataset):
    """MOVE dataset with CLAHE + sharpen + bicubic enhancement applied."""
    dataset_dir = 'MOVE_ENHANCED'

    def __init__(self, root='', verbose=True, **kwargs):
        super().__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'test')
        self._check_before_run()
        query = self._process_dir(self.query_dir)
        gallery = self._process_dir(self.gallery_dir)
        self.train = query
        self.query = query
        self.gallery = gallery
        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.query)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)
        if verbose:
            print("=> MOVE_ENHANCED loaded (CLAHE + sharpen + bicubic)")
            self.print_dataset_statistics(self.query, self.query, self.gallery)

    def _check_before_run(self):
        for d in [self.dataset_dir, self.query_dir, self.gallery_dir]:
            if not osp.exists(d):
                raise RuntimeError("{} not found".format(d))

    def _process_dir(self, dir_path):
        img_paths = glob.glob(osp.join(dir_path, '**', '*.jpg'), recursive=True)
        pattern = re.compile(r'(\d+)C(\d+)')
        dataset = []
        pid_set = set()
        for p in img_paths:
            m = pattern.search(osp.basename(p))
            if m:
                pid, cam = map(int, m.groups())
                pid_set.add(pid)
                dataset.append((p, pid, cam))
        pid_map = {pid: i for i, pid in enumerate(sorted(pid_set))}
        return [(p, pid_map[pid], cam, 0) for p, pid, cam in dataset]
