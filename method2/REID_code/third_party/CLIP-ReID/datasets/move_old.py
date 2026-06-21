import glob, os.path as osp, re
from .bases import BaseImageDataset

class MOVE_OLD(BaseImageDataset):
    dataset_dir = 'MOVE_OLD'
    def __init__(self, root='', verbose=True, **kwargs):
        super().__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'test')
        for d in [self.dataset_dir, self.query_dir, self.gallery_dir]:
            if not osp.exists(d):
                raise RuntimeError(d + ' not found')
        query = self._process_dir(self.query_dir)
        gallery = self._process_dir(self.gallery_dir)
        self.train = query
        self.query = query
        self.gallery = gallery
        info = self.get_imagedata_info
        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = info(self.query)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = info(self.gallery)
        if verbose:
            self.print_dataset_statistics(self.query, self.query, self.gallery)

    def _process_dir(self, dir_path):
        img_paths = glob.glob(osp.join(dir_path, '**', '*.jpg'), recursive=True)
        ptn = re.compile(r'(\d+)C(\d+)')
        ds, ps2 = [], set()
        for p in img_paths:
            m = ptn.search(osp.basename(p))
            if m:
                pid, cam = map(int, m.groups())
                ps2.add(pid)
                ds.append((p, pid, cam))
        pm = {pid: i for i, pid in enumerate(sorted(ps2))}
        return [(p, pm[pid], cam, 0) for p, pid, cam in ds]
