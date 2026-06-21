import glob, os.path as osp, re as re_mod
from .bases import BaseImageDataset

class MOVE_NEW(BaseImageDataset):
    """MOVE_NEW: flat files, C1=query, C2=gallery. Naming: XXXXC1T0001FYYYY.jpg"""
    dataset_dir = 'move_new/data/move_eval_cam'

    def __init__(self, root='', verbose=True, **kwargs):
        super().__init__()
        base = osp.join(root, self.dataset_dir)
        self.query_dir = osp.join(base, 'query')
        self.gallery_dir = osp.join(base, 'gallery')
        for d in [self.query_dir, self.gallery_dir]:
            if not osp.exists(d):
                raise RuntimeError(d + ' not found')

        query = self._process_dir(self.query_dir)
        gallery = self._process_dir(self.gallery_dir)
        self.train = query
        self.query = query
        self.gallery = gallery
        info = self.get_imagedata_info
        for prefix, data in [('train', query), ('query', query), ('gallery', gallery)]:
            p, i, c, v = info(data)
            setattr(self, f'num_{prefix}_pids', p)
            setattr(self, f'num_{prefix}_imgs', i)
            setattr(self, f'num_{prefix}_cams', c)
            setattr(self, f'num_{prefix}_vids', v)

        if verbose:
            self.print_dataset_statistics(query, query, gallery)

    def _process_dir(self, dir_path):
        img_paths = sorted(glob.glob(osp.join(dir_path, '*.jpg')))
        ptn = re_mod.compile(r'(\d+)C(\d+)')
        ds, ids = [], set()
        for p in img_paths:
            m = ptn.search(osp.basename(p))
            if m:
                pid, cam = map(int, m.groups())
                ids.add(pid)
                ds.append((p, pid, cam))
        id_map = {pid: i for i, pid in enumerate(sorted(ids))}
        return [(p, id_map[pid], cam, 0) for p, pid, cam in ds]
