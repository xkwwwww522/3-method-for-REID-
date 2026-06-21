"""CCVID v3: Clothes labels stored. + ClothesAwareSampler for triplet training.

Key change: training entries now store (tracklet_key, pid, camid, clothes_int).
clothes_int = mapping from 'u0_l0_s0_c0_a0' string to 0, 1, 2, ...
The ClothesAwareSampler groups by (pid, clothes_int) to ensure each batch contains:
- Per pid: 2 DIFFERENT clothes (→ clothes-changing positives in triplet loss)
- Across pids: if same clothes label → hard negatives in triplet loss
"""
import glob, os, re, random
import os.path as osp
from collections import defaultdict
from .bases import BaseImageDataset, read_image


class CCVID(BaseImageDataset):
    dataset_dir = 'CCVID_cope'

    def __init__(self, root='', verbose=True, num_train_frames=20, num_test_frames=4, **kwargs):
        super().__init__()
        self.root_dir = osp.join(root, self.dataset_dir)
        self.train_path = osp.join(self.root_dir, 'train.txt')
        self.query_path = osp.join(self.root_dir, 'query.txt')
        self.gallery_path = osp.join(self.root_dir, 'gallery.txt')
        self.num_train_frames = num_train_frames
        self.num_test_frames = num_test_frames
        self._check_before_run()

        # Pre-build file index
        print('=> Indexing CCVID images (one-pass glob)...')
        all_files = glob.glob(osp.join(self.root_dir, '**', '*.jpg'))
        self._tracklet_files = defaultdict(list)
        for fp in all_files:
            basename = osp.basename(fp)
            match = re.match(r'(.+)_\d+\.jpg$', basename)
            if match:
                self._tracklet_files[match.group(1)].append(fp)
        for k in self._tracklet_files:
            self._tracklet_files[k].sort()
        print('=> Indexed %d images into %d tracklets' % (len(all_files), len(self._tracklet_files)))

        # Build clothes string -> int mapping
        self._clothes_map = {}
        self._next_cid = 0
        with open(self.train_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    cs = parts[2]
                    if cs not in self._clothes_map:
                        self._clothes_map[cs] = self._next_cid
                        self._next_cid += 1
        print('=> Clothes types: %d' % len(self._clothes_map))

        # Parse splits
        train = self._process_split_train(self.train_path, relabel=True)
        query = self._process_split_test(self.query_path, relabel=False)
        gallery = self._process_split_test(self.gallery_path, relabel=False)

        self.train = train
        self.query = query
        self.gallery = gallery
        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

        if verbose:
            print('=> CCVID loaded')
            self.print_dataset_statistics(train, query, gallery)

    def _check_before_run(self):
        for p in [self.root_dir, self.train_path, self.query_path, self.gallery_path]:
            if not osp.exists(p):
                raise RuntimeError('{} not found'.format(p))

    def _parse_line(self, line):
        parts = line.strip().split()
        if len(parts) < 2:
            return None
        prefix = parts[0]
        pid = int(parts[1])
        clothes_str = parts[2] if len(parts) >= 3 else ''
        return prefix, pid, clothes_str

    def _get_key(self, prefix):
        return prefix.replace('/', '_')

    def _process_split_train(self, path, relabel=False):
        """Training entries: (tracklet_key, pid, camid, clothes_int).
        The clothes_int is the 4th element (was 0/vid previously).
        ClothesAwareSampler groups by (pid, clothes_int)."""
        dataset = []
        pid_set = set()
        cam_counter = {}

        with open(path) as f:
            for line in f:
                result = self._parse_line(line)
                if result is None:
                    continue
                prefix, pid, clothes_str = result
                key = self._get_key(prefix)
                files = self._tracklet_files.get(key, [])
                if not files:
                    continue

                if pid not in cam_counter:
                    cam_counter[pid] = 0
                camid = cam_counter[pid] % 6
                cam_counter[pid] += 1

                # Encode clothes label as int
                clothes_int = self._clothes_map.get(clothes_str, -1)
                if clothes_int < 0:
                    clothes_int = self._next_cid
                    self._clothes_map[clothes_str] = self._next_cid
                    self._next_cid += 1

                pid_set.add(pid)
                for _ in range(self.num_train_frames):
                    dataset.append((key, pid, camid, clothes_int))

        if relabel:
            pid2label = {pid: i for i, pid in enumerate(sorted(pid_set))}
            dataset = [(k, pid2label[pid], cam, cid) for k, pid, cam, cid in dataset]
        else:
            pass  # already correct format

        return dataset

    def _process_split_test(self, path, relabel=False):
        """Test: fixed evenly-spaced frames. Uses old format (path, pid, cam, 0)."""
        dataset = []
        pid_set = set()
        cam_counter = {}

        with open(path) as f:
            for line in f:
                result = self._parse_line(line)
                if result is None:
                    continue
                prefix, pid, _ = result
                key = self._get_key(prefix)
                files = self._tracklet_files.get(key, [])
                if not files:
                    continue

                if pid not in cam_counter:
                    cam_counter[pid] = 0
                camid = cam_counter[pid] % 6
                cam_counter[pid] += 1

                n_files = len(files)
                n_take = min(self.num_test_frames, n_files)
                step = max(1, n_files // n_take)
                selected = [files[i] for i in range(0, n_files, step)][:n_take]

                for img_path in selected:
                    pid_set.add(pid)
                    dataset.append((img_path, pid, camid))

        if relabel:
            pid2label = {pid: i for i, pid in enumerate(sorted(pid_set))}
            dataset = [(path, pid2label[pid], cam, 0) for path, pid, cam in dataset]
        else:
            dataset = [(path, pid, cam, 0) for path, pid, cam in dataset]

        return dataset


class RandomFrameDataset:
    """Training dataset wrapper: (tracklet_key, pid, camid, clothes_int) -> random frame.

    __getitem__ returns (img, pid, camid, clothes_int, fpath).
    train_collate_fn treats clothes_int as 'viewids' — passed through to model.
    Model ignores view_label (SIE_VIEW is off) — it's harmless passthrough.

    Test items: (path, pid, camid, 0) — handled for backward compatibility.
    """

    def __init__(self, dataset, tracklet_files, transform=None):
        self.dataset = dataset          # list of (key/path, pid, camid, clothes_or_vid)
        self.tracklet_files = tracklet_files  # dict: key -> [file_paths]
        self.transform = transform

    def __getitem__(self, index):
        item = self.dataset[index]
        first, pid, camid, aux = item  # aux = clothes_int (train) or 0 (test)

        if isinstance(first, str) and first in self.tracklet_files:
            fpath = random.choice(self.tracklet_files[first])
        elif isinstance(first, str) and not first.startswith('/'):
            fpath = first
        else:
            fpath = first

        img = read_image(fpath)
        if self.transform is not None:
            img = self.transform(img)

        return img, pid, camid, aux, fpath

    def __len__(self):
        return len(self.dataset)
