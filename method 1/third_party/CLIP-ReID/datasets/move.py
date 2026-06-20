# encoding: utf-8
"""MOve image ReID dataset adapter.

Expected default layout:
    <root>/move_eval_cam/query/*.jpg
    <root>/move_eval_cam/gallery/*.jpg

File names are parsed as:
    3272C2T0001F0360.jpg -> pid=3272, camid=2, track=1, frame=360
"""

import glob
import os.path as osp
import re

from .bases import BaseImageDataset


class MoveEvalCam(BaseImageDataset):
    dataset_dir = "move_eval_cam"
    min_train_pids = 751
    min_train_cams = 6
    _pattern = re.compile(r"^([-\d]+)C(\d+)T(\d+)F(\d+)\.(?:jpg|jpeg|png)$", re.IGNORECASE)

    def __init__(self, root="", verbose=True, pid_begin=0, **kwargs):
        super(MoveEvalCam, self).__init__()
        root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = self._resolve_dataset_dir(root)
        self.train_dir = self._first_existing_dir("bounding_box_train", "train")
        self.query_dir = self._first_existing_dir("query")
        self.gallery_dir = self._first_existing_dir("gallery", "bounding_box_test", "test")

        self._check_before_run()
        self.pid_begin = pid_begin
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)
        if self.train_dir is not None:
            train = self._process_dir(self.train_dir, relabel=True)
        else:
            train = self._relabel_train(query + gallery)

        if verbose:
            print("=> MOve eval camera dataset loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery
        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        # Market-pretrained checkpoints have classifier/camera parameters sized
        # for 751 train IDs and 6 cameras. Keep those model dimensions while
        # evaluating MOve without retraining; query/gallery labels stay unchanged.
        self.num_train_pids = max(self.num_train_pids, self.min_train_pids)
        self.num_train_cams = max(self.num_train_cams, self.min_train_cams)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

    def _resolve_dataset_dir(self, root):
        for name in ("move_eval_cam", "MOVE", "move"):
            path = osp.join(root, name)
            if osp.isdir(path):
                return path
        return root

    def _first_existing_dir(self, *names):
        for name in names:
            path = osp.join(self.dataset_dir, name)
            if osp.isdir(path):
                return path
        return None

    def _check_before_run(self):
        if not osp.isdir(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if self.query_dir is None:
            raise RuntimeError("query dir is not available under '{}'".format(self.dataset_dir))
        if self.gallery_dir is None:
            raise RuntimeError("gallery dir is not available under '{}'".format(self.dataset_dir))

    def _parse_name(self, img_path):
        match = self._pattern.match(osp.basename(img_path))
        if match is None:
            raise RuntimeError("Unexpected MOve image name: {}".format(img_path))
        pid, camid, trackid, _ = map(int, match.groups())
        return pid, camid - 1, trackid

    def _process_dir(self, dir_path, relabel=False):
        img_paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
            img_paths.extend(glob.glob(osp.join(dir_path, ext)))
            img_paths.extend(glob.glob(osp.join(dir_path, "**", ext), recursive=True))
        img_paths = sorted(set(img_paths))

        pid_container = set()
        for img_path in img_paths:
            pid, _, _ = self._parse_name(img_path)
            if pid == -1:
                continue
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(sorted(pid_container))}

        dataset = []
        for img_path in img_paths:
            pid, camid, trackid = self._parse_name(img_path)
            if pid == -1:
                continue
            if relabel:
                pid = pid2label[pid]
            dataset.append((img_path, self.pid_begin + pid, camid, trackid))
        return dataset

    def _relabel_train(self, dataset):
        pid_container = sorted({pid for _, pid, _, _ in dataset})
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        return [(img_path, pid2label[pid], camid, trackid) for img_path, pid, camid, trackid in dataset]
