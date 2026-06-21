import glob, re, os.path as osp
from .bases import BaseImageDataset

class CCVID(BaseImageDataset):
    dataset_dir = "CCVID_cope"

    def __init__(self, root="", verbose=True, **kwargs):
        super(CCVID, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, "train")
        self.query_dir = osp.join(self.dataset_dir, "query")
        self.gallery_dir = osp.join(self.dataset_dir, "gallery")
        self.mask_dir = osp.join(self.dataset_dir, "masks/dummy")

        query = self._process_annotation("query.txt", self.query_dir, "query")
        gallery = self._process_annotation("gallery.txt", self.gallery_dir, "gallery")
        train = self._process_dir(self.train_dir, "train")

        if verbose:
            print("=> CCVID loaded (with dummy masks)")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery
        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

    def _process_annotation(self, anno_file, img_dir, subset_name):
        """Parse tracklet-level annotation (query.txt/gallery.txt) and expand to images.
        
        Annotation format: tracklet_path<TAB>pid<TAB>camera_info
        e.g. "session1/001_01    001    u0_l0_s0_c0_a0"
        """
        dataset = []
        anno_path = osp.join(self.dataset_dir, anno_file)
        
        with open(anno_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('	')
                if len(parts) < 2:
                    continue
                    
                tracklet_path = parts[0]   # e.g. "session1/001_01"
                pid = int(parts[1])        # person ID
                
                # Parse camera from cam_info (3rd column)
                camid = 0
                if len(parts) >= 3:
                    cam_match = re.search(r'c(\d+)', parts[2])
                    camid = int(cam_match.group(1)) if cam_match else 0
                
                # Convert tracklet path to filename prefix
                # "session1/001_01" -> "session1_001_01"
                prefix = tracklet_path.replace('/', '_')
                
                # Find all images for this tracklet
                pattern = osp.join(img_dir, f"{prefix}_*.jpg")
                matches = sorted(glob.glob(pattern))
                
                for img_path in matches:
                    mask_path = osp.join(self.mask_dir, subset_name, 
                                        osp.basename(img_path).replace('.jpg', '.npy'))
                    dataset.append((img_path, pid, camid, 1, mask_path))
        
        return dataset

    def _process_dir(self, dir_path, subset_name):
        """Fallback: scan directory for images (used for train)."""
        dataset = []
        img_paths = sorted(glob.glob(osp.join(dir_path, "*.jpg")))
        for img_path in img_paths:
            fname = osp.basename(img_path)
            parts = fname.replace(".jpg", "").split("_")
            pid = int(parts[1]) if len(parts) >= 2 else 0
            camid = int(parts[2]) if len(parts) >= 3 else 0
            mask_path = osp.join(self.mask_dir, subset_name, 
                                fname.replace('.jpg', '.npy'))
            dataset.append((img_path, pid, camid - 1, 1, mask_path))
        return dataset
