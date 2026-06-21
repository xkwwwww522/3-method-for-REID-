from torch.utils.data.sampler import Sampler
from collections import defaultdict
import copy
import random
import numpy as np

class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list) #dict with list value
        #{783: [0, 5, 116, 876, 1554, 2041],...,}
        for index, (_, pid, _, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        return iter(final_idxs)

    def __len__(self):
        return self.length



"""Clothes-aware batch sampler for clothes-changing ReID.

Must be appended to sampler.py to share the namespace.

Each batch: P pids × 2 clothes × K frames = batch_size
- Same pid, diff clothes in same batch → clothes-changing positive pairs in triplet loss
- Diff pid, same clothes → hard negative pairs in triplet loss

Fallback: if a pid has only 1 clothes label, use it twice (no clothes-changing pairs for that pid).
"""
from torch.utils.data.sampler import Sampler
from collections import defaultdict
import copy
import random
import numpy as np


class ClothesAwareSampler(Sampler):
    """
    Randomly sample P identities, then for each identity,
    randomly sample 2 clothes labels, then for each (identity, clothes),
    randomly sample K instances → batch = P × 2 × K.

    Args:
        data_source (list): list of (tracklet_key, pid, camid, clothes_int).
            clothes_int is at index 3 (the 4th element).
        batch_size (int): IMS_PER_BATCH. Must be divisible by (2 * num_instances).
        num_instances (int): K — frames per (pid, clothes) group.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances  # K: frames per (pid, clothes)
        self.num_clothes = 2                # 2 clothes per pid per batch
        if batch_size % (self.num_clothes * num_instances) != 0:
            raise ValueError(
                'batch_size={} must be divisible by 2*num_instances={}'.format(
                    batch_size, 2 * num_instances))
        self.num_pids_per_batch = batch_size // (self.num_clothes * num_instances)

        # Index: pid -> clothes_int -> [indices]
        self.pid_clothes = defaultdict(lambda: defaultdict(list))
        for idx, item in enumerate(data_source):
            _, pid, _, clothes_int = item
            self.pid_clothes[pid][clothes_int].append(idx)

        self.pids = list(self.pid_clothes.keys())

        # Count pids with >= 2 clothes labels
        self.multi_clothes_pids = [
            pid for pid, cd in self.pid_clothes.items() if len(cd) >= 2]

        # Estimate epoch length: total indices / batch_size rounded up
        total_usable = 0
        for pid in self.pids:
            for cid in self.pid_clothes[pid]:
                n = len(self.pid_clothes[pid][cid])
                if n < self.num_instances:
                    n = self.num_instances  # pad with replacement
                total_usable += n - (n % self.num_instances)
        self._est_len = total_usable  # approximate num indices per epoch

        if not self.multi_clothes_pids:
            print('WARNING: No pids with >=2 clothes! Clothes-aware sampling degraded to standard.')

    def _select_group(self, indices, count):
        """Select exactly `count` indices from `indices` list."""
        if len(indices) >= count:
            return np.random.choice(indices, count, replace=False).tolist()
        else:
            return np.random.choice(indices, count, replace=True).tolist()

    def __iter__(self):
        # Pre-build per-(pid, clothes) batches of size num_instances
        # pid_cid_batches[pid][clothes_int] = list of [batch_of_K_indices, ...]
        pid_cid_batches = defaultdict(lambda: defaultdict(list))

        for pid in self.pids:
            for cid, idxs in self.pid_clothes[pid].items():
                idxs_copy = list(idxs)  # shallow copy
                random.shuffle(idxs_copy)

                # Pad to multiple of num_instances
                while len(idxs_copy) < self.num_instances:
                    idxs_copy.append(random.choice(idxs))

                # Chunk into batches of num_instances
                batch = []
                for idx in idxs_copy:
                    batch.append(idx)
                    if len(batch) == self.num_instances:
                        pid_cid_batches[pid][cid].append(batch)
                        batch = []
                if batch:
                    # Partial batch left — pad with random choices
                    while len(batch) < self.num_instances:
                        batch.append(random.choice(idxs))
                    pid_cid_batches[pid][cid].append(batch)

        # Build flat index list by generating batch after batch
        final_idxs = []

        # Track available pids — each pid becomes "exhausted" when any of its
        # clothes groups runs out of pre-built batches.
        avai_pids = list(self.pids)

        # Prefer multi-clothes pids when available
        avai_multi = [p for p in avai_pids if p in set(self.multi_clothes_pids)]

        while len(avai_pids) >= self.num_pids_per_batch:
            # Decide source: use multi-clothes pids preferentially
            if len(avai_multi) >= self.num_pids_per_batch:
                selected = random.sample(avai_multi, self.num_pids_per_batch)
            else:
                # Mix: all multi-clothes + fill with any pids
                need = self.num_pids_per_batch - len(avai_multi)
                other = [p for p in avai_pids if p not in set(avai_multi)]
                if len(other) >= need:
                    selected = list(avai_multi) + random.sample(other, need)
                elif len(avai_pids) >= self.num_pids_per_batch:
                    selected = random.sample(avai_pids, self.num_pids_per_batch)
                else:
                    # Not enough pids — pad with random
                    selected = random.choices(self.pids, k=self.num_pids_per_batch)

            # For each selected pid, pick batch groups
            batch_indices = []
            exhausted_pids = set()

            for pid in selected:
                cid_list = list(pid_cid_batches[pid].keys())

                # Pick 2 clothes — must be different if possible
                if len(cid_list) >= 2:
                    c1, c2 = random.sample(cid_list, 2)
                else:
                    c1 = c2 = cid_list[0]  # same clothes = no clothes-changing

                # Pop one batch from each clothes group
                for cid in [c1, c2]:
                    batches = pid_cid_batches[pid][cid]
                    if batches:
                        batch_indices.extend(batches.pop(0))
                    else:
                        # Refill: sample randomly from original indices
                        batch_indices.extend(
                            self._select_group(self.pid_clothes[pid][cid],
                                               self.num_instances))

                    # Mark pid as exhausted if any clothes group is empty
                    if not pid_cid_batches[pid][cid]:
                        exhausted_pids.add(pid)

            final_idxs.extend(batch_indices)

            # Remove exhausted pids
            for pid in exhausted_pids:
                if pid in avai_pids and \
                   all(len(pid_cid_batches[pid][c]) == 0 for c in pid_cid_batches[pid]):
                    avai_pids.remove(pid)
                    if pid in avai_multi:
                        avai_multi.remove(pid)

        # Trim to valid batch_size multiples
        valid_len = (len(final_idxs) // self.batch_size) * self.batch_size
        final_idxs = final_idxs[:valid_len]

        return iter(final_idxs)

    def __len__(self):
        # Return a conservative estimate — DataLoader uses this for progress bar
        return max(1, self._est_len // self.batch_size)
