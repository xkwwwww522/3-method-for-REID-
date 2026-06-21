"""Progressive Random Erasing with epoch-based scheduling."""
import torch
from timm.data.random_erasing import RandomErasing

class ProgressiveRandomErasing:
    """RandomErasing wrapper that supports epoch-dependent probability and area.

    Schedule format: [(start_epoch, probability, (min_area, max_area)), ...]
    Example: [(0, 0.3, (0.02, 0.2)), (10, 0.5, (0.02, 0.5)), (20, 0.5, (0.1, 0.6))]
    """
    def __init__(self, schedule=None, mode='pixel', max_count=1, device='cpu'):
        if schedule is None:
            schedule = [(0, 0.5, (0.02, 0.33))]
        self.schedule = sorted(schedule, key=lambda x: x[0])
        self.mode = mode
        self.max_count = max_count
        self.device = device
        self.current_prob = 0.0
        self.current_min_area = 0.02
        self.current_max_area = 0.33
        self._eraser = None
        self._build_eraser()

    def _build_eraser(self):
        if self.current_prob > 0:
            self._eraser = RandomErasing(
                probability=self.current_prob,
                min_area=self.current_min_area,
                max_area=self.current_max_area,
                mode=self.mode,
                max_count=self.max_count,
                device=self.device)
        else:
            self._eraser = None

    def set_epoch(self, epoch):
        """Update RE settings based on current epoch."""
        best_idx = 0
        for i, (start_ep, _, _) in enumerate(self.schedule):
            if epoch >= start_ep:
                best_idx = i
        _, prob, (min_a, max_a) = self.schedule[best_idx]
        self.current_prob = prob
        self.current_min_area = min_a
        self.current_max_area = max_a
        self._build_eraser()

    def __call__(self, img):
        if self._eraser is None:
            return img
        return self._eraser(img)

    def __repr__(self):
        return (f"ProgressiveRandomErasing(schedule={self.schedule}, "
                f"current_prob={self.current_prob}, area=({self.current_min_area},{self.current_max_area}))")
