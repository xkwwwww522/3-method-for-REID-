"""Targeted erasing strategies tailored for MOVE dataset characteristics."""
import torch
import random

class HeadErase:
    """Erase the top 1/3 of the image (simulates face/head occlusion)."""
    def __init__(self, probability=0.3):
        self.probability = probability

    def __call__(self, img_tensor):
        if random.random() > self.probability:
            return img_tensor
        _, H, W = img_tensor.shape
        head_h = max(1, H // 3)
        img_tensor[:, :head_h, :] = 0.0
        return img_tensor

    def __repr__(self):
        return 'HeadErase(p={})'.format(self.probability)


class StripeErase:
    """Erase a horizontal stripe (simulates body blocked by obstacle)."""
    def __init__(self, probability=0.3, stripe_ratio=0.5):
        self.probability = probability
        self.stripe_ratio = stripe_ratio

    def __call__(self, img_tensor):
        if random.random() > self.probability:
            return img_tensor
        _, H, W = img_tensor.shape
        stripe_h = max(1, int(H * self.stripe_ratio))
        start_y = random.randint(0, H - stripe_h)
        img_tensor[:, start_y:start_y + stripe_h, :] = 0.0
        return img_tensor

    def __repr__(self):
        return 'StripeErase(p={}, ratio={})'.format(self.probability, self.stripe_ratio)


class BrightnessPerturb:
    """Randomly darken a region of the image (simulates MOVE dark areas)."""
    def __init__(self, probability=0.3, factor=0.4):
        self.probability = probability
        self.factor = factor

    def __call__(self, img_tensor):
        if random.random() > self.probability:
            return img_tensor
        _, H, W = img_tensor.shape
        # Random region: 30-70% of image area
        area_ratio = 0.3 + random.random() * 0.4
        target_h = max(1, int(H * (area_ratio ** 0.5)))
        target_w = max(1, int(W * (area_ratio ** 0.5)))
        start_y = random.randint(0, H - target_h)
        start_x = random.randint(0, W - target_w)
        
        region = img_tensor[:, start_y:start_y + target_h, start_x:start_x + target_w]
        # Apply gamma-like darkening (not pure black)
        dark_factor = self.factor + random.random() * (1.0 - self.factor) * 0.3
        img_tensor[:, start_y:start_y + target_h, start_x:start_x + target_w] = region * dark_factor
        # Clamp to valid range
        img_tensor = torch.clamp(img_tensor, -1.0, 1.0)
        return img_tensor

    def __repr__(self):
        return 'BrightnessPerturb(p={}, factor={})'.format(self.probability, self.factor)
