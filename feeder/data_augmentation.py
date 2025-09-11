import random
import torch
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import torchvision.transforms.functional as F
from torchvision import transforms


class GeometricAugmentation:
    """Geometric transformations for data augmentation"""

    def __init__(self, shift_ratio=0.1, scale_range=(0.9, 1.1)):
        """
        Args:
            shift_ratio: Maximum shift ratio (10% of image width/height)
            scale_range: Scaling range (0.9 to 1.1 means 90% to 110%)
        """
        self.shift_ratio = shift_ratio
        self.scale_range = scale_range

    def random_shift(self, img):
        """Shifting of the ROI within 10% of the image width"""
        width, height = img.size
        max_shift_x = int(width * self.shift_ratio)
        max_shift_y = int(height * self.shift_ratio)

        shift_x = random.randint(-max_shift_x, max_shift_x)
        shift_y = random.randint(-max_shift_y, max_shift_y)

        # Apply translation
        return F.affine(img, angle=0, translate=[shift_x, shift_y],
                        scale=1.0, shear=0, fill=0)

    def random_scale(self, img):
        """Image scaling between 0.9 to 1.1"""
        scale_factor = random.uniform(self.scale_range[0], self.scale_range[1])
        width, height = img.size
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)

        # Resize and then center crop or pad to original size
        img_scaled = F.resize(img, (new_height, new_width))

        # Center crop or pad to maintain original size
        if scale_factor > 1.0:
            # Crop to original size
            left = (new_width - width) // 2
            top = (new_height - height) // 2
            img_scaled = F.crop(img_scaled, top, left, height, width)
        else:
            # Pad to original size
            pad_width = (width - new_width) // 2
            pad_height = (height - new_height) // 2
            img_scaled = F.pad(img_scaled, [pad_width, pad_height,
                                            width - new_width - pad_width,
                                            height - new_height - pad_height],
                               fill=0)

        return img_scaled

    def __call__(self, img):
        # Apply random shift
        if random.random() < 0.5:
            img = self.random_shift(img)

        # Apply random scale
        if random.random() < 0.5:
            img = self.random_scale(img)

        return img


class PixelLevelAugmentation:
    """Pixel-level transformations for data augmentation"""

    def __init__(self, noise_std=0.1, brightness_range=(0.8, 1.2),
                 contrast_range=(0.8, 1.2), blur_radius_range=(0.5, 2.0)):
        """
        Args:
            noise_std: Standard deviation for Gaussian noise
            brightness_range: Range for brightness adjustment
            contrast_range: Range for contrast adjustment  
            blur_radius_range: Range for Gaussian blur radius
        """
        self.noise_std = noise_std
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.blur_radius_range = blur_radius_range

    def add_gaussian_noise(self, img_tensor):
        """Add Gaussian noise to image tensor"""
        noise = torch.randn_like(img_tensor) * self.noise_std
        noisy_img = img_tensor + noise
        return torch.clamp(noisy_img, 0.0, 1.0)

    def adjust_brightness_contrast(self, img):
        """Brightness and contrast alteration"""
        # Brightness adjustment
        if random.random() < 0.5:
            brightness_factor = random.uniform(
                self.brightness_range[0], self.brightness_range[1])
            img = ImageEnhance.Brightness(img).enhance(brightness_factor)

        # Contrast adjustment
        if random.random() < 0.5:
            contrast_factor = random.uniform(
                self.contrast_range[0], self.contrast_range[1])
            img = ImageEnhance.Contrast(img).enhance(contrast_factor)

        return img

    def gaussian_blur(self, img):
        """Apply Gaussian blurring"""
        blur_radius = random.uniform(
            self.blur_radius_range[0], self.blur_radius_range[1])
        return img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    def __call__(self, img, img_tensor=None):
        # Apply brightness and contrast (on PIL image)
        if random.random() < 0.5:
            img = self.adjust_brightness_contrast(img)

        # Apply Gaussian blur (on PIL image)
        if random.random() < 0.3:  # Lower probability for blur
            img = self.gaussian_blur(img)

        return img

    def apply_noise_to_tensor(self, img_tensor):
        """Apply Gaussian noise to tensor (call this after ToTensor)"""
        if random.random() < 0.5:
            img_tensor = self.add_gaussian_noise(img_tensor)
        return img_tensor


class CombinedAugmentation:
    """Combined geometric and pixel-level augmentations"""

    def __init__(self,
                 shift_ratio=0.1,
                 scale_range=(0.9, 1.1),
                 noise_std=0.1,
                 brightness_range=(0.8, 1.2),
                 contrast_range=(0.8, 1.2),
                 blur_radius_range=(0.5, 2.0)):

        self.geometric_aug = GeometricAugmentation(shift_ratio, scale_range)
        self.pixel_aug = PixelLevelAugmentation(noise_std, brightness_range,
                                                contrast_range, blur_radius_range)

    def __call__(self, img):
        """Apply augmentations to PIL image"""
        # Apply geometric transformations first
        img = self.geometric_aug(img)

        # Apply pixel-level transformations
        img = self.pixel_aug(img)

        return img

    def apply_noise_to_tensor(self, img_tensor):
        """Apply noise to tensor after ToTensor transform"""
        return self.pixel_aug.apply_noise_to_tensor(img_tensor)


class Cutout(object):
    """Randomly mask out one or more patches from an image."""
    def __init__(self, n_holes=3, length=16):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)
        self.n_holes = np.random.choice([0, 1], p=(0.6, 0.4))
        self.length = random.randint(16, 24)
        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            mask[y1:y2, x1:x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask
        return img