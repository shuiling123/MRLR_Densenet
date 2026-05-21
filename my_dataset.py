from PIL import Image
import torch
import numpy as np
from torch.utils.data import Dataset
import random


class MyDataSet(Dataset):
    """Custom dataset - with special handling for FER2013"""

    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])

        # Special handling: histogram equalization for grayscale images from FER2013
        if img.mode != 'RGB':
            # Convert to RGB
            img = img.convert('RGB')

        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def histogram_equalization(self, img):
        """Histogram equalization - enhance contrast of grayscale images"""
        img_array = np.array(img)

        # Compute histogram
        hist, bins = np.histogram(img_array.flatten(), 256, [0, 256])

        # Compute cumulative distribution function
        cdf = hist.cumsum()
        cdf_normalized = cdf * 255 / cdf[-1]

        # Apply histogram equalization using the CDF
        img_equalized = np.interp(img_array.flatten(), bins[:-1], cdf_normalized)
        img_equalized = img_equalized.reshape(img_array.shape).astype(np.uint8)

        return Image.fromarray(img_equalized)

    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels