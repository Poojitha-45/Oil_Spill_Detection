# dataset.py

import cv2
import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np

class OilSpillDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_paths = sorted(list(Path(image_dir).glob("*.png")))
        self.mask_paths  = sorted(list(Path(mask_dir).glob("*.png")))
        self.transform = transform

        assert len(self.image_paths) == len(self.mask_paths), \
            "Images and masks count do not match"

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image (grayscale)
        image = cv2.imread(str(self.image_paths[idx]), cv2.IMREAD_GRAYSCALE)
        mask  = cv2.imread(str(self.mask_paths[idx]), cv2.IMREAD_GRAYSCALE)

        image = image.astype(np.float32) / 255.0
        mask  = (mask > 127).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask  = augmented["mask"]
        else:
            image = torch.from_numpy(image).unsqueeze(0)
            mask  = torch.from_numpy(mask).unsqueeze(0)

        return image, mask
