import glob
import os

import torch
from torch.utils.data import Dataset

from process_image import tensorify


class OperatorDataset(Dataset):
    def __init__(self, root, image_size=256):
        self.paths = sorted(glob.glob(os.path.join(root, "*.png")))
        self.image_size = image_size
        print(f"loading {len(self.paths)} images into memory...")
        self.images = [
            tensorify(path, resize=[image_size, image_size])[0] for path in self.paths
        ]
        self.stacked = torch.stack(self.images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]

    def random_batch(self, batch_size):
        idx = torch.randint(0, len(self.images), (batch_size,))
        return self.stacked[idx]

    def random_one(self):
        idx = torch.randint(0, len(self.images), (1,))
        return self.stacked[idx]
