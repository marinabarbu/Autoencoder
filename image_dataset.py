""""https://www.cs.toronto.edu/~lczhang/360/lec/w05/autoencoder.html"""

import os
import numpy as np
from torch.utils.data import Dataset



class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        assert os.path.exists(self.root_dir), "Path to videos cannot be found"
        self.images = sorted([os.path.join(self.root_dir, file) for file in os.listdir(self.root_dir) if file.endswith(".npy")])
        #self.images = sorted([os.path.join(self.root_dir, file) for file in os.listdir(self.root_dir) if file.endswith(".npy")])[:200]

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        img = np.load(self.images[item])
        #print(img.min(), img.max())
        img = self.transform(img*255).unsqueeze(0)
        return img

