import torch
from torch.utils.data import Dataset
import os
import pandas as pd
from PIL import Image



class ConditionalDiffusionDataset1D(Dataset):
    def __init__(self, Particles, Observations):
        self.Particles = Particles
        self.Observations = Observations
        self.DataSize = self.Particles.shape[0]

    def __len__(self):
        return self.DataSize

    def __getitem__(self, idx):
        return self.Particles[idx,...], self.Observations[idx,...]
    

class UnconditionalDiffusionDataset2D(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 1])
        image = Image.open(img_path,)

        if self.transform:
            image = self.transform(image)

        return image