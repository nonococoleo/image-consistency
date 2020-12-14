import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from RandomCropTransformer import RandomCropTransformer


class PredictExifDataset(Dataset):
    """
    Dataset for PredictExifModel
    """

    def __init__(self, csv_file, root_dir, replica, patch_size):
        self.exif_info = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = RandomCropTransformer(patch_size)
        self.replica = replica

        self.patches = []
        self.exifs = []
        length = range(len(self.exif_info))
        for r in range(0, self.replica):
            for i in length:
                img_name = os.path.join(self.root_dir, self.exif_info.iloc[i, 0])
                self.patches.append(self.transform(Image.open(img_name).convert('RGB')))
                self.exifs.append(torch.tensor(self.exif_info.iloc[i, 1:]))

    def __len__(self):
        return len(self.exif_info) * self.replica

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        patch = self.patches[idx]
        exif = self.exifs[idx]

        return patch, exif
