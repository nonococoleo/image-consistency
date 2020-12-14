import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from random_crop_transformer import RandomCropTransformer


class PredictExifDataset(Dataset):
    """
    Dataset for PredictExifModel
    """

    def __init__(self, csv_file, root_dir, replica, patch_size):
        self.exif_info = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = RandomCropTransformer(patch_size)

    def __len__(self):
        return len(self.exif_info)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.exif_info.iloc[idx, 0])
        patch = self.transform(Image.open(img_name).convert('RGB'))
        exif = torch.tensor(self.exif_info.iloc[idx, 1:])

        return patch, exif
