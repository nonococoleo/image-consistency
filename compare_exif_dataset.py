import os
import random
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from random_crop_transformer import RandomCropTransformer


class CompareExifDataset(Dataset):
    """
    Dataset for CompareExifModel
    """

    def __init__(self, csv_file, root_dir, num_pairs, patch_size):
        self.exif_info = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transformer = RandomCropTransformer(patch_size)

        self.patches = []
        length = range(len(self.exif_info))
        for i in length:
            img_name = os.path.join(self.root_dir, self.exif_info.iloc[i, 0])
            self.patches.append(self.transformer(Image.open(img_name).convert('RGB')))
        print("finish generating patches")

        self.pairs = []
        self.labels = []
        # self.pairs = [(i, i) for i in length]
        # self.labels = [torch.ones(6, dtype=torch.float32) for _ in length]
        while len(self.pairs) < num_pairs:
            a = random.choice(length)
            b = random.choice(length)
            exif_a = self.exif_info.iloc[a, 1:]
            exif_b = self.exif_info.iloc[b, 1:]
            label = exif_a == exif_b
            label = torch.from_numpy(label.astype('float32').to_numpy())
            if torch.sum(label) >= 2:
                self.pairs.append((a, b))
                self.labels.append(label)
        print("finish generating pairs")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        a, b = self.pairs[idx]
        sample = {'image': (self.patches[a], self.patches[b]), 'label': self.labels[idx]}
        return sample
