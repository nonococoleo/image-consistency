import os
import random
import numpy as np
from PIL import Image
import torch
import torchvision
from torch.utils.data import Dataset


def get_consistency_score(p1, p2, size):
    """
    use number of different bit in mask to represent score
    :param p1: patch 1
    :param p2: patch 2
    :param size: patch size
    :return: Consistency Score
    """

    return torch.sum(abs(p1 - p2)) / size


class FeatureDataset(Dataset):
    """
    dataset for feature model
    """

    def __init__(self, root_dir, num_pairs, patch_size):
        self.root_dir = root_dir
        self.size = patch_size
        self.pairs = []

        for i in os.listdir(self.root_dir + '/images')[:2]:
            image = Image.open(os.path.join(self.root_dir + '/images/' + i))
            height, width = np.array(image).shape[:2]
            for j in range(num_pairs):
                # random pick two left top corner
                x1 = random.randint(0, height - patch_size)
                y1 = random.randint(0, width - patch_size)
                x2 = random.randint(0, height - patch_size)
                y2 = random.randint(0, width - patch_size)
                self.pairs.append((i.split('.')[0], (x1, y1), (x2, y2)))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        file_name, point1, point2 = self.pairs[idx]

        image = Image.open(os.path.join(self.root_dir + '/images/' + file_name + ".jpg")).convert('RGB')
        mask = Image.open(os.path.join(self.root_dir + '/masks/' + file_name + ".png"))

        image = torchvision.transforms.ToTensor()(image)
        mask = torchvision.transforms.ToTensor()(mask)

        x1, y1 = point1
        p1 = image[:3, x1:x1 + self.size, y1:y1 + self.size]
        m1 = mask[:, x1:x1 + self.size, y1:y1 + self.size]

        x2, y2 = point2
        p2 = image[:3, x2:x2 + self.size, y2:y2 + self.size]
        m2 = mask[:, x2:x2 + self.size, y2:y2 + self.size]

        sample = {'p1': p1, 'p2': p2, 'score': get_consistency_score(m1, m2, self.size ** 2)}
        return sample
