import os
import random
import numpy as np
from PIL import Image
import torchvision
from torch.utils.data import Dataset


class DirectDataset(Dataset):
    def __init__(self, root_dir, size=400):
        self.root_dir = root_dir
        self.size = size
        self.files = os.listdir(self.root_dir + '/images')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx].split(".")[0]

        image = np.array(Image.open(os.path.join(self.root_dir + '/images/' + file_name + ".jpg")).convert('RGB'))
        mask = np.array(Image.open(os.path.join(self.root_dir + '/masks/' + file_name + ".png")))

        height, width = image.shape[0:2]
        x = random.randint(0, height - self.size)
        y = random.randint(0, width - self.size)
        p = image[x: x + self.size, y: y + self.size, :3]
        m = mask[x: x + self.size, y: y + self.size]

        p = torchvision.transforms.ToTensor()(p)

        m = torchvision.transforms.Resize((100, 100), interpolation=Image.NEAREST)(Image.fromarray(m))
        m = torchvision.transforms.ToTensor()(m)

        return p, m
