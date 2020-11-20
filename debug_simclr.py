from nt_xent import NTXentLoss
import torch
from PIL import Image
import torchvision
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, dataset_path, image_count): 

         self.dataset_path = dataset_path
         self.image_count = image_count
         self.transforms = transforms.Compose([
         	transforms.FiveCrop(10),
         	transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            transforms.Lambda(lambda tensors:
                torch.stack([t for t in tensors]))
           ])

    def __getitem__(self, index):
        image = Image.open(self.dataset_path+str(index)+".jpg").convert('RGB')
        
        return self.transforms(image)

    def __len__(self):
        return self.image_count

batch_size = 10
temperature = 0.5
training_images_count = 10 # number of training images


train_data = CustomDataset('dataset/cvFinalData/train/', training_images_count)
train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

class SimCLR(nn.Module):
    def __init__(self, encoder, n_features, projection_dim):
        super(SimCLR, self).__init__()

        self.encoder = encoder
        self.n_features = n_features

        self.projector = nn.Sequential(
            nn.Linear(self.n_features, 300, bias=False),
            nn.ReLU(),
            nn.Linear(300, projection_dim),
            nn.ReLU()
        )

    def forward(self, x_i, x_j):
        h_i = self.encoder(x_i)
        h_j = self.encoder(x_j)

        z_i = self.projector(h_i)
        z_j = self.projector(h_j)

        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        return z_i, z_j

if __name__ == '__main__':

    encoder = torchvision.models.resnet18(pretrained=False)
    n_features = 1000
    projection_dim = 128
    device = 'cpu'
    lr = 0.001


    model = SimCLR(encoder, n_features, projection_dim)
    loss_function = NTXentLoss(device, batch_size, temperature, True)  
    optimizer = torch.optim.SGD(model.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_data_loader), eta_min=0,
                                                               last_epoch=-1)

    epochs = 10000

    for epoch_counter in range(epochs):
        loss_sum = 0
        counter = 0
        print('Epoch: ', epoch_counter)
        for xs in train_data_loader:
            xis = xs[:, 0, :, :, :] # patch from one corner
            xjs = xs[:, 1, :, :, :] # patch from the other corner
            optimizer.zero_grad()

            zi, zj = model(xis, xjs)

            loss = loss_function(zi, zj)

            loss_sum += loss

            loss.backward()
            optimizer.step()
            counter += 1
            # print("Loss: ", (str(loss)))

            if counter % 1 == 0:
            	print("Training ",counter * batch_size,"/", training_images_count, " Loss: ", loss_sum / (batch_size))
            	loss_sum = 0










