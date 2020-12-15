import torch
import numpy as np
import matplotlib.pyplot as plt
from direct_model import DirectModel
from direct_dataset import DirectDataset


def imshow_gray(inp):
    mean = 0.1307
    std = 0.3081
    inp = ((mean * inp) + std)
    plt.imshow(inp, cmap='gray')
    plt.show()


def imshow_color(img, is_unnormlize=False):
    print(img.shape)
    if is_unnormlize:
        img = img / 2 + 0.5  # unnormalize
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show()


if __name__ == '__main__':
    model_path='models/checkpoint_6790.tar'
    direct = DirectModel()
    # direct.load_state_dict(torch.load(model_path))
    direct.eval()

    # one image
    # img = Image.open('datasets/label_in_wild/images/2508.jpg').convert('RGB')
    # img = torchvision.transforms.Resize((400, 400), interpolation=Image.NEAREST)(img)
    # img = torchvision.transforms.ToTensor()(img)
    # imshow_color(img.view(3, 400, 400))
    # output = direct(img.view(1, 3, 400, 400))
    # output = output.view(100, 100, 2)[:, :, 0]
    # output = output.detach().numpy()
    # print(output)
    # imshow_gray(output)

    # iterate over all evaluation images
    eval_dataset = DirectDataset("datasets/label_in_wild_eval")
    evalloader = torch.utils.data.DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=0)
    iterator = iter(evalloader)

    for img, _ in iterator:
        imshow_color(img.view(3, 400, 400))
        output = direct(img.view(1, 3, 400, 400))
        output = output.view(100, 100, 2)[:, :, 0]
        output = output.detach().numpy()
        imshow_gray(output)
