from PIL import Image
import math
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from torchvision import utils
from simclr import SimCLR


def imshow(img, is_unnormlize=False):
    print(img.shape)
    if is_unnormlize:
        img = img / 2 + 0.5  # unnormalize
    plt.imshow(np.transpose(img, (1, 2, 0)))
    # plt.imshow(img)
    plt.show()


def crop(img, size=20):
    res = []
    height = img.shape[0]
    width = img.shape[1]
    for i in range(height // size):
        for j in range(width // size):
            x = size * i
            y = size * j
            xs = min(x + size, height)
            ys = min(y + size, width)
            part = img[x:xs, y:ys]
            res.append(part)
    return np.array(res)


def get_features(simclr_model, device, img):
    x = torch.tensor(img, dtype=torch.float32).to(device)

    # get encoding
    with torch.no_grad():
        h, _, z, _ = simclr_model(x, x)

    return z.detach().numpy()


def get_mask(imgs):
    count = 0
    for i in imgs.sum(axis=(1, 2)):
        print(i, end='\t')
        count += 1
        if count % 32 == 0:
            print()


def get_resnet(name, pretrained=False):
    resnets = {
        "resnet18": torchvision.models.resnet18(pretrained=pretrained),
        "resnet50": torchvision.models.resnet50(pretrained=pretrained),
    }
    if name not in resnets.keys():
        raise KeyError(f"{name} is not a valid ResNet version")
    return resnets[name]


# calculate cross entropy
def cross_entropy(p, q):
    return -sum([p[i] * math.log2(q[i]) for i in range(len(p))])


def cos_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


if __name__ == '__main__':
    filename = "2395"
    image_path = "dataset/label_in_wild/images/"
    mask_path = "dataset/label_in_wild/masks/"
    img = Image.open(image_path + filename + ".jpg")

    img_size = 100
    npimg = np.array(img)
    patches = crop(npimg, img_size)
    # print(patches.shape)

    # t = torch.tensor(np.transpose(imgs, (0, 3, 1, 2)))
    # grid = utils.make_grid(t, nrow=npimg.shape[1] // 50, padding=0).numpy()
    # imshow(grid)

    # train_dataset = torchvision.datasets.ImageFolder('dataset/label_in_wild')
    #
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     batch_size=1,
    #     collate_fn=lambda x: tuple(zip(*x))
    # )
    # dataiter = iter(train_loader)
    # images, labels = dataiter.next()
    # img = images[0]
    # npimg = np.transpose(np.array(img), (2, 0, 1))
    # imshow(npimg)

    # get features
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device('cpu')
    resnet = 'resnet50'
    projection_dim = 64
    # model_path = "models/checkpoint_100.tar"

    encoder = get_resnet(resnet, pretrained=False)
    n_features = encoder.fc.in_features

    simclr_model = SimCLR(encoder, n_features, projection_dim)
    # simclr_model.load_state_dict(torch.load(model_path, map_location=device))
    simclr_model = simclr_model.to(device)

    f = get_features(simclr_model, device, np.transpose(patches, (0, 3, 1, 2)))
    print(f.shape)

    # calculate similarity
    res = []
    num_patches = patches.shape[0]
    nrows = img.size[0] // img_size
    for i in range(num_patches):
        temp = 0
        for j in range(num_patches):
            temp += cos_sim(f[i], f[j])
        res.append(temp)

    for i in range(len(res)):
        if i % nrows == 0:
            print()
        print(res[i], end="\t")
    print()

    # calculate mask
    img = Image.open(mask_path + filename + ".png")
    npimg = np.array(img)
    imgs = crop(npimg, img_size)
    masks = imgs.sum(axis=(1, 2))

    for i in range(num_patches):
        if i % nrows == 0:
            print()
        print(masks[i], end="\t")
    print()
