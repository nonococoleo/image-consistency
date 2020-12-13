import torchvision
import os
import torch
import numpy as np
import itertools
from compareExifModel import CompareExifModel
from predictExifModel import PredictExifModel


def get_resnet(name, pretrained=False):
    resnets = {
        "resnet18": torchvision.models.resnet18(pretrained=pretrained),
        "resnet50": torchvision.models.resnet50(pretrained=pretrained),
    }
    if name not in resnets.keys():
        raise KeyError(f"{name} is not a valid ResNet version")
    return resnets[name]


def save_model(model_path, model, current_epoch):
    out = os.path.join(model_path, "checkpoint_{}.tar".format(current_epoch))

    if isinstance(model, torch.nn.DataParallel):
        torch.save(model.module.state_dict(), out)
    else:
        torch.save(model.state_dict(), out)


# compare one patch with all other patches
def evaluatePatch(features, eval_model, device):
    x = torch.stack(features).to(device)
    with torch.no_grad():
        output = eval_model(x)
    output = output.detach()
    return output.sum().tolist()


# calculate consistency map
def getConsistencyMap(image, size, exif_model, eval_model, device):
    height = image.shape[1]
    width = image.shape[2]
    patches = []
    for i in range(height // size):
        for j in range(width // size):
            x = size * i
            y = size * j
            xs = min(x + size, height)
            ys = min(y + size, width)
            p = image[:, x:xs, y:ys]
            patches.append(p)

    length = len(patches)
    patch_pairs = itertools.product(patches, patches)
    feature_vector = []
    results = []
    counter = 0
    for x, y in patch_pairs:
        x = torch.stack((x,)).to(device)
        y = torch.stack((y,)).to(device)
        with torch.no_grad():
            output = exif_model(x, y)
        output = output.detach()
        feature_vector.extend(output)
        if len(feature_vector) == length:
            score = evaluatePatch(feature_vector, eval_model, device)
            counter += 1
            print("patch %d: %f" % (counter, score), flush=True)
            results.append(score)
            feature_vector.clear()

    consistency_map = np.array(results).reshape(height // size, width // size)
    return consistency_map


def get_output(model, x, y):
    with torch.no_grad():
        if type(model) == PredictExifModel:
            output_x = model(x)
            output_y = model(y)
            output = (torch.argmax(output_x, 1) == torch.argmax(output_y, 1)).float()
        elif type(model) == CompareExifModel:
            output = model(x, y)
            output = output.detach()
        else:
            raise NotImplemented
    return output


def get_features(model, loader, device):
    feature_vector = []
    labels_vector = []
    for (step, ((x, y), label)) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)
        label = label.to(device)
        output = get_output(model, x, y)

        feature_vector.extend(output.cpu().numpy())
        labels_vector.extend(label.cpu().numpy())

        if step % 20 == 0:
            print(f"Step [{step}/{len(loader)}]\t Computing features...", flush=True)

    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    print("Features shape {}".format(feature_vector.shape))
    return feature_vector, labels_vector


def create_data_loaders_from_arrays(X_train, y_train, batch_size):
    train = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train)
    )
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=False
    )

    return train_loader
