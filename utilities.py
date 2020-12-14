import os
import torch
import torchvision
import numpy as np
from compare_exif_model import CompareExifModel
from predict_exif_model import PredictExifModel


def get_resnet(name, pretrained=False):
    """
    get resnet layer
    :param name: type of resnet
    :param pretrained: whether use pretrained model
    :return: resnet layer
    """

    resnets = {
        "resnet18": torchvision.models.resnet18(pretrained=pretrained),
        "resnet50": torchvision.models.resnet50(pretrained=pretrained),
    }
    if name not in resnets.keys():
        raise KeyError(f"{name} is not a valid ResNet version")
    return resnets[name]


def save_model(model, model_path, epoch):
    """
    save given model to target path
    :param model: the model object
    :param model_path: target file path
    :param epoch: training epoch
    :return: None
    """

    out = os.path.join(model_path, "checkpoint_{}.tar".format(epoch))

    if isinstance(model, torch.nn.DataParallel):
        torch.save(model.module.state_dict(), out)
    else:
        torch.save(model.state_dict(), out)


def get_consistency_score(features, consistency_model, device):
    """
    calculate consistency score of given feature vectors
    :param features: feature vectors between patch pairs
    :param consistency_model: the consistency model
    :param device: device type
    :return: list of consistency scores
    """

    x = torch.stack(features).to(device)
    with torch.no_grad():
        output = consistency_model(x)
    output = output.detach()
    return output.sum().tolist()


# calculate consistency map
def get_consistency_map(image, size, feature_model, consistency_model, device):
    """
    calculate consistency map of given image by calculate every patch split by the image
    :param image: the image object
    :param size: patch edge size
    :param feature_model: the feature model
    :param consistency_model: the consistency model
    :param device: device type
    :return: consistency map of given image
    """

    import itertools

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
        output = get_feature(feature_model, x, y)
        feature_vector.extend(output)
        if len(feature_vector) == length:
            score = get_consistency_score(feature_vector, consistency_model, device)
            counter += 1
            print("patch %d: %f" % (counter, score), flush=True)
            results.append(score)
            feature_vector.clear()

    consistency_map = np.array(results).reshape(height // size, width // size)
    return consistency_map


def get_feature(feature_model, x, y):
    """
    get the feature vector of given patch pair
    :param feature_model: the feature model
    :param x: patch x
    :param y: patch y
    :return: feature vector
    """

    with torch.no_grad():
        if type(feature_model) == PredictExifModel:
            output_x = feature_model(x)
            output_y = feature_model(y)
            output = (torch.argmax(output_x, 1) == torch.argmax(output_y, 1)).float()
        elif type(feature_model) == CompareExifModel:
            output = feature_model(x, y)
        else:
            raise NotImplemented
    return output.detach()


def get_data(model, loader, device):
    """
    get feature and label array of all patch pairs
    :param model: the feature model
    :param loader: patch pairs dataloader
    :param device: device type
    :return: feature and label array
    """
    features = []
    labels = []
    for (step, ((x, y), label)) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)
        output = get_feature(model, x, y)

        features.extend(output.cpu().numpy())
        labels.extend(label.cpu().numpy())

        if step % 20 == 0:
            print(f"Step [{step}/{len(loader)}]\t Computing features...", flush=True)

    features = torch.tensor(features)
    labels = torch.tensor(labels)
    print("Features shape {}".format(features.shape))
    return features, labels
