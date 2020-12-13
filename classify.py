from featureDataset import FeatureDataset
from featureModel import FeatureModel
import torch.nn as nn
from PIL import Image
from utils import *


def collate_fn(batch):
    images_a = []
    images_b = []
    labels = []
    for i in batch:
        images_a.append(i['p1'])
        images_b.append(i['p2'])
        labels.append(i['score'])
    return (torch.stack(images_a), torch.stack(images_b)), torch.stack(labels).reshape(-1, 1)


def train(device, loader, model, criterion, optimizer):
    loss_epoch = 0
    model.train()
    for step, (x, score) in enumerate(loader):
        optimizer.zero_grad()

        x = x.to(device)
        score = score.to(device)

        output = model(x)
        loss = criterion(output, score)

        loss.backward()
        optimizer.step()

        loss_epoch += loss.item()
        if step % 50 == 0:
            print(f"Step [{step}/{len(loader)}]\t Loss: {loss.item()}", flush=True)

    return loss_epoch


def evaluate(device, exif_model, eval_model, image, size, threshold):
    consistency_map = getConsistencyMap(image, size, exif_model, eval_model, device)
    #     if_sliced = ifSliced(consistency_map, threshold)

    return consistency_map


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device('cpu')

    model_type = "predict"
    # model_type = "compare"

    exif_epoch = 300
    exif_model_folder = "exif_model"
    exif_model_path = os.path.join(exif_model_folder, "checkpoint_{}.tar".format(exif_epoch))
    encoder = get_resnet('resnet50', pretrained=True)
    n_features = encoder.fc.out_features  # get dimensions of fc layer
    exif_dim = 6
    if model_type == "compare":
        exif_model = CompareExifModel(encoder, n_features, exif_dim)
    elif model_type == "predict":
        exif_model = PredictExifModel(encoder, n_features, 6, 6, 10)
    else:
        raise NotImplemented
    # exif_model.load_state_dict(torch.load(exif_model_path, map_location=device))
    exif_model.to(device)
    exif_model.eval()

    patch_size = 128
    train_dataset = FeatureDataset('datasets/label_in_wild', 2, patch_size)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=64,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

    train_X, train_y = get_features(exif_model, train_loader, device)
    arr_train_loader = create_data_loaders_from_arrays(train_X, train_y, 128)

    eval_epoch = 300
    eval_model_folder = "eval_model"
    eval_model_path = os.path.join(eval_model_folder, "checkpoint_{}.tar".format(eval_epoch))
    eval_model = FeatureModel(exif_dim)
    # eval_model.load_state_dict(torch.load(eval_model_path, map_location=device))
    eval_model.to(device)
    optimizer = torch.optim.Adam(eval_model.parameters(), lr=1e-4)
    criterion = nn.BCELoss()

    logistic_epochs = 300
    for epoch in range(1, logistic_epochs + 1):
        lr = optimizer.param_groups[0]["lr"]
        loss_epoch = train(device, arr_train_loader, eval_model, criterion, optimizer)

        if epoch % 10 == 0:
            save_model(eval_model_folder, eval_model, epoch)

        print(f"Epoch [{epoch}/{logistic_epochs}]\t Loss: {loss_epoch / len(train_loader)}\t lr: {round(lr, 5)}",
              flush=True)

    # Evaluating
    threshold = 0.5
    eval_model.eval()

    image = Image.open(os.path.join('datasets/label_in_wild/images/2251.jpg'))
    image = torchvision.transforms.ToTensor()(image)

    print(evaluate(device, exif_model, eval_model, image, patch_size, threshold))
