from utilities import *
from predict_exif_model import PredictExifModel
from predict_exif_dataset import PredictExifDataset


def collate_fn(batch):
    """
    collate function for predict feature dataset
    :param batch: data batch
    :return: organized data
    """

    images = []
    labels = []
    for i, l in batch:
        images.append(i)
        labels.append(l)
    return torch.stack(images), torch.stack(labels)


def train(device, loader, model, criterion, optimizer):
    """
    train the model for one epoch
    :param device: device type
    :param loader: dataloader
    :param model: the model
    :param criterion: loss function
    :param optimizer: the optimizer
    :return: training loss
    """

    loss_epoch = 0
    model.train()
    for (step, (image, label)) in enumerate(loader):
        optimizer.zero_grad()

        image = image.to(device)
        label = label.to(device)

        output = model(image)
        loss = criterion(output, label)

        loss.backward()
        optimizer.step()

        loss_epoch += loss.item()
        if step % 50 == 0:
            print(f"Step [{step}/{len(loader)}]\t Loss: {loss.item()}", flush=True)

    return loss_epoch


def test(device, loader, model, criterion):
    """
    test the model for one epoch
    :param device: device type
    :param loader: dataloader
    :param model: the model
    :param criterion: loss function
    :return: testing loss
    """

    loss_epoch = 0
    model.eval()
    for (step, ((x, y), label)) in enumerate(loader):
        model.zero_grad()

        x = x.to(device)
        y = y.to(device)
        label = label.to(device)

        output = model(x, y)
        loss = criterion(output, label)

        loss_epoch += loss.item()

    return loss_epoch


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device('cpu')

    train_dataset = PredictExifDataset("datasets/exif/train.csv", "datasets/exif/images", replica=10, patch_size=128)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=16,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

    start_epoch = 1
    model_folder = "feature_model"
    model_path = os.path.join(model_folder, "checkpoint_{}.tar".format(start_epoch - 1))
    encoder = get_resnet('resnet50', pretrained=True)
    n_features = encoder.fc.out_features  # get dimensions of fc layer

    model = PredictExifModel(encoder, n_features, labels=6, partitions=10)
    # model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
    criterion = torch.nn.CrossEntropyLoss()

    epochs = 500
    for epoch in range(start_epoch, epochs + 1):
        lr = optimizer.param_groups[0]["lr"]
        loss_epoch = train(device, train_loader, model, criterion, optimizer)

        # save every 10 epochs
        if epoch % 10 == 0:
            save_model(model, model_folder, epoch)

        print(f"Epoch [{epoch}/{epochs}]\t Loss: {loss_epoch / len(train_loader)}\t lr: {round(lr, 5)}", flush=True)

    print("finish training")

    test_dataset = PredictExifDataset("datasets/exif/test.csv", "datasets/exif/images", replica=1, patch_size=128)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=True,
        batch_size=64,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

    print(test(device, test_loader, model, criterion))
