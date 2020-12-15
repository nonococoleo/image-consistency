from utilities import *
from compare_exif_dataset import CompareExifDataset


def collate_fn(batch):
    """
    collate function for compare exif dataset
    :param batch: data batch
    :return: organized data
    """

    images_a = []
    images_b = []
    labels = []
    for i in batch:
        images_a.append(i['image'][0])
        images_b.append(i['image'][1])
        labels.append(i['label'])
    return (torch.stack(images_a), torch.stack(images_b)), torch.stack(labels)


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
    for (step, ((x, y), label)) in enumerate(loader):
        optimizer.zero_grad()

        x = x.to(device)
        y = y.to(device)
        label = label.to(device)

        output = model(x, y)
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

    train_dataset = CompareExifDataset("datasets/exif/train.csv", "datasets/exif/images", num_pairs=8192, patch_size=128)
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

    model = CompareExifModel(encoder, n_features, 6)
    #    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    criterion = torch.nn.BCELoss()

    epochs = 300
    for epoch in range(start_epoch, epochs + 1):
        lr = optimizer.param_groups[0]["lr"]
        loss_epoch = train(device, train_loader, model, criterion, optimizer)

        # save every 10 epochs
        if epoch % 10 == 0:
            save_model(model, model_folder, epoch)

        print(f"Epoch [{epoch}/{epochs}]\t Loss: {loss_epoch / len(train_loader)}\t lr: {round(lr, 5)}", flush=True)

    print("finish training")

    test_dataset = CompareExifDataset("datasets/exif/test.csv", "datasets/exif/images", num_pairs=2048, patch_size=128)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=True,
        batch_size=64,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

    print(test(device, test_loader, model, criterion))
