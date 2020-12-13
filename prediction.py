from utils import *
from predictExifModel import PredictExifModel
from predictExifDataset import PredictExifDataset


def collate_fn(batch):
    images = []
    labels = []
    for i, l in batch:
        images.append(i)
        labels.append(l)
    return torch.stack(images), torch.stack(labels)


def train(device, loader, model, criterion, optimizer):
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
        if step % 20 == 0:
            print(f"Step [{step}/{len(loader)}]\t Loss: {loss.item()}", flush=True)

    return loss_epoch


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device('cpu')

    train_dataset = PredictExifDataset("datasets/exif/train.csv", "datasets/exif/images", 10)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=False,
        batch_size=20,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # middle dim with 6 starts with epoch 100
    start_epoch = 201
    model_folder = "models"
    model_path = os.path.join(model_folder, "checkpoint_{}.tar".format(start_epoch - 1))
    encoder = get_resnet('resnet18', pretrained=True)
    n_features = encoder.fc.out_features  # get dimensions of fc layer

    model = PredictExifModel(encoder, n_features, 200, 6, 10)
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
            save_model(model_folder, model, epoch)

        print(f"Epoch [{epoch}/{epochs}]\t Loss: {loss_epoch / len(train_loader)}\t lr: {round(lr, 5)}", flush=True)
