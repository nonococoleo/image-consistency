from utilities import *
from direct_model import DirectModel
from direct_dataset import DirectDataset


def collate_fn(batch):
    """
    collate function for direct dataset
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
    for (step, (image, mask)) in enumerate(loader):
        torch.set_printoptions(edgeitems=20000)
        optimizer.zero_grad()
        image = image.to(device)

        output = model(image)
        mask = mask.view(mask.size(0), 100, 100)
        mask0 = mask.float()
        mask1 = torch.ones(mask.size(0), 100, 100) - mask0
        mask = torch.stack([mask0, mask1], dim=1)
        mask = mask.permute(0, 2, 3, 1)
        mask = mask.to(device)

        loss = criterion(output, mask)

        loss.backward()
        optimizer.step()

        loss_epoch += loss.item()
        if step % 50 == 0:
            print(f"Step [{step}/{len(loader)}]\t Loss: {loss.item()}", flush=True)

    return loss_epoch


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device('cpu')

    train_dataset = DirectDataset("datasets/label_in_wild")
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=16,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

    # middle dim with 6 starts with epoch 100
    start_epoch = 1
    model_folder = "direct_models"
    model_path = os.path.join(model_folder, "checkpoint_{}.tar".format(start_epoch - 1))
    model = DirectModel()
    # model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
    criterion = torch.nn.BCELoss()

    epochs = 5000
    for epoch in range(start_epoch, epochs):
        lr = optimizer.param_groups[0]["lr"]
        loss_epoch = train(device, train_loader, model, criterion, optimizer)

        # save every 10 epochs
        if epoch % 10 == 0:
            save_model(model, model_folder, epoch)

        print(f"Epoch [{epoch}/{epochs}]\t Loss: {loss_epoch / len(train_loader)}\t lr: {round(lr, 5)}", flush=True)
