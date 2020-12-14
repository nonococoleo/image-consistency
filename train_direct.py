from utilities import *
from directModel import Direct
from directDataLoader import DirectDataset


def collate_fn(batch):
    images = []
    labels = []
    for i, l in batch:
        images.append(i)
        labels.append(l)
    return torch.stack(images), torch.stack(labels)


def train(device, loader, model, criterion, optimizer):
    loss_epoch = 0
    for (step, (image, mask)) in enumerate(loader):
        torch.set_printoptions(edgeitems=20000)
        optimizer.zero_grad()
        image = image.to(device)

        output = model(image)
        mask = mask.view(mask.size(0), 100, 100)
        mask0 = mask.float()
        mask1 = torch.ones(mask.size(0), 100, 100) - mask0
        mask = torch.stack([mask0, mask1], dim = 1)
        mask = mask.permute(0, 2, 3, 1)
        mask = mask.to(device)

        loss = criterion(output, mask)
        # print('loss', loss.item())

        loss.backward()
        optimizer.step()

        loss_epoch += loss.item()
        # if step % 20 == 0:
        # print(f"Step [{step}/{len(loader)}]\t Loss: {loss.item()}", flush=True)

    return loss_epoch


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device('cpu')

    # middle dim with 6 starts with epoch 100
    start_epoch = 1
    model_folder = "models"
    model_path = os.path.join(model_folder, "checkpoint_direct{}.tar".format(start_epoch-1))
    model = Direct()
    # model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
    criterion = torch.nn.BCELoss()

    train_dataset = DirectDataset("datasets/label_in_wild")
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=False,
        batch_size=7,
        collate_fn=collate_fn,
        num_workers=4,
    )

    epochs = 5000
    for epoch in range(start_epoch, epochs):
        loss_epoch = train(device, train_loader, model, criterion, optimizer)
        # if epoch % 10 == 0:
        save_model(model, '', epoch)

        print(f"Epoch [{epoch}/{epochs}]\t Loss: {loss_epoch / len(train_loader)}\t lr: {round(3e-4, 5)}", flush=True)

