from utilities import *
import torch.nn as nn
from consistency_model import ConsistencyModel
from feature_dataset import FeatureDataset


def collate_fn(batch):
    """
    collate function for feature dataset
    :param batch: data batch
    :return: organized data
    """

    images_a = []
    images_b = []
    labels = []
    for i in batch:
        images_a.append(i['p1'])
        images_b.append(i['p2'])
        labels.append(i['score'])
    return (torch.stack(images_a), torch.stack(images_b)), torch.stack(labels).reshape(-1, 1)


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


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device('cpu')

    model_type = "predict"
    # model_type = "compare"

    feature_epoch = 300
    feature_model_folder = "feature_model"
    feature_model_path = os.path.join(feature_model_folder, "checkpoint_{}.tar".format(feature_epoch))
    encoder = get_resnet('resnet50', pretrained=True)
    n_features = encoder.fc.out_features  # get dimensions of fc layer

    feature_dim = 6
    if model_type == "compare":
        feature_model = CompareExifModel(encoder, n_features, feature_dim)
    elif model_type == "predict":
        feature_model = PredictExifModel(encoder, n_features, feature_dim, partitions=10)
    else:
        raise NotImplemented
    feature_model.load_state_dict(torch.load(feature_model_path, map_location=device))
    feature_model.to(device)
    feature_model.eval()

    feature_dataset = FeatureDataset('datasets/label_in_wild', num_pairs=2, patch_size=128)
    feature_loader = torch.utils.data.DataLoader(
        feature_dataset,
        shuffle=True,
        batch_size=64,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

    train_X, train_y = get_data(feature_model, feature_loader, device)
    train_dataset = torch.utils.data.TensorDataset(train_X, train_y)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=False)

    consistency_epoch = 300
    consistency_model_folder = "consistency_model"
    consistency_model_path = os.path.join(consistency_model_folder, "checkpoint_{}.tar".format(consistency_epoch))
    consistency_model = ConsistencyModel(feature_dim)
    # consistency_model.load_state_dict(torch.load(consistency_model_path, map_location=device))
    consistency_model.to(device)
    optimizer = torch.optim.Adam(consistency_model.parameters(), lr=1e-4)
    criterion = nn.BCELoss()

    logistic_epochs = 300
    for epoch in range(1, logistic_epochs + 1):
        lr = optimizer.param_groups[0]["lr"]
        loss_epoch = train(device, train_loader, consistency_model, criterion, optimizer)

        if epoch % 10 == 0:
            save_model(consistency_model, consistency_model_folder, epoch)

        print(f"Epoch [{epoch}/{logistic_epochs}]\t Loss: {loss_epoch / len(train_loader)}\t lr: {round(lr, 5)}", flush=True)
