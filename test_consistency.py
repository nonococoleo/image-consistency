from utilities import *
from PIL import Image
from consistency_model import ConsistencyModel


def evaluate(device, feature_model, consistency_model, image, size, threshold):
    consistency_map = get_consistency_map(image, size, feature_model, consistency_model, device)
    #     if_sliced = ifSliced(consistency_map, threshold)

    return consistency_map


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

    consistency_epoch = 300
    consistency_model_folder = "consistency_model"
    consistency_model_path = os.path.join(consistency_model_folder, "checkpoint_{}.tar".format(consistency_epoch))
    consistency_model = ConsistencyModel(feature_dim)
    consistency_model.load_state_dict(torch.load(consistency_model_path, map_location=device))
    consistency_model.to(device)

    # Evaluating
    patch_size = 128
    threshold = 0.5
    consistency_model.eval()

    image = Image.open(os.path.join('datasets/label_in_wild/images/2251.jpg'))
    image = torchvision.transforms.ToTensor()(image)

    print(evaluate(device, feature_model, consistency_model, image, patch_size, threshold))
