import torchvision


class RandomCropTransformer:
    """
    Transformer for Random crop in given size
    """

    def __init__(self, size):
        self.train_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomCrop(size=size),
                torchvision.transforms.ToTensor(),
                # torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

    def __call__(self, x):
        return self.train_transform(x)
