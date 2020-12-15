import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


class DirectModel(nn.Module):
    def __init__(self):
        super(DirectModel,self).__init__()

        # define encoder
        res50 = torchvision.models.__dict__['resnet50']()
        # res50.load_state_dict(state_dict, strict=False)
        res50_C4_layers = []
        for name, module in res50.named_children():
            if name in ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3']:
                res50_C4_layers.append(module)
        self.encoder = nn.Sequential(*res50_C4_layers)

        # define transformation
        self.fc_transform = nn.Sequential(
            nn.Linear(25 * 25, 25 * 25),
            nn.ReLU(),
            nn.Linear(25 * 25, 100 * 100),
            nn.ReLU()
        )

        pool_scales = [1, 10, 30, 50, 70]
        ppm = []
        for scale in pool_scales:
            ppm.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(1024, 64, kernel_size=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            ))
        self.ppm = nn.ModuleList(ppm)

        self.upsample = nn.Upsample((100, 100), mode='bilinear', align_corners=False)

        self.conv_last = nn.Sequential(
            nn.Conv2d(320, 50, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(50),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(50, 2, kernel_size=1)
        )

    def forward(self, y):
        l = len(y[:, 0, 0, 0])
        x = y.view(l, 3, 400, 400)

        x = self.encoder(x)
        # print('2 shape', x.shape)
        x = x.view(x.size(0), x.size(1), x.size(2) * x.size(3))
        # print('3 shape', x.shape)
        view_comb = self.fc_transform(x)
        # print('4 shape', view_comb.shape)
        view_comb = view_comb.view(x.size(0), x.size(1), 100, 100)
        # view_comb = self.upsample(view_comb)

        ppm_out = []
        for pool_scale in self.ppm:
            out = self.upsample(pool_scale(view_comb))
            ppm_out.append(out)
        # # print('5 shape', x.shape)
        ppm_out = torch.cat(ppm_out, 1)

        x = self.conv_last(ppm_out)
        # 10 * 2 * 100 * 100
        # print('6 shape', x.shape)

        x = x.permute(0, 2, 3, 1)
        x = F.softmax(x, dim=3)
        # print('7 shape', x.shape)

        return x
