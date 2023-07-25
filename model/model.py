import torch
import torch.nn as nn


class NASBlock(nn.Module):
    def __init__(self, num_channels, num_stages=1):
        super(NASBlock, self).__init__()
        layers = [nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
                  nn.BatchNorm2d(num_channels),
                  nn.ReLU()]
        for _ in range(num_stages - 1):
            layers += [nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
                       nn.BatchNorm2d(num_channels),
                       nn.ReLU()]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class SuperNet(nn.Module):
    def __init__(self,
                 input_channels=1,
                 num_classes=10,
                 block_num_stages=(1, 2, 3)):
        super(SuperNet, self).__init__()

        self.conv = nn.Sequential(
                nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True))

        self.nasblock1_choice = None
        self.nas_block1_variants = nn.ModuleList(
            [NASBlock(num_channels=64, num_stages=num_stages) for num_stages in block_num_stages])

        self.conv_stride = nn.Sequential(
                nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True))

        self.nasblock2_choice = None
        self.nas_block2_variants = nn.ModuleList(
            [NASBlock(num_channels=32, num_stages=num_stages) for num_stages in block_num_stages])

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.nas_block1_variants[self.nasblock1_choice](x)
        x = self.conv_stride(x)
        x = self.nas_block2_variants[self.nasblock2_choice](x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def sample_subnet(self, num_stages_nasblock1=None, num_stages_nasblock2=None):
        self.nasblock1_choice = num_stages_nasblock1 if num_stages_nasblock1 is not None \
            else torch.randint(0, 3, size=(1,)).item()
        self.nasblock2_choice = num_stages_nasblock2 if num_stages_nasblock2 is not None \
            else torch.randint(0, 3, size=(1,)).item()
