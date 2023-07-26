import torch
import torch.nn as nn


class NASBlock(nn.Module):
    """A building block for the SuperNet that applies a sequence of convolutional layers
       with batch normalization and ReLU activation.

        Args:
            num_channels (int): The number of input and output channels for the convolutional layers.
            num_stages (int): The number of times to apply the convolutional layers.

        Attributes:
            block (nn.Sequential): A sequence of convolutional layers with batch normalization and ReLU activation.
    """

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
        """Applies the NASBlock to the input tensor.

        Args:
            x (torch.Tensor): A tensor of shape (batch_size, num_channels, height, width).

        Returns:
            torch.Tensor: A tensor of shape (batch_size, num_channels, height, width) with the NASBlock applied.
        """

        return self.block(x)


class SuperNet(nn.Module):
    """A class for one-shot neural architecture search that uses two NASBlocks
       to process images and classify them into categories.

    Args:
        input_channels (int): The number of input channels for the network.
        num_classes (int): The number of output classes for the network.
        block_num_stages (tuple of int): A tuple of three integers that specifies the number of times to apply
            the subblocks for each NASBlock.

    Attributes:
        conv (nn.Sequential): A sequence of convolutional layers with batch normalization and ReLU activation.
        nasblock1_choice (int or None): An integer that specifies which variant of the first NASBlock
            to use in the network, or None if it has not been sampled yet.
        nas_block1_variants (nn.ModuleList): A list of three NASBlock instances with different numbers of stages.
        conv_stride (nn.Sequential): A sequence of convolutional layers with batch normalization and ReLU activation
            that reduces the spatial dimensions of the input tensor.
        nasblock2_choice (int or None): An integer that specifies which variant of the second NASBlock
            to use in the network, or None if it has not been sampled yet.
        nas_block2_variants (nn.ModuleList): A list of three NASBlock instances with different numbers of stages.
        global_pool (nn.AdaptiveAvgPool2d): A pooling layer that reduces the spatial dimensions
            of the input tensor to 1x1.
        fc (nn.Linear): A linear layer that maps the output of the global pooling layer to the number of output classes.
    """

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
        """Applies the SuperNet to the input tensor.

        Args:
            x (torch.Tensor): A tensor of shape (batch_size, input_channels, height, width).

        Returns:
            torch.Tensor: A tensor of shape (batch_size, num_classes) with the predicted scores for each class.
        """

        x = self.conv(x)
        x = self.nas_block1_variants[self.nasblock1_choice](x)
        x = self.conv_stride(x)
        x = self.nas_block2_variants[self.nasblock2_choice](x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def sample_subnet(self, num_stages_nasblock1=None, num_stages_nasblock2=None):
        """Samples a subnet from the SuperNet by selecting variants of the NASBlocks.

        Args:
            num_stages_nasblock1 (int or None): An integer that specifies which variant of the first NASBlock to use,
                or None to sample randomly.
            num_stages_nasblock2 (int or None): An integer that specifies which variant of the second NASBlock to use,
                or None to sample randomly.
        """

        self.nasblock1_choice = num_stages_nasblock1 if num_stages_nasblock1 is not None \
            else torch.randint(0, 3, size=(1,)).item()
        self.nasblock2_choice = num_stages_nasblock2 if num_stages_nasblock2 is not None \
            else torch.randint(0, 3, size=(1,)).item()
