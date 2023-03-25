from torch import nn, Tensor, concat
from torch.nn import ConvTranspose2d, MaxPool2d, Conv2d
import math

class uNetContractingBlock(nn.Module):

    def __init__(self, depth_level: int):

        super().__init__()

        self.depth_level = depth_level
        self.input_channels = 3 if depth_level == 0 else 64 * math.pow(2, depth_level - 1)
        self.output_channels = 64 * math.pow(2, depth_level)

        self.conv1 = Conv2d(self.input_channels, self.output_channels, 3, 2, bias=False)
        self.conv2 = Conv2d(self.output_channels, self.output_channels, 3, 2, bias=False)
        self.maxpool = MaxPool2d(2)
    
    def forward(self, x: Tensor):
        
        x_coppied = self.conv1(x)
        x_coppied = self.conv2(x_coppied)
        x_pooled = self.maxpool(x_coppied)
        return x_coppied, x_pooled


class uNetBottleneck(nn.Module):

    def __init__(self, depth_level: int):

        super().__init__()

        self.depth_level = depth_level
        self.input_channels = 64 * math.pow(2, depth_level - 1)
        self.mid_channels = 64 * math.pow(2, depth_level)
        self.output_channels = self.input_channels

        self.conv1 = Conv2d(self.input_channels, self.mid_channels, 3, 2, bias=False)
        self.conv2 = Conv2d(self.mid_channels, self.mid_channels, 3, 2, bias=False)
        self.convtranspose = ConvTranspose2d(self.mid_channels, self.output_channels, 1, bias=False)

    def forward(self, x: Tensor):
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.convtranspose(x)
        return x


class uNetExpandingBlock(nn.Module):

    def __init__(self, depth_level: int):

        super().__init__()

        self.depth_level = depth_level
        self.input_channels = 64 * math.pow(2, depth_level + 1)
        self.mid_channels = 64 * math.pow(2, depth_level)
        self.output_channels = 64 * math.pow(2, depth_level - 1)
        
        self.conv1 = Conv2d(self.input_channels, self.output_channels, 3, 2, bias=False)
        self.conv2 = Conv2d(self.output_channels, self.output_channels, 3, 2, bias=False)
        self.convtranspose = ConvTranspose2d(self.output_channels, self.output_channels, 1, bias=False)

    def forward(self, x_coppied: Tensor, x_previous_layer: Tensor):
        
        print(f"x_coppied.shape: {x_coppied.shape}")
        print(f"x_previous_layer.shape: {x_previous_layer.shape}")
        
        x_cropped = x_coppied[:, :, x_previous_layer.shape[1], x_previous_layer.shape[2]]
        x = concat(x_cropped, x_previous_layer)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.convtranspose(x)
        return x


class uNetCityscapes(nn.Module):

    def __init__(self, max_depth_level: int, n_classes: int):

        super().__init__()
        self.max_depth_level = max_depth_level

        self.contracting_path = [uNetContractingBlock(depth_level) for depth_level in range(self.max_depth_level)]
        self.bottleneck = uNetBottleneck(self.max_depth_level)
        self.expanding_path = [uNetExpandingBlock(depth_level) for depth_level in reversed(range(self.max_depth_level))]
        self.final_layer = Conv2d(self.output_channels, n_classes, 1, bias=False)

    def forward(self, x: Tensor):
        
        coppied_tensors = []
        for contracting_block in self.contracting_path:
            x_coppied, x = contracting_block(x)
            coppied_tensors.append(x_coppied)

        x = self.bottleneck(x)

        for expanding_block in self.expanding_path:
            x = expanding_block(x)

        return x