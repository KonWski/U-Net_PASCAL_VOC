from torch import nn, Tensor, concat
from torch.nn import ConvTranspose2d, MaxPool2d, Conv2d
import math
import torch
import logging
import argparse

class uNetContractingBlock(nn.Module):

    def __init__(self, depth_level: int):

        super().__init__()

        self.depth_level = depth_level
        self.input_channels = 3 if depth_level == 0 else int(64 * math.pow(2, depth_level - 1))
        self.output_channels = 64 * int(math.pow(2, depth_level))

        self.conv1 = Conv2d(self.input_channels, self.output_channels, 3, bias=False)
        self.conv2 = Conv2d(self.output_channels, self.output_channels, 3, bias=False)
        self.maxpool = MaxPool2d(2, stride=2)
    
    def forward(self, x: Tensor):

        x_coppied = self.conv1(x)
        x_coppied = self.conv2(x_coppied)
        x_pooled = self.maxpool(x_coppied)

        return x_coppied, x_pooled


class uNetBottleneck(nn.Module):

    def __init__(self, depth_level: int):

        super().__init__()

        self.depth_level = depth_level
        self.input_channels = int(64 * math.pow(2, depth_level - 1))
        self.mid_channels = int(64 * math.pow(2, depth_level))
        self.output_channels = self.mid_channels

        self.conv1 = Conv2d(self.input_channels, self.mid_channels, 3, bias=False)
        self.conv2 = Conv2d(self.mid_channels, self.mid_channels, 3, bias=False)

    def forward(self, x: Tensor):

        x = self.conv1(x)
        x = self.conv2(x)

        return x


class uNetExpandingBlock(nn.Module):

    def __init__(self, depth_level: int):

        super().__init__()

        self.depth_level = depth_level
        self.input_channels = int(64 * math.pow(2, depth_level + 1))
        self.mid_channels = int(64 * math.pow(2, depth_level))
        self.output_channels = self.mid_channels
        
        self.upsample = ConvTranspose2d(self.input_channels, self.input_channels, 3, 2)
        self.conv1 = Conv2d(self.input_channels, self.mid_channels, 2, bias=False)
        self.conv2 = Conv2d(self.input_channels, self.mid_channels, 3, bias=False)
        self.conv3 = Conv2d(self.mid_channels, self.output_channels, 3, bias=False)


    def forward(self, x_previous_layer: Tensor, x_coppied: Tensor):
        
        # upsampling
        x = self.upsample(x_previous_layer)
        x = self.conv1(x)

        # concatenation
        x_cropped = x_coppied[:, :, :x.shape[2], :x.shape[3]]
        x = concat([x, x_cropped], 1)

        # convolution part
        x = self.conv2(x)
        x = self.conv3(x)

        return x


class uNetPascalVOC(nn.Module):

    def __init__(self, max_depth_level: int, n_classes: int):

        super().__init__()
        self.max_depth_level = max_depth_level

        self.contracting_path = nn.ModuleList([uNetContractingBlock(depth_level) for depth_level in range(self.max_depth_level)])
        self.bottleneck = uNetBottleneck(self.max_depth_level)
        self.expanding_path = nn.ModuleList([uNetExpandingBlock(depth_level) for depth_level in reversed(range(self.max_depth_level))])
        self.final_layer = Conv2d(self.expanding_path[-1].output_channels, n_classes, 1, bias=False)

    def forward(self, x: Tensor):
        
        coppied_tensors = []
        for contracting_block in self.contracting_path:
            x_coppied, x = contracting_block(x)
            coppied_tensors.append(x_coppied)

        x = self.bottleneck(x)

        for expanding_block, coppied_tensor in zip(self.expanding_path, reversed(coppied_tensors)):
            x = expanding_block(x, coppied_tensor)

        x = self.final_layer(x)

        return x
    

def test_model(height: int, width: int, n_channels: int, max_depth_level: int, n_classes: int):
    
    dummy_model = uNetPascalVOC(max_depth_level, n_classes)
    dummy_tensor = torch.zeros([1, n_channels, height, width])

    output = dummy_model(dummy_tensor)
    logging.info(f"Output shape: {output.shape}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Paramaters for dummy model test')
    parser.add_argument('--height', type=int)
    parser.add_argument('--width', type=int)
    parser.add_argument('--n_channels', type=int)
    parser.add_argument('--max_depth_level', type=int)
    parser.add_argument('--n_classes', type=int)

    logging.basicConfig(level=logging.INFO)
    args = vars(parser.parse_args())

    test_model(args["height"], args["width"], args["n_channels"], args["max_depth_level"], args["n_classes"])