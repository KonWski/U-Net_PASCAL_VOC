from torch import nn, Tensor, concat
from torch.nn import ConvTranspose2d, MaxPool2d, Conv2d, BatchNorm2d, ReLU
import math
import torch
import logging
import argparse
from torch.nn.init import normal_
from torch.optim import Adam

class uNetContractingBlock(nn.Module):

    def __init__(self, depth_level: int):

        super().__init__()

        self.depth_level = depth_level
        self.input_channels = 3 if depth_level == 0 else int(64 * math.pow(2, depth_level - 1))
        self.output_channels = 64 * int(math.pow(2, depth_level))
        self.activation_function = ReLU()

        self.conv1 = Conv2d(self.input_channels, self.output_channels, 3, bias=False)
        self.batchnorm1 = BatchNorm2d(self.output_channels)
        self.conv2 = Conv2d(self.output_channels, self.output_channels, 3, bias=False)
        self.batchnorm2 = BatchNorm2d(self.output_channels)
        self.maxpool = MaxPool2d(2, stride=2)
    
    def forward(self, x: Tensor):

        x_coppied = self.conv1(x)
        x_coppied = self.batchnorm1(x_coppied)
        x_coppied = self.activation_function(x_coppied)

        x_coppied = self.conv2(x_coppied)
        x_coppied = self.batchnorm2(x_coppied)
        x_coppied = self.activation_function(x_coppied)

        x_pooled = self.maxpool(x_coppied)

        return x_coppied, x_pooled


class uNetBottleneck(nn.Module):

    def __init__(self, depth_level: int):

        super().__init__()

        self.depth_level = depth_level
        self.input_channels = int(64 * math.pow(2, depth_level - 1))
        self.mid_channels = int(64 * math.pow(2, depth_level))
        self.output_channels = self.mid_channels
        self.activation_function = ReLU()

        self.conv1 = Conv2d(self.input_channels, self.mid_channels, 3, bias=False)
        self.batchnorm1 = BatchNorm2d(self.mid_channels)
        self.conv2 = Conv2d(self.mid_channels, self.mid_channels, 3, bias=False)
        self.batchnorm2 = BatchNorm2d(self.mid_channels)

    def forward(self, x: Tensor):

        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.activation_function(x)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.activation_function(x)

        return x


class uNetExpandingBlock(nn.Module):

    def __init__(self, depth_level: int):

        super().__init__()

        self.depth_level = depth_level
        self.input_channels = int(64 * math.pow(2, depth_level + 1))
        self.mid_channels = int(64 * math.pow(2, depth_level))
        self.output_channels = self.mid_channels
        self.activation_function = ReLU()        
        
        self.upsample = ConvTranspose2d(self.input_channels, self.input_channels, 3, 2)
        self.conv1 = Conv2d(self.input_channels, self.mid_channels, 2, bias=False)
        self.batchnorm1 = BatchNorm2d(self.mid_channels)
        self.conv2 = Conv2d(self.input_channels, self.mid_channels, 3, bias=False)
        self.batchnorm2 = BatchNorm2d(self.mid_channels)
        self.conv3 = Conv2d(self.mid_channels, self.output_channels, 3, bias=False)
        self.batchnorm3 = BatchNorm2d(self.output_channels)

    def forward(self, x_previous_layer: Tensor, x_coppied: Tensor):
        
        # upsampling
        x = self.upsample(x_previous_layer)
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.activation_function(x)

        # concatenation
        x_cropped = x_coppied[:, :, :x.shape[2], :x.shape[3]]
        x = concat([x, x_cropped], 1)

        # convolution part
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.activation_function(x)

        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.activation_function(x)

        return x


class uNetPascalVOC(nn.Module):

    def __init__(self, max_depth_level: int, n_classes: int, inititialize_weights: bool = False):

        super().__init__()
        self.max_depth_level = max_depth_level

        self.contracting_path = nn.ModuleList([uNetContractingBlock(depth_level) for depth_level in range(self.max_depth_level)])
        self.bottleneck = uNetBottleneck(self.max_depth_level)
        self.expanding_path = nn.ModuleList([uNetExpandingBlock(depth_level) for depth_level in reversed(range(self.max_depth_level))])
        self.final_layer = Conv2d(self.expanding_path[-1].output_channels, n_classes, 1, bias=False)

        if inititialize_weights:
            self.apply(self._init_model_weights)

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

    def _init_model_weights(self, module):

        if isinstance(module, Conv2d) or isinstance(module, BatchNorm2d) or isinstance(module, ConvTranspose2d):
            module.weight.data.normal_(mean=0.0, std=0.5)


def save_checkpoint(checkpoint: dict, checkpoint_path: str):
    '''
    saves checkpoint on given checkpoint_path
    '''
    torch.save(checkpoint, checkpoint_path)

    logging.info(8*"-")
    logging.info(f"Saved model to checkpoint: {checkpoint_path}")
    logging.info(f"Epoch: {checkpoint['epoch']}")
    logging.info(8*"-")


def load_checkpoint(model: uNetPascalVOC, optimizer: torch.optim, checkpoint_path: str):
    '''
    loads model checkpoint from given path

    Parameters
    ----------
    model : uNetPascalVOC
    optimizer : torch.optim
    checkpoint_path : str
        Path to checkpoint

    Notes
    -----
    checkpoint: dict
                parameters retrieved from training process i.e.:
                - last finished number of epoch
                - depth level of model
                - selected classes
                - model_state_dict
                - optimizer_state_dict
                - save time
    '''
    checkpoint = torch.load(checkpoint_path)

    # initiate model, optimizer
    model = uNetPascalVOC(checkpoint["max_depth_level"], len(checkpoint["selected_classes"]))
    optimizer = Adam(model.parameters(), lr=1e-5)

    # load parameters from checkpoint
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # print loaded parameters
    logging.info(f"Loaded model from checkpoint: {checkpoint_path}")
    logging.info(f"Epoch: {checkpoint['epoch']}")
    logging.info(f"Save dttm: {checkpoint['save_dttm']}")
    logging.info(f"Selected classes: {checkpoint['selected_classes']}")
    logging.info(f"Depth level of model: {checkpoint['max_depth_level']}")
    logging.info(f"Test loss: {checkpoint['test_loss']}")

    logging.info(8*"-")

    return model, optimizer, checkpoint


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