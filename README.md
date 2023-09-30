# U-Net_PASCAL_VOC
Following repository is an example implementation of U-Net neural network architecture introduced by Olaf Ronneberger et. al. in [Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597). Following document consists of short explanations refering to the original paper and of instructions on how to train and use U-net inspired neural net.

Data used for project come from PASCAL VOC dataset available through Torchvision. Model was created using PyTorch library.

# U-Net
The issue that faced the authors of U-Net came directly from medical industry - the task was not only to classify images with HeLa cells but also to show the areas of their occurence.

| ![example_raw_image](/images/example_raw_image.PNG) |
|:--:|
| (a) raw image, (b) overlay with ground truth segmentations [1]|

[1]: https://arxiv.org/abs/1505.04597

This goal came with several difficulties that had to be adressed. Among those was missing context at the borders of images - not all cells were fully included inside the picture. That's why the "Overlap-tile strategy" was introduced which filled the missing parts using mirroring of actual image. The consequence of this ideas is visible directly in architecture of U-Net which operates on input tensors that are bigger than the output ones. Moreover the authors decided to cut whole images into smaller pieces for better computing efficiency. 

| ![overlap_tile_strategy](/images/overlap_tile_strategy.PNG) |
|:--:|
| Overlap-tile strategy [2]|

[2]: https://arxiv.org/abs/1505.04597

U-Net owes its name to characteristic structure consisting of contracting and expansive paths which set by each other form an u-shape. Beside a solid connection between two paths (visible at the bottom of figure 3) U-Net at 4 points transfers cropped pieces of feature maps from contracting to expansive part. Copied fragments are merged with equivalent map. The reason standing behind sending only cropped maps is the 'overlapping-tile strategy' - tensors at the contracting path are bigger than their counterparts on expanding path.

Contracting path is a repeating combination of two convolutional layers (filter 3x3, ReLU activation function) followed by max pooling layer (filter 2x2). Before each applying max pooling layer cropped tensors are sent on the expanding path. The expanding path itself is constructed using sequences of transposed convolution layer (2x2 layer) and two convolutional layers (filter 3x3, ReLU activation function). No fully connected layers were used.

| ![u_net_architecture](/images/u_net_architecture.PNG) |
|:--:|
| U-Net architecture [3]|

[3]: https://arxiv.org/abs/1505.04597

One of the main challenges of the project was to avoid situation in which separate cells were marked as one big field. To distinguish those cells authors used weight function which prioritises boundaries between cells and balances class frequencies (in this case 2):

$w(\textbf{x}) = w_c(\textbf{x}) + w_0 \cdot exp(-\frac{(d_1(\textbf{x}) + d_2(\textbf{x}))^2}{2\sigma^2})$,

where:
-  $\textbf{x}$ represents pixel's coordinates ($x \in \Omega, \Omega \subset \textbf{Z}^2$),
- $w_c:\Omega \rightarrow \textbf{R}$ returns value of weight map balancing class frequencies
- $d_1:\Omega \rightarrow \textbf{R}$ returns distance of pixel $\textbf{x}$ to the border of nearest cell
- $d_2:\Omega \rightarrow \textbf{R}$ returns distance of pixel $\textbf{x}$ to the border of second nearest cell

For the sake of the experiment $w_0$ was set to 10 and $\sigma$ to 5 pixels.

Thanks to the weight map created by $w$ function the authors could enhance the calculation of error value:

$E = \sum_{\textbf{x} \in \Omega} w(\textbf{x}) log(p_{l(\textbf{x})}(\textbf{x}))$,

which penalizes small U-Net's soft max values for correct class. In this case $l:\Omega \rightarrow \{1,..,K\}$ returns correct class for pixel $\textbf{x}$, where K is the number of classes.

# How to work with project

## Training a model
```
!python /content/U-Net_PASCAL_VOC/main.py --n_epochs 400 \
                                          --checkpoints_dir 'drive/MyDrive/Colab Notebooks/tutoriale/u-net/modele'\
                                          --download_datasets 'true' \
                                          --root_datasets_dir 'PASCAL_VOC_dataset' \
                                          --years_train '2007,2008,2009,2010,2011,2012' \
                                          --years_test '2007,2008,2009,2010' \
                                          --selected_classes 'border,cow' \
                                          --splitted_mask_size 100 \
                                          --default_boundary_size 98 \
                                          --use_balanced_class_weights 'N' \
                                          --initialize_model_weights 'Y' \
                                          --load_model 'N'
```

Args used in command:
- n_epochs - number of epochs
- checkpoints_dir - path to directory where checkpoint will be saved
- download_datasets - download dataset from Torchvision repo or use already existing dataset
- root_datasets_dir - path where dataset should be downloaded or where is it already stored
- years_train - years of Pascal VOC competition "2007" to "2012" separated by commas used for training
- years_test - years of Pascal VOC competition "2007" to "2012" separated by commas used for testing
- selected_classes - classes seperated by commas
- splitted_mask_size - width and height of smaller piece of mask
- default_boundary_size - padding size around cut out image piece
- use_balanced_class_weights - use balanced  weights for unequal classes distribiution
- initialize_model_weights - initialize model weights using normal distribiution
- load_model - Y -> continue learning using existing model and optimizer

## Visualizing masks
Import libraries:
```python
from model import load_checkpoint, uNetPascalVOC
from dataset import PascalVOCSegmentation
from visualize import concat_split_mask, show_mask
from torch.utils.data import DataLoader
from torchvision.utils import draw_segmentation_masks
from torchvision.transforms.functional import to_tensor
import matplotlib.pyplot as plt
import torch
```

Initialise Dataset and Dataloader
```python
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# parameters for dataset
selected_classes = ["background", "border", "cow"]
splitted_mask_size = 100
default_boundary = 98

# initiate dataset, dataloader
dataset = PascalVOCSegmentation("/path/to/your/dataset", "2007", "test", selected_classes, splitted_mask_size, default_boundary, False, download=True)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
```

Load model
```python
# load checkpoint
checkpoint = torch.load("path/to/your/checkpoint")

# initiate model
model = uNetPascalVOC(checkpoint["max_depth_level"], len(checkpoint["selected_classes"]))
model.to(device)

# load parameters from checkpoint
model.load_state_dict(checkpoint["model_state_dict"])
```

Visualize mask
```python
selected_img = 10
selected_class = 2

for id, batch in enumerate(dataloader, 0):

    if id == selected_img:

        # extract data from batch
        image, split_image, split_mask = batch
        image = image.permute(0, 3, 1, 2)

        split_image = [img_piece[0] for img_piece in split_image]
        split_image = torch.stack(split_image)

        # run model on extracted data
        split_image = split_image.to(device)

        # softmax output and concat to a single image
        outputs = model(split_image).to(device)
        outputs = torch.nn.functional.softmax(outputs, dim=1)
        outputs = concat_split_mask(outputs, image, 3)

        show_mask(image[0], outputs, selected_class)
        break
```
Example visualization:

![mask_visualization](/images/mask_visualization.png)