# U-Net_PASCAL_VOC [Under construction]
Following repository is an example implementation of U-Net neural network architecture introduced by Olaf Ronneberger et. al. in [Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597). Built model isn't exactly the same as in original paper because of different type of data used in project. However Readme consists not only of short explanations refering to original paper but also of instructions on how to train and use U-net inspired neural net.

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

# How to train model