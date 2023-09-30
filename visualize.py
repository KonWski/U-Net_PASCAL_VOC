from typing import List
from math import ceil
import torch
import matplotlib.pyplot as plt

def concat_split_mask(split_mask: List[torch.Tensor], original_image: torch.Tensor, n_selected_classes: int):
    '''
    Concatenates all output masks into one. Pieces of mask should be cut and concatenated in a way that it fits the original image.

    split_mask: List[torch.Tensor]
        list of outputted by model mask
    original_image: torch.Tensor
        image to which mask refer
    n_selected_classes: int
        number of classes to which mask refer
    '''

    # all pieces of mask have the same size
    height_mask_split = split_mask[0].shape[1]
    width_mask_split = split_mask[0].shape[2]

    original_image_height = original_image.shape[2]
    original_image_width = original_image.shape[3]

    mask = torch.zeros([n_selected_classes, original_image_height, original_image_width])

    n_rows = ceil(original_image_height / height_mask_split)
    n_cols = ceil(original_image_width / width_mask_split)

    mask_split_iterator = 0

    for n_row in range(n_rows):

        for n_col in range(n_cols):

            mask_piece = split_mask[mask_split_iterator]

            height_index_0 = n_row * height_mask_split
            height_index_1 = min((n_row + 1) * height_mask_split, original_image_height)
            width_index_0 = n_col * width_mask_split
            width_index_1 = min((n_col + 1) * width_mask_split, original_image_width)

            cut_mask_piece = mask_piece[:, 0 : height_index_1 - height_index_0, 0 : width_index_1 - width_index_0]
            mask[:, height_index_0 : height_index_1, width_index_0 : width_index_1] = cut_mask_piece
            mask_split_iterator += 1

    return mask


def show_mask(original_image: torch.Tensor, mask: torch.Tensor, presented_class: int):
    '''
    shows pair of images - original image and its generated mask (for presented_class)
    
    Parameters
    ----------
    original_image: torch.Tensor
        input for the model generating mask
    mask: torch.Tensor
        mask generated from original image
    presented_class: int
        class which will be emphasized in the visualization
    '''

    # mark pixels with class with biggest output value
    highest_class = torch.argmax(mask, 0)
    unique_classes = highest_class.unique()

    # turn tensor's values into 1-0 
    for value in unique_classes:
        value = value.item()
        boolean_mask = (highest_class == value)

    # select only those pixels reffering to presented class
    boolean_mask = (highest_class == presented_class)

    # create grid with 1 row and 2 columns
    fig, ax = plt.subplots(nrows = 1, ncols = 2)

    ax[0].imshow(original_image.permute(1, 2, 0))
    ax[0].axis("off")
    
    ax[1].imshow(boolean_mask.int())
    ax[1].axis("off")