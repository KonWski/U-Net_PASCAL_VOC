from typing import List
from math import ceil
import torch

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
    print(f"original_image.shape: {original_image.shape}")
    print(f"split_mask[0].shape: {split_mask[0].shape}")

    # all pieces of mask have the same size
    height_mask_split = split_mask[0].shape[1]
    width_mask_split = split_mask[0].shape[2]

    original_image_height = original_image.shape[2]
    original_image_width = original_image.shape[3]
    print(f"original_image_height: {original_image_height}")
    print(f"original_image_width: {original_image_width}")

    mask = torch.zeros([n_selected_classes, original_image_height, original_image_width])
    print(f"mask.shape: {mask.shape}")

    n_rows = ceil(original_image_height / height_mask_split)
    n_cols = ceil(original_image_width / width_mask_split)
    print(f"n_rows: {n_rows}")
    print(f"n_cols: {n_cols}")

    mask_split_iterator = 0

    for n_row in range(n_rows):

        for n_col in range(n_cols):

            mask_piece = split_mask[mask_split_iterator]

            height_index_0 = n_row * height_mask_split
            height_index_1 = min((n_row + 1) * height_mask_split, original_image_height)
            width_index_0 = n_col * width_mask_split
            width_index_1 = min((n_col + 1) * width_mask_split, original_image_width)

            try:
                cut_mask_piece = mask_piece[:, 0 : height_index_1 - height_index_0, 0 : width_index_1 - width_index_0]
                mask[:, height_index_0 : height_index_1, width_index_0 : width_index_1] = cut_mask_piece
                mask_split_iterator += 1
            except:
                print(f"Height indices -> {height_index_0}:{height_index_1}, Width indices -> {width_index_0}:{width_index_1}")
                print(f"mask piece shape: {mask_piece[:, 0 : height_index_1 - height_index_0, 0 : width_index_1 - width_index_0].shape}")

    return mask