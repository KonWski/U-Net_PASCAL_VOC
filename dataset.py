from torchvision.datasets import VOCSegmentation
from torchvision import transforms
from torchvision.transforms.functional import vflip
import cv2
import numpy as np
import torch
from typing import List
import math
import torch.nn.functional as F
import random
from PIL import Image
from torchvision.transforms.functional import to_tensor

class PascalVOCSegmentation(VOCSegmentation):

    def __init__(
            self, 
            root: str, 
            year: str, 
            image_set: str, 
            download: bool,
            selected_classes: List[str],
            augmentation: bool = False
            ):
        '''
        Dataset limiting original VOCSegmentation to specified class_name

        root: str
            path to PascalVOC dataset content
        year: str
            year of competition "2007" to "2012"
        image_set: str
            "train", "trainval", "test"
        download: bool
            True -> downloads dataset from repo
            False -> uses already existing dataset
        selected_classes: List[str]:
            names of selected classes
        augmentation: bool = False
            perform augmentation using parallel transformations on image and mask        
        '''
        super().__init__(root, year, image_set, download, transforms=transforms)
        self.selected_classes = selected_classes
        self.augmentation = augmentation
        self._classes_names = [
            "background",
            "aeroplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "diningtable",
            "dog",
            "horse",
            "motorbike",
            "person",
            "potted plant",
            "sheep",
            "sofa",
            "train",
            "tv/monitor",
            "border"
        ]

        all_classes_available = all([selected_class in self._classes_names for selected_class in self.selected_classes])
        assert all_classes_available, "Not all specified classes are available"

        # enoding of classes in mask
        self._color_map = [
            [0, 0, 0],
            [128, 0, 0],
            [0, 128, 0],
            [128, 128, 0],
            [0, 0, 128],
            [128, 0, 128],
            [0, 128, 128],
            [128, 128, 128],
            [64, 0, 0],
            [192, 0, 0],
            [64, 128, 0],
            [192, 128, 0],
            [64, 0, 128],
            [192, 0, 128],
            [64, 128, 128],
            [192, 128, 128],
            [0, 64, 0],
            [128, 64, 0],
            [0, 192, 0],
            [128, 192, 0],
            [0, 64, 128],
            [224, 224, 192]
        ]

        # class_name -> (id, color in mask)
        self.class_to_color = {selected_class: (id, self._color_map[self._classes_names.index(selected_class)]) 
                               for id, selected_class in enumerate(self.selected_classes)}


    def _transform(self, image, mask):

        # random vertical flip
        if random.random() > 0.5:
            image = vflip(image)
            mask = vflip(mask)
                    
        return image, mask


    def __getitem__(self, idx):
        
        # image = cv2.imread(self.images[idx])
        # mask = cv2.imread(self.masks[idx])

        image = Image.open(self.images[idx])
        mask = Image.open(self.masks[idx])

        if self.augmentation:
            image, mask = self._transform(image, mask)

        image = to_tensor(image)
        mask = to_tensor(mask)

        # additional channel for background
        encoded_mask = torch.zeros([mask.shape[0], mask.shape[1], len(self.selected_classes) + 1])
        
        # convert color encoding into channel encoding -
        # - each channel refers to specific class
        for selected_class in self.selected_classes:
            
            channel_id = self.class_to_color[selected_class][0]
            class_color_encoding = self.class_to_color[selected_class][1]
            class_pixels_indices = np.where(mask == class_color_encoding)
            encoded_mask[class_pixels_indices[0], class_pixels_indices[1], channel_id] = 1        

        return image, encoded_mask
    

# 572x572
def split_image_mask(image: torch.Tensor, mask: torch.Tensor, splitted_mask_size: int, default_boundary: int):
    '''
    Converts image and mask into smaller pieces. Each of smaller pieces of image have the same shape.
    The same rule goes for each smaller piece of mask.
    
    image: torch.Tensor
        image split into smaller parts
    mask: torch.Tensor
        mask split into smaller parts
    splitted_mask_size: int
        width and height of smaller piece of mask
    default_boundary: int:
        padding size around cut out image piece
    '''
    
    # temporarily change order of dimensions in mask
    mask = mask.view(2, 0, 1)

    # cut out parts of original image and mask
    output_subimages = []
    output_submasks = []
    
    image_height = image.shape[1]
    image_width = image.shape[2]
    
    # n_rows x n_cols is the number of output smaller parts
    n_rows = math.ceil(image_height / splitted_mask_size)
    print(f"n_rows: {n_rows}")
    n_cols = math.ceil(image_width / splitted_mask_size)
    print(f"n_cols: {n_cols}")

    for n_row in range(n_rows):
        
        # upper and lower border of mask/image piece
        print(f"n_row: {n_row}")
        row_split_0 = n_row * splitted_mask_size
        print(f"row_split_0: {row_split_0}")
        row_split_1 = (n_row + 1) * splitted_mask_size
        print(f"row_split_1: {row_split_1}")
        
        # corner case - lower border exceeds original image
        if row_split_1 > image_height:
            padding_img_bottom = default_boundary + (row_split_1 - image_height)
            padding_mask_bottom = row_split_1 - image_height
        else:
            padding_img_bottom = default_boundary
            padding_mask_bottom = 0
        
        for n_column in range(n_cols):
            
            # left and right border of mask/image piece
            print(f"n_column: {n_column}")
            column_split_0 = n_column * splitted_mask_size
            print(f"column_split_0: {column_split_0}")
            column_split_1 = (n_column + 1) * splitted_mask_size
            print(f"column_split_1: {column_split_1}")
            
            # corner case - right border exceeds original image
            if column_split_1 > image_width:
                padding_img_right = default_boundary + (column_split_1 - image_width)
                padding_mask_right = column_split_1 - image_width
            else:
                padding_img_right = default_boundary
                padding_mask_right = 0
            
            '''
            padding size for image and mask pieces
            padding size (left, right, top, bottom)
            '''
            padding_img = (default_boundary, padding_img_right, default_boundary, padding_img_bottom)
            print(f"padding_img: {padding_img}")
            padding_mask = (0, padding_mask_right, 0, padding_mask_bottom)
            print(f"padding_mask: {padding_mask}")

            # extract subimage and submask from input image and mask
            sub_image = image[:, row_split_0:row_split_1, column_split_0:column_split_1]
            sub_mask = mask[:, row_split_0:row_split_1, column_split_0:column_split_1]            

            # return to original dimensions order in piece of mask
            sub_mask = sub_mask.view(1, 2, 0)

            print(f"sub_image before padding shape: {sub_image.shape}")
            print(f"sub_mask before padding shape: {sub_mask.shape}")
            
            # add padding for image and mask piece
            sub_image = F.pad(sub_image, padding_img)
            sub_mask = F.pad(sub_mask, padding_mask)
            
            print(f"sub_image after padding shape: {sub_image.shape}")
            print(f"sub_mask after padding shape: {sub_mask.shape}")            
            
            # collect next pieces of image and mask part
            output_subimages.append(sub_image)
            output_submasks.append(sub_mask)
    
    return output_subimages, output_submasks