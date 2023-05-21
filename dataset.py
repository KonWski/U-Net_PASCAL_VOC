from torchvision.datasets import VOCSegmentation
from torchvision.transforms.functional import hflip
import cv2
import numpy as np
import torch
from typing import List
import math
import torch.nn.functional as F
import random
from torchvision.transforms.functional import to_tensor
from torchvision import transforms

class PascalVOCSegmentation(VOCSegmentation):

    def __init__(
            self, 
            root: str, 
            year: str, 
            image_set: str, 
            selected_classes: List[str],
            splitted_mask_size: int, 
            default_boundary: int,
            augmentation: bool,
            download: bool
            ):
        '''
        Dataset limiting original VOCSegmentation to specified class_name

        root: str
            path to PascalVOC dataset content
        year: str
            year of competition "2007" to "2012"
        image_set: str
            "train", "trainval", "test"
            True -> downloads dataset from repo
            False -> uses already existing dataset
        selected_classes: List[str]:
            names of selected classes
        splitted_mask_size: int
            width and height of smaller piece of mask
        default_boundary: int:
            padding size around cut out image piece
        augmentation: bool = False
            perform augmentation using parallel transformations on image and mask        
        download: bool
            download dataset from repo
        '''
        super().__init__(root, year, image_set, download)
        self.splitted_mask_size = splitted_mask_size
        self.default_boundary = default_boundary        
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

        self.selected_classes = self._classes_names if selected_classes[0] == "all" else selected_classes

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
        # print(self.class_to_color)

    def _transform(self, image, mask):
        
        # shape of mask before flip should look like this: (class_dim, height, width)
        mask = mask.permute(2, 0, 1)

        # random horizontal flip
        if random.random() > 0.5:
            image = hflip(image)
            mask = hflip(mask)
        
        # return to original dimensions order
        mask = mask.permute(1, 2, 0)

        return image, mask


    def _split_image_mask(self, image: torch.Tensor, mask: torch.Tensor):
        '''
        Converts image and mask into smaller pieces. Each of smaller pieces of image have the same shape.
        The same rule goes for each smaller piece of mask.
        
        image: torch.Tensor
            image split into smaller parts
        mask: torch.Tensor
            mask split into smaller parts
        '''
        
        # temporarily change order of dimensions in mask
        mask = mask.permute(2, 0, 1)

        # print(f"split_image_mask method mask.shape: {mask.shape}")
        # print(f"split_image_mask method image.shape: {image.shape}")

        # cut out parts of original image and mask
        output_subimages = []
        output_submasks = []
        
        image_height = image.shape[1]
        # print(f"image_height: {image_height}")
        image_width = image.shape[2]
        # print(f"image_width: {image_width}")

        # n_rows x n_cols is the number of output smaller parts
        n_rows = math.ceil(image_height / self.splitted_mask_size)
        # print(f"n_rows: {n_rows}")
        n_cols = math.ceil(image_width / self.splitted_mask_size)
        # print(f"n_cols: {n_cols}")

        for n_row in range(n_rows):
            
            # upper and lower border of mask/image piece
            # print(f"n_row: {n_row}")
            
            row_split_mask_0 = n_row * self.splitted_mask_size
            row_split_img_0 = max(0, row_split_mask_0 - self.default_boundary)
            # row_split_img_0 = row_split_mask_0

            row_split_mask_1 = (n_row + 1) * self.splitted_mask_size
            # print(f"row_split_1: {row_split_1}")

            row_split_img_1 = min(image_height, row_split_mask_1 + self.default_boundary)
            # print(f"row_split_image_1: {row_split_img_1}")

            # TODO fix - gives negative values
            padding_img_top = abs(row_split_mask_0 - self.default_boundary) - row_split_img_0

            padding_img_bottom = (row_split_mask_1 + self.default_boundary) - row_split_img_1
            # print(f"padding_img_top: {padding_img_top}")
            # print(f"padding_img_bottom: {padding_img_bottom}")

            # corner case - lower border exceeds original image
            if row_split_mask_1 > image_height:
                # padding_img_bottom = self.default_boundary + (row_split_mask_1 - image_height)
                padding_mask_bottom = row_split_mask_1 - image_height
            else:
                # padding_img_bottom = self.default_boundary
                padding_mask_bottom = 0
            
            for n_column in range(n_cols):
                
                # left and right border of mask/image piece
                column_split_mask_0 = n_column * self.splitted_mask_size
                column_split_img_0 = max(0, column_split_mask_0 - self.default_boundary)
                # column_split_img_0 = column_split_mask_0

                column_split_mask_1 = (n_column + 1) * self.splitted_mask_size
                column_split_img_1 = min(image_width, column_split_mask_1 + self.default_boundary)
                
                # TODO fix - gives negative values
                padding_img_left = abs(column_split_mask_0 - self.default_boundary) - column_split_img_0

                padding_img_right = (column_split_mask_1 + self.default_boundary) - column_split_img_1

                # corner case - right border exceeds original image
                if column_split_mask_1 > image_width:
                    # padding_img_right = self.default_boundary + (column_split_mask_1 - image_width)
                    padding_mask_right = column_split_mask_1 - image_width
                else:
                    # padding_img_right = self.default_boundary
                    padding_mask_right = 0
                
                '''
                padding size for image and mask pieces
                padding size (left, right, top, bottom)
                '''
                # padding_img = (self.default_boundary, padding_img_right, self.default_boundary, padding_img_bottom)
                padding_img = (padding_img_left, padding_img_right, padding_img_top, padding_img_bottom)
                # print(f"padding_img: {padding_img}")
                padding_mask = (0, padding_mask_right, 0, padding_mask_bottom)
                # print(f"padding_mask: {padding_mask}")

                # extract subimage and submask from input image and mask
                # sub_image = image[:, row_split_mask_0:row_split_mask_1, column_split_0:column_split_1]
                sub_image = image[:, row_split_img_0:row_split_img_1, column_split_img_0:column_split_img_1]
                sub_mask = mask[:, row_split_mask_0:row_split_mask_1, column_split_mask_0:column_split_mask_1]            

                # print(f"Image, height: {row_split_img_0}:{row_split_img_1}, width: {column_split_img_0}:{column_split_img_1}")
                # print(f"Mask, height: {row_split_mask_0}:{row_split_mask_1}, width: {column_split_mask_0}:{column_split_mask_1}")

                # return to original dimensions order in piece of mask
                # sub_mask = sub_mask.permute(1, 2, 0)

                # print(f"sub_image before padding shape: {sub_image.shape}")
                # print(f"sub_mask before padding shape: {sub_mask.shape}")
                
                # add padding for image and mask piece
                sub_image = F.pad(sub_image, padding_img)
                sub_mask = F.pad(sub_mask, padding_mask)

                # set part of mask that was extended by padding as background
                # print(f"sub_mask.shape: {sub_mask.shape}")
                # print(f"padding_mask_right: {padding_mask_right}")
                # print(f"padding_mask_bottom: {padding_mask_bottom}")

                if padding_mask_right > 0:
                    sub_mask[0, :, -padding_mask_right:] = torch.ones([sub_mask.shape[1], padding_mask_right])
                
                if padding_mask_bottom > 0:
                    sub_mask[0, -padding_mask_bottom:, :] = torch.ones([padding_mask_bottom, sub_mask.shape[2]])


                # print(f"sub_image after padding shape: {sub_image.shape}")
                # print(f"sub_mask after padding shape: {sub_mask.shape}")            
                
                # collect next pieces of image and mask part
                output_subimages.append(sub_image)
                output_submasks.append(sub_mask)
        
        return output_subimages, output_submasks


    def __getitem__(self, idx):
        
        image_vis = cv2.imread(self.images[idx])
        image_vis = cv2.cvtColor(image_vis, cv2.COLOR_BGR2RGB)
        
        transform_image = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        image = transform_image(image_vis)

        mask = cv2.imread(self.masks[idx])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        # additional channel for background
        encoded_mask = torch.zeros([mask.shape[0], mask.shape[1], len(self.selected_classes)])

        # information if selected class was found in the picture
        no_selected_classes_found = True

        # convert color encoding into channel encoding -
        # - each channel refers to specific class
        for selected_class in self.selected_classes:
            
            channel_id = self.class_to_color[selected_class][0]
            class_color_encoding = self.class_to_color[selected_class][1]

            class_pixels_indices = np.all(mask == class_color_encoding, -1)
            found_class = np.any(class_pixels_indices)
            class_pixels_indices = torch.Tensor(np.where(class_pixels_indices, 1, 0))

            # array empty if no pixels belong to specified class
            if found_class:
                encoded_mask[:, :, channel_id] = class_pixels_indices
                no_selected_classes_found = False if selected_class not in ("background", "border") else no_selected_classes_found

        if no_selected_classes_found:
            split_image, split_mask = torch.Tensor(), torch.Tensor()

        else:

            if self.augmentation:
                image, encoded_mask = self._transform(image, encoded_mask)
            
            # split images and masks to smaller parts
            split_image, split_mask = self._split_image_mask(image, encoded_mask)

        return image_vis, split_image, split_mask, no_selected_classes_found