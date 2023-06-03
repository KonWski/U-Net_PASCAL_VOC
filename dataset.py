from torchvision.datasets import VOCSegmentation
from torchvision.transforms.functional import hflip, rotate, crop
from torchvision.transforms import ColorJitter
import cv2
import numpy as np
import torch
from typing import List
import math
import torch.nn.functional as F
import random
from torchvision import transforms
import random
from bs4 import BeautifulSoup

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

        self.limited_images, self.limited_masks = self._get_images_and_masks()

    def _get_images_and_masks(self):
        
        # 2011 dataset has different path to annotations
        if int(self.year) == 2011:
            dir_name = "TrainVal" if self.image_set == "train" else "Test"
            xml_dir_path = f"{self.root}{dir_name}/VOCdevkit/VOC{self.year}/Annotations"
        else:
            xml_dir_path = f"{self.root}/VOCdevkit/VOC{self.year}/Annotations"

        limited_images = []
        limited_masks = []
        
        for image_path, mask_path in zip(self.images, self.masks):

            image_id = image_path.split("/")[-1].split(".")[0]
            xml_annotation_path = f"{xml_dir_path}/{image_id}.xml"

            with open(xml_annotation_path, "r") as xml_file:
                xml_data = xml_file.read()

            # convert xml data to BeautifulSoup
            xml_data = BeautifulSoup(xml_data, "xml")

            # check if any selected classes occure in listed objects inside of xml file
            objects = xml_data.find_all("object")

            for obj in objects:
                for selected_class in self.selected_classes:
                    if selected_class in obj.get_text():

                        limited_images.append(image_path)
                        limited_masks.append(mask_path)
        
        return limited_images, limited_masks


    def _transform(self, image, mask):
        
        # shape of mask before flip should look like this: (class_dim, height, width)
        mask = mask.permute(2, 0, 1)

        # horizontal flip
        if random.random() > 0.5:
            image = hflip(image)
            mask = hflip(mask)

        # random crop
        if random.random() > 0.5:
            
            channels, image_height, image_width = image.shape

            min_height = 285
            min_width = 284

            if image_height > min_height and image_width > min_width:

                top = random.randint(0, image_height - min_height - 1)
                left = random.randint(0, image_width - min_width - 1)
                
                crop_height = random.randint(min_height, image_height - top + 1)
                crop_width = random.randint(min_width, image_width - left + 1)

                image = crop(image, top, left, crop_height, crop_width)
                mask = crop(mask, top, left, crop_height, crop_width)


        # rotation
        # if random.random() > 0.5:

        #     angle = random.randint(1, 30)
        #     image = rotate(image, angle, fill=0)
        #     mask = rotate(mask, angle, fill=0)

        #     # fill new mask values with background class
        #     mask = mask.permute(1,2,0)
        #     misssing_background_indices = np.all(mask.numpy() == [0, 0, 0], -1)
        #     misssing_background_indices = torch.Tensor(np.where(misssing_background_indices, 1, 0))
        #     mask[:, :, 0] = misssing_background_indices
        #     mask = mask.permute(2, 0, 1)

        # color jitter
        if random.random() > 0.5:
            color_jitter = ColorJitter(brightness=0.15, contrast=0.15)
            image = color_jitter(image)

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

        # cut out parts of original image and mask
        output_subimages = []
        output_submasks = []
        
        image_height = image.shape[1]
        image_width = image.shape[2]

        # n_rows x n_cols is the number of output smaller parts
        n_rows = math.ceil(image_height / self.splitted_mask_size)
        n_cols = math.ceil(image_width / self.splitted_mask_size)

        for n_row in range(n_rows):
            
            # upper and lower border of mask/image piece            
            row_split_mask_0 = n_row * self.splitted_mask_size
            row_split_img_0 = max(0, row_split_mask_0 - self.default_boundary)

            row_split_mask_1 = (n_row + 1) * self.splitted_mask_size
            row_split_img_1 = min(image_height, row_split_mask_1 + self.default_boundary)

            padding_img_top = abs(row_split_mask_0 - self.default_boundary) - row_split_img_0
            padding_img_bottom = (row_split_mask_1 + self.default_boundary) - row_split_img_1

            # corner case - lower border exceeds original image
            if row_split_mask_1 > image_height:
                padding_mask_bottom = row_split_mask_1 - image_height
            else:
                padding_mask_bottom = 0
            
            for n_column in range(n_cols):
                
                # left and right border of mask/image piece
                column_split_mask_0 = n_column * self.splitted_mask_size
                column_split_img_0 = max(0, column_split_mask_0 - self.default_boundary)

                column_split_mask_1 = (n_column + 1) * self.splitted_mask_size
                column_split_img_1 = min(image_width, column_split_mask_1 + self.default_boundary)
                
                padding_img_left = abs(column_split_mask_0 - self.default_boundary) - column_split_img_0
                padding_img_right = (column_split_mask_1 + self.default_boundary) - column_split_img_1

                # corner case - right border exceeds original image
                if column_split_mask_1 > image_width:
                    padding_mask_right = column_split_mask_1 - image_width
                else:
                    padding_mask_right = 0
                
                '''
                padding size for image and mask pieces
                padding size (left, right, top, bottom)
                '''
                padding_img = (padding_img_left, padding_img_right, padding_img_top, padding_img_bottom)
                padding_mask = (0, padding_mask_right, 0, padding_mask_bottom)

                # extract subimage and submask from input image and mask
                sub_image = image[:, row_split_img_0:row_split_img_1, column_split_img_0:column_split_img_1]
                sub_mask = mask[:, row_split_mask_0:row_split_mask_1, column_split_mask_0:column_split_mask_1]            
                
                # add padding for image and mask piece
                sub_image = F.pad(sub_image, padding_img)
                sub_mask = F.pad(sub_mask, padding_mask)

                # set part of mask that was extended by padding as background
                if padding_mask_right > 0:
                    sub_mask[0, :, -padding_mask_right:] = torch.ones([sub_mask.shape[1], padding_mask_right])
                
                if padding_mask_bottom > 0:
                    sub_mask[0, -padding_mask_bottom:, :] = torch.ones([padding_mask_bottom, sub_mask.shape[2]])
                
                # collect next pieces of image and mask part
                output_subimages.append(sub_image)
                output_submasks.append(sub_mask)
        
        return output_subimages, output_submasks


    def __getitem__(self, idx):
        
        image_vis = cv2.imread(self.limited_images[idx])
        image_vis = cv2.cvtColor(image_vis, cv2.COLOR_BGR2RGB)

        image = transforms.ToTensor()(image_vis)

        mask = cv2.imread(self.limited_masks[idx])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        # additional channel for background
        encoded_mask = torch.zeros([mask.shape[0], mask.shape[1], len(self.selected_classes)])

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

        # fill uncovered mask parts (when only part of classes were used) with background
        misssing_background_indices = np.all(encoded_mask.numpy() == [0 for selected_class in len(self.selected_classes)], -1)
        misssing_background_indices = torch.Tensor(np.where(misssing_background_indices, 1, 0))

        if torch.numel(misssing_background_indices) > 0:
            encoded_mask[:, :, 0] = encoded_mask[:, :, 0] + misssing_background_indices

        if self.augmentation:
            image, encoded_mask = self._transform(image, encoded_mask)
        
        # image normalization
        image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)

        # split images and masks to smaller parts
        split_image, split_mask = self._split_image_mask(image, encoded_mask)

        return image_vis, split_image, split_mask
    
    
    def __len__(self) -> int:
        return len(self.limited_images)