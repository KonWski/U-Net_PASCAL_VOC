from torchvision.datasets import VOCSegmentation
from torchvision import transforms
import cv2
import numpy as np
import torch
from typing import List

class PascalVOCSegmentation(VOCSegmentation):

    def __init__(
            self, 
            root: str, 
            year: str, 
            image_set: str, 
            selected_classes: List[str], 
            download: bool = False, 
            transform: transforms = None
            ):
        '''
        Dataset limiting original VOCSegmentation to specified class_name

        root: str
            path to PascalVOC dataset content
        year: str
            year of competition "2007" to "2012"
        image_set: str
            "train", "trainval", "test"
        selected_classes: List[str]:
            names of selected classes
        download: bool
            True -> downloads dataset from repo
            False -> uses already existing dataset
        transform:
            transforms performed on image and mask
        '''
        super().__init__(root, year, image_set, download, transform)
        self.selected_classes = selected_classes
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


    def __getitem__(self, idx):
        
        image = cv2.imread(self.images[idx])
        mask = cv2.imread(self.masks[idx])

        if self.transform:
            transforms = self.transform(image = image, mask = mask)
            image = transforms(image)
            mask = transforms(mask)

        # additional channel for background
        encoded_mask = torch.zeros([mask.shape[0], mask.shape[1], len(self.selected_classes) + 1])

        for selected_class in self.selected_classes:
            
            channel_id = self.class_to_color[selected_class][0]
            class_color_encoding = self.class_to_color[selected_class][1]
            class_pixels_indices = np.where(mask == class_color_encoding)
            encoded_mask[class_pixels_indices[0], class_pixels_indices[1], channel_id] = 1

        return image, encoded_mask