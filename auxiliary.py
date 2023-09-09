from dataset import PascalVOCSegmentation
from torch.utils.data import DataLoader
from typing import List
import torch

def get_balanced_class_weights(datasets: List[PascalVOCSegmentation]):
    '''
    Calculates balanced loss weights for each class according to following equation:
    class_weight = all_observations / (n_classes * class_weight_observations)

    datasets: List[PascalVOCSegmentation]
        datasets from which all_observations and class_weight_observations will be calculated
    '''
    
    n_classes = len(datasets[0].selected_classes)
    class_occurences = {n_class: 0.0 for n_class in range(n_classes)}
    class_weights = {n_class: 0.0 for n_class in range(n_classes)}

    dataloaders = [DataLoader(dataset, batch_size=1, shuffle=False) for dataset in datasets]

    # count occurences across all datasets
    for dataloader in dataloaders:

        for id, batch in enumerate(dataloader, 0):
            
            image, split_image, split_mask = batch

            # perform calculations for each of mask piece
            for mask_piece in split_mask:

                # calculate number of pixels with given class
                for n_class in range(n_classes):
                    class_occurences[n_class] = class_occurences[n_class] + mask_piece[0][n_class, :, :].sum().item()

    # calculate weights
    all_observations = sum([class_occurences[n_class] for n_class in range(n_classes)])

    for n_class in class_weights.keys():
        class_weights[n_class] = all_observations / (n_classes * class_occurences[n_class])

    # loss class require weights to be a tensor
    class_weights = torch.tensor([class_weights[n_class] for n_class in range(n_classes)])

    return class_weights