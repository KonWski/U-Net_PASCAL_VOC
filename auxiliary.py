from dataset import PascalVOCSegmentation
from torch.utils.data import DataLoader

BATCH_SIZE = 1

def get_class_weights(dataset: PascalVOCSegmentation):
    '''
    Calculates loss weights for each class
    '''
    
    n_selected_classes = len(dataset.selected_classes)
    class_occurences = {n_class: 0.0 for n_class in range(n_selected_classes)}
    class_weights = {n_class: 0.0 for n_class in range(n_selected_classes)}

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    for id, batch in enumerate(dataloader, 0):
        
        image, split_image, split_mask = batch

        # perform calculations for each of mask piece
        for mask_piece in split_mask:
            print(f"mask_piece.shape: {mask_piece.shape}")
            # calculate number of pixels with given class
            for n_class in range(n_selected_classes):
                class_occurences[n_class] = class_occurences[n_class] + mask_piece[0][n_class, :, :].sum().item()

    # find class with most occurences
    most_common_class = 0
    for n_class, n_occurences in class_occurences.items():

        if n_occurences > class_occurences[most_common_class]:
            most_common_class = n_class

    # calculate weights
    for n_class in class_weights.keys():
        class_weights[n_class] = class_occurences[most_common_class] / class_occurences[n_class]

    return class_weights