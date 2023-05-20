from typing import List
from dataset import PascalVOCSegmentation
import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from model import uNetPascalVOC, save_checkpoint, load_checkpoint
import logging
from datetime import datetime

def train_model(
        device, 
        n_epochs: int,
        checkpoints_dir: str,
        download_datasets: bool,
        root_datasets_dir: str,
        year: str,
        selected_classes: List[str],
        splitted_mask_size: int,
        default_boundary: int,
        initialize_model_weights: bool,
        load_model: bool
    ):
    '''
    n_epochs: int
        number of training epochs
    batch_size: int
        number of images inside single batch
    checkpoints_dir: str
        path to directory where checkpoints will be stored
    download_datasets: bool
        True -> download dataset from torchvision repo
    year: str
        year of Pascal VOC competition "2007" to "2012"
    root_datasets_dir: str
        path to directory where dataset should be downloaded (download_datasets = True)
        or where dataset is already stored
    selected_classes: List[str]
        names of selected classes
    splitted_mask_size: int
        width and height of smaller piece of mask
    default_boundary: int:
        padding size around cut out image piece
    '''

    BATCH_SIZE = 1

    # datasets and dataloaders
    trainset = PascalVOCSegmentation(f'{root_datasets_dir}/train/', year, "train", selected_classes, splitted_mask_size, 
                                     default_boundary, True, download_datasets)
    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

    testset = PascalVOCSegmentation(f'{root_datasets_dir}/test/', year, "test", selected_classes, splitted_mask_size,
                                     default_boundary, False, download_datasets)
    test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

    # update selected classes in case of 'all'
    selected_classes = trainset.selected_classes
    n_classes = len(selected_classes)

    model = uNetPascalVOC(4, n_classes, initialize_model_weights)
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=1e-5)

    if load_model:
        model, optimizer, loaded_checkpoint = load_checkpoint(model, optimizer, f"{checkpoints_dir}/uNetPascalVOC")
        best_test_loss = min(float("inf"), loaded_checkpoint["test_loss"])
        start_epoch = loaded_checkpoint["epoch"] + 1

    else:
        best_test_loss = float("inf")
        start_epoch = 0

    for epoch in range(start_epoch, n_epochs):
        
        checkpoint = {}

        for state, loader in zip(["train", "test"], [train_loader, test_loader]):

            # calculated parameters
            running_loss = 0.0
            n_imgs_loss = 0

            criterion = CrossEntropyLoss()

            if state == "train":
                model.train()
            else:
                model.eval()

            for id, batch in enumerate(loader, 0):

                with torch.set_grad_enabled(state == 'train'):
                    
                    image, split_image, split_mask, no_selected_classes_found = batch

                    # run only on images where at least one selected class was found
                    if no_selected_classes_found.item():
                        continue                        
                    
                    n_imgs_loss += 1

                    # first element - loader creates one element batch
                    split_image = [img_piece[0] for img_piece in split_image]
                    split_mask = [mask_piece[0] for mask_piece in split_mask]

                    split_image = torch.stack(split_image)
                    split_mask = torch.stack(split_mask)

                    optimizer.zero_grad()
                    
                    split_image = split_image.to(device)
                    outputs = model(split_image).to(device)
                    split_mask = split_mask.to(device)
                    loss = criterion(outputs, split_mask)

                    if state == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item()

            # save and log epoch statistics
            epoch_loss = round(running_loss / n_imgs_loss, 5)

            # save stats for potential checkpoint
            checkpoint[f"{state}_loss"] = epoch_loss

            logging.info(f"Epoch: {epoch}, state: {state}, loss: {epoch_loss}")

        if checkpoint["test_loss"] < best_test_loss:
            
            # update lowest test loss
            best_test_loss = checkpoint["test_loss"]

            # save model to checkpoint
            checkpoint["epoch"] = epoch
            checkpoint["selected_classes"] = selected_classes
            checkpoint["max_depth_level"] = 4
            checkpoint["model_state_dict"] = model.state_dict()
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
            checkpoint["save_dttm"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            checkpoint_path = f"{checkpoints_dir}/uNetPascalVOC"
            save_checkpoint(checkpoint, checkpoint_path)

        else:
            logging.info(8*"-")
            
    return model