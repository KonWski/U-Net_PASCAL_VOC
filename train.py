from typing import List
from dataset import PascalVOCSegmentation
import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.optim import Adam
from model import uNetPascalVOC
import logging
from datetime import datetime
import sys

def train_model(
        device, 
        n_epochs: int,
        checkpoints_dir: str,
        download_datasets: bool,
        root_datasets_dir: str,
        year: str,
        selected_classes: List[str],
        splitted_mask_size: int,
        default_boundary: int
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

    n_classes = len(selected_classes)
    len_trainset = len(trainset)
    len_testset = len(testset)

    # model
    model = uNetPascalVOC(max_depth_level=4, n_classes=n_classes)
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=1e-5)

    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    model_size = round((param_size + buffer_size) / 1024**2, 2)
    print(f"model_size: {model_size}")

    for epoch in range(n_epochs):
        
        checkpoint = {}

        for state, loader, len_dataset in zip(["train", "test"], [train_loader, test_loader], [len_trainset, len_testset]):

            # calculated parameters
            running_loss = 0.0

            if n_classes == 2:
                criterion = BCEWithLogitsLoss()
            else:
                criterion = CrossEntropyLoss()

            if state == "train":
                model.train()
            else:
                model.eval()

            for id, batch in enumerate(loader, 0):

                with torch.set_grad_enabled(state == 'train'):
                    
                    split_images, split_masks = batch
                    # print(f"images.shape: {split_images.shape}")
                    # print(f"masks.shape: {split_images.shape}")

                    # split_images = split_images[0]
                    # split_masks = split_masks[0]
                    # TODO image.shape: torch.Size([3, 500, 334])
                    # print(f"image.shape: {image.shape}")
                    # TODO mask.shape: torch.Size([1, 500, 334, 2])
                    # print(f"mask.shape: {mask.shape}")

                    split_images = torch.stack(split_images).unsqueeze(dim=0)[0]
                    split_masks = torch.stack(split_masks).unsqueeze(dim=0)[0]

                    print(f"split_images.shape: {split_images.shape}")
                    print(f"split_masks.shape: {split_masks.shape}")

                    print(f"split_images.size: {(split_images.element_size() * split_images.nelement()) / 1000000000}")
                    print(f"split_masks.size: {(split_masks.element_size() * split_masks.nelement()) / 1000000000}")

                    # images = torch.unsqueeze(image, 0)
                    # masks = torch.unsqueeze(mask, 0)

                    # split_images = split_images.to(device)
                    # split_masks = split_masks.to(device)
                    optimizer.zero_grad()

                    # calculate loss
                    # TODO model output: (1, 1, 292, 292), (batch_n, n_classes, height, width)
                    outputs = model(split_images).to(device)
                    print(f"outputs.shape: {outputs.shape}")
                    loss = criterion(outputs, split_masks)

                    if state == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item()

            # save and log epoch statistics
            epoch_loss = round(running_loss / len_dataset, 2)

            # save stats for potential checkpoint
            checkpoint[f"{state}_loss"] = epoch_loss

            logging.info(f"Epoch: {epoch}, state: {state}, loss: {epoch_loss}")

        if checkpoint["test_loss"] < best_test_loss:
            
            # update lowest test loss
            best_test_loss = checkpoint["test_loss"]

            # save model to checkpoint
            checkpoint["epoch"] = epoch
            checkpoint["n_classes"] = n_classes
            checkpoint["selected_classes"] = ",".join(n_classes)
            checkpoint["model_state_dict"] = model.state_dict()
            checkpoint["save_dttm"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            checkpoint_path = f"{checkpoints_dir}/uNetPascalVOC"
            # save_checkpoint(checkpoint, checkpoint_path)

        else:
            logging.info(8*"-")
            
    return model