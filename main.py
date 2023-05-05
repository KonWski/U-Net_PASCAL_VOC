import argparse
import logging
import torch
from train import train_model

def get_args():

    parser = argparse.ArgumentParser(description='Paramaters for model training')
    parser.add_argument('--n_epochs', type=int, help='Number of epochs')
    parser.add_argument('--checkpoints_dir', type=str, help='Path to directory where checkpoint will be saved')
    parser.add_argument('--download_datasets', type=str, help='Download dataset from Torchvision repo or use already existing dataset')
    parser.add_argument('--root_datasets_dir', type=str, help='Path where dataset should be downloaded or where is it already stored')
    parser.add_argument('--year', type=str, help='year of Pascal VOC competition "2007" to "2012"')
    parser.add_argument('--selected_classes', type=str, help='classes seperated by commas')

    parser.add_argument('--splitted_mask_size', type=int, help='width and height of smaller piece of mask')
    parser.add_argument('--default_boundary_size', type=int, help='padding size around cut out image piece')

    args = vars(parser.parse_args())
    
    # parse str to boolean
    str_true = ["Y", "y", "Yes", "yes", "true", "True"]
    bool_params = ["download_datasets"]
    for param in bool_params:
        if args[param] in str_true:
            args[param] = True
        else:
            args[param] = False

    # parse str to list (selected_classes)
    args["selected_classes"] = args["selected_classes"].split(",")

    # extend selected_classes with background
    if "background" not in args["selected_classes"] and args["selected_classes"][0] != "all":
        args["selected_classes"].append("background")

    # log input parameters
    logging.info(8*"-")
    logging.info("PARAMETERS")
    logging.info(8*"-")

    for parameter in args.keys():
        logging.info(f"{parameter}: {args[parameter]}")
    logging.info(8*"-")

    return args

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    args = get_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Device: {device}")

    model = train_model(device, args["n_epochs"], args["checkpoints_dir"], args["download_datasets"], 
                        args["root_datasets_dir"], args["year"], args["selected_classes"],
                        args["splitted_mask_size"], args["default_boundary_size"])