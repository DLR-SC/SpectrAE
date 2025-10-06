#
# Author(s): Peer Schütt
#
# See the file "LICENSE" for the full license and copyright governing this code.
#

import os
import argparse
import logging
import json
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from autoencoder import Autoencoder
from MultispectralImageDataset import MultispectralImageDataset
from utils import save_img_comparisons, loss_func, plot_image_wandb, load_all_img_paths, set_global_random_seed
from utils import sample_data_paths # import all image paths

logger = logging.getLogger(__name__)
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
USE_WANDB = False

def train(model, train_data_loader, device, num_epochs, config, save_model_folder=f"{DIR_PATH}/trained_models/", val_data_loader=None, save_img_prefix=""):
    
    min_loss = float('inf')
    
    history = {
        "config": config,
        "accum_loss": [],
        "mean_loss_per_batch": [],
        "mean_loss_per_image": []
    }

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=1e-5
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10
    )
    
    for epoch in tqdm(range(num_epochs), desc="Training"):
        epoch_loss = 0.0
        save_first_batch = True

        model.train()
        for batch_idx, (corrected_images, raw_images, img_full_path) in enumerate(tqdm(train_data_loader, disable=True)):
            batch_size = raw_images.shape[0]
            raw_images = raw_images.to(device)

            outputs = model(raw_images)
            loss = loss_func(outputs, raw_images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if save_first_batch:
                save_img_comparisons(raw_images, outputs, img_full_path, config["channels_to_use"], save_img_prefix)
                if USE_WANDB:
                    plot_image_wandb(outputs, raw_images, "train", epoch, img_full_path)
                save_first_batch = False
            

        mean_loss_per_batch = epoch_loss / len(train_data_loader)
        mean_loss_per_image = mean_loss_per_batch / batch_size

        logging.debug(f"Epoch {epoch}: Loss = {mean_loss_per_image:.4f} per image")

        # Update history
        history["accum_loss"].append(epoch_loss)
        history["mean_loss_per_batch"].append(mean_loss_per_batch)
        history["mean_loss_per_image"].append(mean_loss_per_image)

        # Save training history
        with open(f"{save_model_folder}{model.model_name}.json", "w", encoding="utf-8") as f:
            json.dump(history, f)

        # Validation phase
        if val_data_loader is not None:
            val_loss = test(model, val_data_loader, device, config["channels_to_use"],
                          save_img_prefix=save_img_prefix, split="val", step=epoch)

            scheduler.step(val_loss)

            # Save best model
            if val_loss < min_loss:
                min_loss = val_loss
                torch.save(model.state_dict(), f"{save_model_folder}{model.model_name}.pth")

        if USE_WANDB:
            wandb.log({"train/mean_loss_per_image": mean_loss_per_image}, step=epoch)
            wandb.log({"train/learning_rate": scheduler.get_last_lr()}, step=epoch)

    return model


def test(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    channels_to_use: list[int],
    save_img_prefix: str = "",
    every_other_batch: int = 10,
    split: str = "test",
    step: int = -1
) -> torch.Tensor:
    """Test the model by encoding and decoding images from the dataloader.

    Args:
        model: PyTorch model to test
        data_loader: DataLoader containing test images
        device: Device to run inference on
        channels_to_use: List of channel indices to visualize
        save_img_prefix: Prefix for saved image files
        every_other_batch: Only save reconstructions every N batches
        split: Dataset split being tested ("train", "val", or "test")
        step: Current training step for logging

    Returns:
        torch.Tensor: Accumulated reconstruction loss
    """
    if not isinstance(every_other_batch, int) or every_other_batch < 1:
        raise ValueError("every_other_batch must be a positive integer")
    model.eval()
    accum_loss = 0.0
    save_first_batch = True
    batch_size = None

    with torch.no_grad():
        for batch_idx, (corrected_images, raw_images, img_full_path) in enumerate(tqdm(data_loader, desc=f"Testing {split}", disable=(split=="val"))):
            raw_images = raw_images.to(device)
            batch_size = raw_images.shape[0] if batch_size is None else batch_size

            # Forward pass
            outputs = model(raw_images)
            loss = loss_func(outputs, raw_images, per_img_and_pixel=True)
            accum_loss += torch.mean(loss)

            # Save visualizations
            if save_first_batch and USE_WANDB:
                plot_image_wandb(outputs, raw_images, split, step, img_full_path)
                save_first_batch = False

            if batch_idx % every_other_batch == 0:
                save_img_comparisons(
                    raw_images, outputs, img_full_path, channels_to_use, save_img_prefix
                )

        # Log metrics
        if USE_WANDB:
            mean_loss = (accum_loss / len(data_loader)) / batch_size
            wandb.log(
                {f"{split}/mean_loss_per_image": mean_loss},
                step=(step if step >= 0 else None)
            )

    return accum_loss


def main(num_epochs_train, use_wandb, img_type, device):
    global USE_WANDB, OVERFIT
    USE_WANDB = use_wandb
    OVERFIT = False

    dense_layer_dim = 2048  
    channels_to_use = [0,1,2,3,4,5,6,7,8]
    
    # Configuration parameters
    config = {
        'dense_layer_dim': dense_layer_dim,
        'autoencoder_depth': 3,
        'batch_size': 128,
        'learning_rate': 1e-4,
        'channels_to_use': channels_to_use,
        'val_split': 0.1,
        'big_img_size_x': 962,
        'big_img_size_y': 1250,
        'save_prefix': f"{dense_layer_dim}_{len(channels_to_use)}_",
        "img_type": img_type,
        "cs":16,
    }
    
    config["img_size_x"] = int(config['big_img_size_x']/3)
    config["img_size_y"] = int(config['big_img_size_y']/3)
    
    # Transform definition
    transform = transforms.Compose([
        transforms.CenterCrop([config['img_size_x'], config['img_size_y']])
    ])

    ##################################
    ######## Model definition ########
    ##################################

    model = Autoencoder(**config).to(device)
    config["model_name"] = model.model_name
    
    # WandB configuration
    if USE_WANDB:
        wandb_config = {
            **config,  # Include all configuration parameters
            'epochs': num_epochs_train,
        }
        run = wandb.init(
            project="spectrae",
            config=wandb_config,
            save_code=False
        )
        wandb.watch(model, log_freq=10)
        logging.info("Using WANDB backup to log model progress")

    set_global_random_seed(config)

    ##################################
    ####### Dataset Creation #########
    ##################################

    # Create dataset with common parameters
    dataset_params = {
        'transform': transform,
        'img_type': img_type,
        'channels_to_use': config['channels_to_use'],
        'device': device,
        'overfit': OVERFIT,
        'augment_data': False
    }    
    
    ##################################
    ########## Train Setup ###########
    ##################################
    
    # Combine training folders
    train_folder = [
        *sample_data_paths.values(),
    ]

    # Load and prepare datasets
    train_img_paths = load_all_img_paths(train_folder, img_type, split="train")
    
    # Initialize datasets
    train_val_dataset = MultispectralImageDataset(train_img_paths, **dataset_params)

    # Calculate split sizes
    dataset_size = len(train_val_dataset)
    val_size = int(np.ceil(dataset_size * config['val_split']))
    train_size = dataset_size - val_size

    # Split dataset
    train_set, val_set = torch.utils.data.random_split(
        train_val_dataset,
        [train_size, val_size]
    )

    # Create data loaders with common parameters
    dataloader_params = {
        'batch_size': config['batch_size'],
        'num_workers': 4,
        'pin_memory': True
    }

    train_data_loader = DataLoader(train_set, shuffle=True, **dataloader_params)
    val_data_loader = DataLoader(val_set, shuffle=False, **dataloader_params)

    logging.info(f"Dataset splits - Train: {len(train_data_loader.dataset)}, "
                f"Validation: {len(val_data_loader.dataset)}")
    
    ##################################
    ########## Test Setup ###########
    ##################################

    # Prepare test dataset
    test_folder = [
        *sample_data_paths.values(), 
    ]
    test_img_paths = load_all_img_paths(test_folder, img_type, split="test")
    test_dataset = MultispectralImageDataset(test_img_paths, **dataset_params)
    test_data_loader = DataLoader(test_dataset, shuffle=False, **dataloader_params)

    logging.info(f"Test set size: {len(test_data_loader.dataset)}")
        
    ##################################
    ######### Model Pipeline ########
    ##################################

    # Training phase
    trained_model = train(
        model=model,
        train_data_loader=train_data_loader,
        device=device,
        num_epochs=num_epochs_train,
        val_data_loader=val_data_loader,
        save_img_prefix=config['save_prefix'],
        config=config
    )

    # Load pretrained model if needed
    # model_path = f"{DIR_PATH}/trained_models/2025-04-18_16-14-34_musero_autoencoder_NIR.pth"
    # if os.path.exists(model_path):
    #     trained_model.load_state_dict(torch.load(model_path))
    #     logging.info(f"Loaded pretrained model from {model_path}")

    # Testing phase
    test(
        model=trained_model,
        data_loader=test_data_loader,
        device=device,
        channels_to_use=config['channels_to_use'],
        save_img_prefix=config['save_prefix']
    )

    ##################################
    ####### Visualization ###########
    ##################################

    # # Visualize embeddings
    # for dims in [2, 3]:
    #     visualize_embedding_space(
    #         model=trained_model,
    #         train_data_loader=train_data_loader,
    #         val_data_loader=val_data_loader,
    #         test_data_loader=test_data_loader,
    #         device=device,
    #         plot_dims=dims
    #     )

    if USE_WANDB:
        wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train an autoencoder model for image processing')
    parser.add_argument(
        '-log_lvl',
        choices=['debug', 'info', 'warning', 'error', 'critical'],
        help='Logging level',
        type=str.lower,
        default="info"
    )
    parser.add_argument(
        '-epochs',
        help='Number of epochs to train the autoencoder',
        type=int,
        default=1,
        metavar='N'
    )
    parser.add_argument(
        '-gpu_id',
        help='GPU ID for calculations (-1 for CPU)',
        type=int,
        default=-1,
        metavar='ID'
    )
    parser.add_argument(
        '-wandb',
        help='Enable Weights & Biases logging',
        action="store_true"
    )
    parser.add_argument(
        '-img_type',
        choices=['VIS', 'NIR'],
        help='Image type to process',
        type=str.upper,
        default="NIR"
    )
    args = parser.parse_args()

    # Configure logging
    log_format = "%(levelname)s::%(message)s"
    logging.basicConfig(
        level=args.log_lvl.upper(),
        format=log_format,
        handlers=[logging.StreamHandler()]
    )
    logging.getLogger('matplotlib.font_manager').disabled = True

    # Configure device
    if torch.cuda.is_available() and args.gpu_id >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        device = torch.device(f"cuda:{args.gpu_id}")
        logging.info(f"Using GPU {args.gpu_id}")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU")
        
    # Run main function
    main(args.epochs, args.wandb, args.img_type, device)
