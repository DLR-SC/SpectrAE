#
# Author(s): Peer Schütt
#
# See the file "LICENSE" for the full license and copyright governing this code.
#

import os, glob
import numpy as np
from pathlib import Path
import logging
import torch
import matplotlib.pyplot as plt
from sklearn import manifold
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import torch.nn as nn

import xml.etree.ElementTree as ET

import wandb
import random


logger = logging.getLogger(__name__)
DIR_PATH = os.path.dirname(os.path.realpath(__file__))

# Data paths
data_storage_path = "sample_images"

sample_data_paths = {
    "sample": f"{data_storage_path}/", 
}

def load_crosstalk_matrix(img_type: str) -> np.ndarray:
    """Load and parse the crosstalk correction matrix from XML file."""
    if img_type not in ["VIS", "NIR"]:
        raise ValueError(f"Image type must be 'VIS' or 'NIR', got '{img_type}'")

    filename = 'CMS-S.xml' if img_type == "NIR" else 'CMS-C.xml'
    tree = ET.parse(os.path.join(DIR_PATH, filename))
    root = tree.getroot()
    crosstalk_coefficients = root.find('crosstalkCorrectionCoefficients').text.split("*")

    # Convert to numpy array and reshape
    crosstalk_coefficients = np.array(
        crosstalk_coefficients,
        dtype=np.float32
    ).reshape(9, 9)

    return crosstalk_coefficients
    

def find_image(img_search_name):
    """Find the full path of the image file with the matching name.

    :param string img_search_name: Name of the image you want to get the full path of
    :return string: full path of the image file
    """
    img_type = "VIS" if "VIS" in img_search_name else "NIR"

    img_paths = load_all_img_paths([*sample_data_paths.values()], img_type=img_type)

    return next((path for path in img_paths if img_search_name in os.path.basename(path)), None)


def load_all_img_paths(dir_list, img_type, split="all"):
    """Load all the .tif and .npy images that are in the folders for the specified img_type (either VIS or NIR). For certain folder we need to exclude some images that are faulty or overexposed.

    :param list dir_list: List of all directories that contain images
    """    
    assert isinstance(dir_list, list), "dir_list has to be a LIST!"
    assert split in ["train", "test", "all"], "options for split are train/test/all"
    all_img_list = []
    
    for directory in dir_list:
        
        img_list = glob.glob(f"{directory}"+"/*"+f"{img_type}"+"*.npy")
        
        all_img_list.extend(img_list)
        
    return all_img_list
        

def loss_func(x,y, per_img_and_pixel = False):
    """Calculate the loss.

    :param Tensor x: Input
    :param Tensor y: Output
    :param bool per_image_and_pixel: If set to True the resulting loss has the same shape as x and y, defaults to False
    :return tensor: Loss
    """
    if per_img_and_pixel is False:
        loss = nn.MSELoss()
        # loss = nn.L1Loss()
    else:
        loss = nn.MSELoss(reduction="none")
        # loss = nn.L1Loss(reduction="none")

    return loss(x,y)


def save_img_comparisons(inputs, outputs, img_full_path, channels_to_vis=[0,1,2,3,4,5,6,7,8], prefix="", max_batch_save_number=10, save_path_override=None, dpi="figure", fontsize_title=20):
    """For every image in the batch, save a grid of images, where original image, reconstructed image and loss are compared side-by-side. Use channels_to_vis to choose which channels you want to plot.

    :param tensor inputs: Input to the network, shape: batch_size x num_channels x H x W
    :param tensor outputs: Output of the network, shape: batch_size x num_channels x H x W
    :param list img_full_path: Paths of every original image
    :param tensor loss: Loss per pixel, shape: batch_size x num_channels x H x W
    :param str prefix: String prefix for the saved grid
    :param list channels_to_vis: Image channels that should be visualized
    :param int max_save_number: Maximum number of images in the batch that should be saved
    """
    
    loss = loss_func(outputs, inputs, per_img_and_pixel=True).clone().detach()
    
    for batch_idx in range(inputs.shape[0]):
        # only save a certain number of images by default
        if batch_idx > max_batch_save_number:
            break
        
        if save_path_override is not None:
            save_path = save_path_override
        else:
            save_path = get_save_path(img_full_path[batch_idx], prefix=prefix)
        
        # channel 8 is the greyscale channel
        input_img = inputs[batch_idx][:,:,:].detach().cpu()
        output_img = outputs[batch_idx][:,:,:].detach().cpu()
        loss_img = loss[batch_idx,:,:].detach().cpu()
        
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
        fig, axarr = plt.subplots(3,len(channels_to_vis), figsize=(5*len(channels_to_vis), 12), constrained_layout=True)
        fig.suptitle(f'Visualization for {os.path.basename(save_path)}', fontsize=16)
        
        for i in channels_to_vis:
            
            axarr[0,i].imshow(input_img[i,:,:], cmap="viridis", vmin=0., vmax=1.)
            axarr[0,i].set_axis_off()
            axarr[0,i].set_title(f"Input Channel {i}", fontsize=fontsize_title)
            
            axarr[1,i].imshow(output_img[i,:,:], cmap="viridis", vmin=0., vmax=1.)
            axarr[1,i].set_axis_off()
            axarr[1,i].set_title(f"Reconstruction Channel {i}", fontsize=fontsize_title)
            
            diff_img = loss_img[i,:,:]
            # diff_img = 1. - diff_img
            diff_img_max = diff_img.flatten().max(dim=0)[0]
            diff_img_min = diff_img.flatten().min(dim=0)[0]
            # diff_img_normalized = (diff_img - diff_img_min) / (diff_img_max - diff_img_min)
            # cax = axarr[2,i].imshow(diff_img_normalized, cmap="viridis", vmin=0., vmax=1.)
            cax = axarr[2,i].imshow(diff_img, cmap="viridis_r")
            axarr[2,i].set_axis_off()
            axarr[2,i].set_title(f"Loss Channel {i}", fontsize=fontsize_title)
        
            cbar = fig.colorbar(cax, ax=axarr[2,i], shrink=0.75)
            cbar.ax.tick_params(labelsize=15)
        
        plt.savefig(save_path, pad_inches=0., dpi=dpi)
        
        plt.close()

def plot_image_wandb(outputs, inputs, split, step, img_full_path):
    """Plot a grid of the input, output and loss image next to each other and log it on Weights & Biases. 
    Only use the first image in the batch.
    Only use the last channel, which is usually the panchromatic channel.

    :param torch.tensor outputs: Decoded image
    :param torch.tensor inputs: Input image
    :param string split: train/val/test
    :param int step: Current epoch for logging
    :param string img_full_path: List with all the image names in the batch
    """
    
    loss = loss_func(outputs, inputs, per_img_and_pixel=True).clone().detach()
        
    # grid_img = make_grid([inputs[0,-1,:,:], outputs[0,-1,:,:], loss_image_inverted], padding=50, nrow=3, scale_each=True, normalize=False).unsqueeze(1)
    # grid_img_wandb = wandb.Image(grid_img, caption="Left: Input, Middle: Output, Right: Inverted Loss")
    
    # wandb.log({f"{split}/example_reconstruction": grid_img_wandb}, step=(step if step>=0 else None))
    
    # save the per channel histogram
    # for i in range(images.shape[1]):
    #     wandb.log({f"{split}/channel_{i}": wandb.Histogram(loss[0,i,::].cpu())}, step=(step if step>=0 else None))
    
    # only get the first image in the batch
    input_img = inputs[0][:,:,:].detach().cpu()
    output_img = outputs[0][:,:,:].detach().cpu()
    loss_img = loss[0,:,:].detach().cpu()
    
    fig, axarr = plt.subplots(1,3, figsize=(12, 4), constrained_layout=True)
    fig.suptitle(f'Reconstruction of last channel for {os.path.basename(img_full_path[0])}', fontsize=16)
    
    # only get the last channel (panchromatic filter) for this visualization
    i = -1
        
    axarr[0].imshow(input_img[i,:,:], cmap="viridis", vmin=0., vmax=1.)
    axarr[0].set_axis_off()
    axarr[0].set_title(f"Input image channel {i}")
    
    axarr[1].imshow(output_img[i,:,:], cmap="viridis", vmin=0., vmax=1.)
    axarr[1].set_axis_off()
    axarr[1].set_title(f"Output image channel {i}")
    
    diff_img = loss_img[i,:,:]
    # diff_img = 1. - diff_img
    diff_img_max = diff_img.flatten().max(dim=0)[0]
    diff_img_min = diff_img.flatten().min(dim=0)[0]
    diff_img_normalized = (diff_img - diff_img_min) / (diff_img_max - diff_img_min)
     # reverse it for better visualization
    # cax=axarr[2].imshow(diff_img_normalized, cmap="viridis", vmin=0., vmax=1.)
    cax=axarr[2].imshow(diff_img, cmap="viridis_r")
    axarr[2].set_axis_off()
    axarr[2].set_title(f"Loss channel {i}")
    
    cbar = fig.colorbar(cax, ax=axarr[2], shrink=0.75)
    
    wandb.log({f"{split}/reconstruction": fig}, step=(step if step>=0 else None))
    
    plt.close()
    

def visualize_embedding_space(autoencoder, train_data_loader, val_data_loader, test_data_loader, device, plot_dims=2, save_path="tsne_embeddings.png"):
    """Visualize latent space embeddings using t-SNE dimensionality reduction.

    Args:
        autoencoder (torch.nn.Module): Trained autoencoder model
        train_data_loader (DataLoader): Training data loader
        val_data_loader (DataLoader): Validation data loader
        test_data_loader (DataLoader): Test data loader
        device (str): Device to run inference on ('cpu' or 'cuda')
        plot_dims (int, optional): Number of t-SNE dimensions (2 or 3). Defaults to 2.
        save_path (str, optional): Path to save the plot. Defaults to "tsne_embeddings.png"
    """
    # Input validation
    if plot_dims not in [2, 3]:
        raise ValueError("plot_dims must be 2 or 3")
    data_loaders = [train_data_loader, val_data_loader, test_data_loader]
    colors = ListedColormap(["tab:cyan", "tab:olive", "tab:red"])
    classes = ["Train", "Validation", "Test"]

    # Collect embeddings
    embeddings = []
    labels = []
    autoencoder.eval()
    with torch.no_grad():
        for loader_idx, loader in enumerate(data_loaders):
            for images, _, _ in loader:
                batch_embeddings = autoencoder.encode(images.to(device)).cpu()
                embeddings.extend(batch_embeddings.squeeze().numpy())
                labels.extend([loader_idx] * len(images))

    # Convert to numpy arrays
    embeddings = np.array(embeddings)
    labels = np.array(labels)

    # Fit t-SNE
    tsne = manifold.TSNE(
        n_components=plot_dims,
        perplexity=30,
        init="random",
        n_iter=250,
        random_state=42
    )
    embeddings_tsne = tsne.fit_transform(embeddings)

    # Create plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d' if plot_dims == 3 else None)
    if plot_dims == 3:
        scatter = ax.scatter(
            embeddings_tsne[:,0],
            embeddings_tsne[:,1],
            embeddings_tsne[:,2],
            c=labels,
            cmap=colors,
            s=20,
            alpha=0.6
        )
    else:
        scatter = ax.scatter(
            embeddings_tsne[:,0],
            embeddings_tsne[:,1],
            c=labels,
            cmap=colors,
            s=20,
            alpha=0.6
        )

    ax.set_title("t-SNE Visualization of Latent Space", pad=15)
    ax.legend(handles=scatter.legend_elements()[0], labels=classes)
    ax.grid(True, linestyle='--', alpha=0.7)

    # Save plot
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    
def get_save_path(img_full_path, save_decoded_img_path = f"{DIR_PATH}/reconstructed_images/", prefix="DECODE_", file_extension=".png"):
    """Get the path where the reconstructed image should be saved to and create all folders on the way.

    :param str img_full_path: Path of the input image, which was en- and decoded 
    :param str save_decoded_img_path: Parent folder where decoded images should be saved
    :return str: path for the image
    """
        
    parts = img_full_path.split("/")
    folder = parts[-2]
    img_filename = parts[-1]
    
    save_path = save_decoded_img_path + folder + "/"
    path = Path(save_path)
    path.mkdir(parents=True, exist_ok=True)
    save_path += prefix + img_filename[:-4]+file_extension
    logging.debug(f"Saving the decoded image at {save_path}")
    
    return save_path


def set_global_random_seed(config, with_wandb=False):
    """Give all possible sources of randomness the same seed to allow reproducibility. 
    If config["seed"]==-1, then the system generates a random seed between 0 and 100000,
    saves that in the config file and uses it as a global random seed. 

    :param _dict_ config: Config file from which we want the random seed that is saved under the key "seed".
    """
    if 'random_seed' in dict.keys(config):
        random_seed = config["random_seed"]
    else:
        random_seed = random.randint(0,100000)
    
    if with_wandb is True:
        wandb.config.update({"random_seed": random_seed}, allow_val_change=True)
    config["random_seed"] = random_seed
            
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)


def labelme_plot_with_custom_cmap(image, class_names, road_label):
    
    fixed_cmap_values = [
            [1.00, 1.00, 1.00],  # 0: White
            [0.90, 0.17, 0.17],  # 1: Bright Red
            [0.12, 0.75, 0.12],  # 2: Bright Green
            [0.12, 0.35, 0.95],  # 3: Royal Blue
            [1.00, 0.60, 0.00],  # 4: Orange
            [0.75, 0.13, 0.83],  # 5: Magenta
            [0.00, 0.85, 0.90],  # 6: Cyan
            [0.95, 0.90, 0.15],  # 7: Yellow
            [0.55, 0.27, 0.07],  # 8: Brown
            [0.00, 0.50, 0.50],  # 9: Teal
            [1.00, 0.40, 0.70],  # 10: Hot Pink
            [0.50, 0.50, 0.50],  # 11: Gray
            [0.70, 0.87, 0.30],  # 12: Lime
            [1., 0.1, 0.1],  # 13: Burnt Orange
            [0.37, 0.15, 0.55],  # 14: Deep Purple
            [0.88, 0.63, 0.86],  # 15: Lavender
            [0.60, 0.98, 0.60],  # 16: Mint Green
            [1., 1., 1.],  # 16: Mint Green
        ]
    cmap = ListedColormap(fixed_cmap_values)
    
    # get rid of the road
    image[image == road_label] = 0
    
    # Get unique values in the image
    unique_values = np.unique(image)
    
    # Plot the image
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(image, cmap=cmap, vmin=0, vmax=len(cmap.colors)-1, interpolation='nearest')
    
    # Create legend patches
    patches = []
    for val in unique_values:
        if val in class_names:
            color = cmap.colors[val]
            label = f"{val}: {class_names[val]}"
            patches.append(mpatches.Patch(color=color, label=label))
    
    # Add legend
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.xticks([])   
    # disabling yticks by setting yticks to an empty list
    plt.yticks([])  
    
    return fig, ax