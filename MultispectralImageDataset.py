# pylint: disable=W1201, W1203, W0612, C0301
#
# Author(s): Peer Schütt
#
# See the file "LICENSE" for the full license and copyright governing this code.
#

import os
import logging
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
from utils import load_crosstalk_matrix


DIR_PATH = os.path.dirname(os.path.realpath(__file__))
# the first two rows of the camera filter don't have any filters and are therefore always unusable
MULTISPECTRAL_CAMERA_OFFSET_HEIGHT = 0
MULTISPECTRAL_CAMERA_OFFSET_WIDTH = 2

# the positions of each pixel of the macropixel are given in the documentation from Silios
# explanation to each filter position:
# 0: lowest wavelength band-pass filter
# 7: highest wavelength band-pass filter
# 8: panchromatic filter (grayscale image)
MULTISPECTRAL_CAMERA_FILTER_POS = [4,3,5,8,2,6,0,1,7]

class MultispectralImageDataset(Dataset):
    """Loading multispectral images. The images are either VIS (visual spectrum - 400-700nm) or NIR (near-infrared spectrum - 650-950nm) images."""

    def __init__(self, img_path_list, transform=None, img_type="VIS", channels_to_use=[0,1,2,3,4,5,6,7,8], device="cpu", overfit=False, augment_data=False):
        self.transform = transform
        
        # Use tuple instead of list for immutable default argument
        self.channels_to_use = tuple(channels_to_use)
        logging.info(f"Using {img_type} images!")

        self.img_path_list = img_path_list

        logging.debug("Loading these image files: " + str(self.img_path_list))

        # load the crosstalk matrix for the image type
        # NIR and VIS have different matrices
        self.cross = load_crosstalk_matrix(img_type)
        
        self.offset_x = MULTISPECTRAL_CAMERA_OFFSET_WIDTH
        self.offset_y = MULTISPECTRAL_CAMERA_OFFSET_HEIGHT

        self.device = device

        # Initialize transform once
        self.horizontal_flip_transform = transforms.RandomHorizontalFlip(p=1.0)
        # simple augmenation: Add the horizontally flipped image to the dataset - results in a dataset of double the size
        self.augment_data = augment_data
        if overfit is False:
            self.len = len(self.img_path_list)
        else:
            self.len = 4
            logging.info("Overfitting the model with " + str(self.len) + " images!")

    def __len__(self):
        if self.augment_data is True:
            return self.len * 2
        return self.len

    def __getitem__(self, idx):
        if self.augment_data is True:
            img_idx = int(idx / 2)
        else:
            img_idx = idx

        img_name = self.img_path_list[img_idx]

        # values of image are between 0 and 255
        # image has shape 1024x1280
        if img_name.endswith('.tif'):
            image = np.array(Image.open(img_name))
        else:
            try:
                image = np.load(img_name)
            except:
                print("Can't load ", img_name)
                return torch.zeros((9, 320, 416)), torch.zeros((9, 320, 416)), ""

        # get the cross correlation corrected image and the not corrected image
        # image after this is between 0 and 1.
        corrected_image, raw_image = convert_to_multispec_img(image, self.cross, self.offset_x, self.offset_y, self.transform)

        # Convert to tensor in one step
        corrected_image = torch.tensor(corrected_image, dtype=torch.float32)
        raw_image = torch.tensor(raw_image, dtype=torch.float32)

        # augment the image by performing the augmentation transformation
        if (self.augment_data is True) and ((idx%2) == 1):
            corrected_image = self.horizontal_flip_transform(corrected_image)
            raw_image = self.horizontal_flip_transform(raw_image)

        # get the right channels
        corrected_image = corrected_image[self.channels_to_use,:,:]
        raw_image = raw_image[self.channels_to_use,:,:]

        return corrected_image, raw_image, img_name


def single_channel_to_multispec_image(input_img, offset_x=2, offset_y=0, transform=None):
    """Convert a single channel image to multispectral format.

    Args:
        input_img (np.ndarray): Input image array
        offset_x (int, optional): X offset for sampling. Defaults to 2.
        offset_y (int, optional): Y offset for sampling. Defaults to 0.

    Returns:
        torch.tensor: Multispectral image data with shape (9, 339, 426)
    """
    input_img = torch.tensor(input_img)
    # remove alignment cross
    # it seems to appear in most images, therefore I change the values by default
    # replace the values with values from the same channel. We have a 3x3 grid, therefore we have to go 3 pixels to the right/bottom
    input_img[512,640] = input_img[515,643]

    input_img[507:512,640] = input_img[507:512,643]
    input_img[513:517,640] = input_img[513:517,643]

    input_img[512,635:640] = input_img[515,635:640]
    input_img[512,641:645] = input_img[515,641:645]
    
    # cut away the offset
    input_img = input_img[offset_x:,offset_y:]
        
    # divide by 3, because of the 3x3 macropixel
    output_height, output_width = int(input_img.shape[0]/3), int(input_img.shape[1]/3)
    num_channels = 9
    multispec_img_data = np.zeros((num_channels, output_height, output_width), dtype=np.float32)

    # I want the values for the image with channel channel_idx
    # input_img[i * 3 + offset_x     , j * 3 + offset_y],  # channel_idx = 0
    # input_img[i * 3 + offset_x + 1 , j * 3 + offset_y],  # channel_idx = 1
    # input_img[i * 3 + offset_x + 2 , j * 3 + offset_y],  # channel_idx = 2
    # input_img[i * 3 + offset_x     , j * 3 + offset_y + 1], # channel_idx = 3
    # input_img[i * 3 + offset_x + 1 , j * 3 + offset_y + 1], # channel_idx = 4
    # input_img[i * 3 + offset_x + 2 , j * 3 + offset_y + 1], # channel_idx = 5
    # input_img[i * 3 + offset_x     , j * 3 + offset_y + 2], # channel_idx = 6
    # input_img[i * 3 + offset_x + 1 , j * 3 + offset_y + 2], # channel_idx = 7
    # input_img[i * 3 + offset_x + 2 , j * 3 + offset_y + 2]  # channel_idx = 8

    # Create lookup tables for indices
    x_offsets = [i for i in range(3)] * 3
    y_offsets = [0] * 3 + [1] * 3 + [2] * 3
    
    for channel_idx in range(num_channels):
        # Calculate indices using broadcasting
        i_indices = np.arange(output_height)[:, np.newaxis] * 3 + x_offsets[channel_idx]
        j_indices = np.arange(output_width) * 3 + y_offsets[channel_idx]

        # Extract values and assign to correct spectral channel
        multispec_img_data[MULTISPECTRAL_CAMERA_FILTER_POS[channel_idx]] = input_img[i_indices, j_indices]
        
    # perform transformation - usually a centercrop
    if transform is not None:
        multispec_img_data = torch.tensor(multispec_img_data)
        multispec_img_data = transform(multispec_img_data)    
        multispec_img_data = np.array(multispec_img_data)
    
    return multispec_img_data


def convert_to_multispec_img(input_img, cross=None, offset_x=2, offset_y=0, transform=None):
    """Convert the 1024x1280 .tif image into the 9 images for each channel and perform the cross correlation correction. Returns two arrays, the corrected array and the not corrected array.

    :param array input_img: 1024x1280 int numpy array
    :param array cross: 9x9 cross correlation matrix
    :param int offset_x: Offset in x dimension
    :param int offset_y: Offset in y dimension
    :return array: cross correlation corrected image of size 9x339x426, image without correction
    """

    # some images have dtype uint8 and some others uint16 and therefore their maximum value differs
    max_dtype_input_img = np.iinfo(input_img.dtype).max

    # Convert input image to float32 to have the same datatype for all images
    input_img = input_img.astype(np.float32)

    # normalize the image for each specific datatype
    input_img = (input_img / max_dtype_input_img) * 255.
    
    # Convert single channel to multispectral image
    multispec_img_data = single_channel_to_multispec_image(input_img, offset_x, offset_y, transform)
    multispec_img_data_corrected = np.copy(multispec_img_data)

    if cross is not None:
        # normalize based on white reference
        # white_reference = multispec_img_data[:,0:11,100:240] # this area is the board that is in the top of the image
        # white_reference_avg = np.mean(white_reference, axis=(1,2))
        # white_reference_scaling = 255/white_reference_avg
        # multispec_img_data_corrected = multispec_img_data_corrected * white_reference_scaling[:, np.newaxis, np.newaxis]
        
        # Cross correlation correction using vectorized operations
        crosstalk_reshaped = cross.T  # Transpose for more efficient dot product
        multispec_img_reshaped = multispec_img_data_corrected.reshape(9, -1)
        multispec_img_data_corrected = np.dot(crosstalk_reshaped, multispec_img_reshaped)
        multispec_img_data_corrected = multispec_img_data_corrected.reshape(multispec_img_data.shape)

    # Clip values to valid range [0, 254]
    multispec_img_data = np.clip(multispec_img_data, 0., 254.)
    multispec_img_data_corrected = np.clip(multispec_img_data_corrected, 0., 254.)
    
    # Normalize to 0. to 1.
    multispec_img_data = multispec_img_data / 255.
    multispec_img_data_corrected = multispec_img_data_corrected / 255.

    return multispec_img_data_corrected, multispec_img_data