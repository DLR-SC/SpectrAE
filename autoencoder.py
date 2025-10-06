#
# Author(s): Peer Schütt
#
# See the file "LICENSE" for the full license and copyright governing this code.
#

import torch.nn as nn
from datetime import datetime
from torchinfo import summary

def conv_block(in_channels, out_channels, dropout_rate):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.1, inplace=True),
        nn.BatchNorm2d(out_channels),
        nn.Dropout(p=dropout_rate),
    )

def conv_transpose_block(in_channels, out_channels, dropout_rate, only_conv=False):
    module_list = []
    # module_list.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1))
    module_list.append(nn.Upsample(scale_factor=2.))
    module_list.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))

    if only_conv is False:
        module_list.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
        module_list.append(nn.BatchNorm2d(out_channels))
        module_list.append(nn.Dropout(p=dropout_rate))

    return nn.Sequential(*module_list)

# Define the Autoencoder architecture
class Autoencoder(nn.Module):
    def __init__(self, channels_to_use:list, batch_size:int, img_size_x:int, img_size_y:int, img_type:str, model_name=None, cs=16, dense_layer_dim=1024, dropout=0., autoencoder_depth=2, **kwargs):
        """Initialize the autoencoder model.

        Args:
            channels_to_use (list): List of channels to use from input
            batch_size (int): Batch size used for model summary
            img_size_x (int): Width of input images
            img_size_y (int): Height of input images
            img_type (str): Type of images being processed
            model_name (str, optional): Name for the model. Defaults to timestamp if None
            cs (int, optional): Initial channel dimension. Defaults to 16
            dense_layer_dim (int, optional): Dimension of latent space. Defaults to 1024
            dropout (float, optional): Dropout rate. Defaults to 0.0
            autoencoder_depth (int, optional): Number of downsampling layers. Defaults to 2
        """
        in_channels = len(channels_to_use)

        super(Autoencoder, self).__init__()
        self.in_channels = in_channels
        self.batch_size = batch_size
        self.dense_layer_dim = dense_layer_dim
        self.img_size_y = img_size_y
        self.img_size_x = img_size_x

        if model_name is None:
            now = datetime.now()
            now_date = now.strftime("%Y-%m-%d_%H-%M-%S")
            self.model_name = f"{now_date}_autoencoder_{img_type}"
        else:
            self.model_name = model_name

        # we actually have one depth less, because we have a mandatory sequential layer at the beginning and end
        autoencoder_depth = autoencoder_depth-1

        # create the module list for the encoder
        encoder_modules = []
        encoder_modules.append(conv_block(in_channels, cs, dropout))
        for i in range(0,autoencoder_depth):
            encoder_modules.append(conv_block(cs * 2**i, cs * 2**(i+1), dropout))
        encoder_modules.append(nn.Flatten())
        encoder_modules.append(nn.Linear(in_features=int(cs * 2**autoencoder_depth * (img_size_x / (2**autoencoder_depth * 2)) * (img_size_y / (2**autoencoder_depth * 2))),
                                         out_features=dense_layer_dim))

        # create the module list for the decoder
        decoder_modules = []
        decoder_modules.append(nn.Linear(in_features=dense_layer_dim,
                                         out_features=int(cs * 2**autoencoder_depth * (img_size_x / (2**autoencoder_depth * 2)) * (img_size_y / (2**autoencoder_depth * 2)))))
        decoder_modules.append(nn.Unflatten(1, (cs * 2**autoencoder_depth, int(img_size_x/(2**autoencoder_depth * 2)),int(img_size_y/(2**autoencoder_depth * 2)))))
        for i in reversed(range(0,autoencoder_depth)):
            decoder_modules.append(conv_transpose_block(cs * 2**(i+1), cs * 2**i, dropout))
        decoder_modules.append(conv_transpose_block(cs, in_channels, dropout, only_conv=True))
        # Sigmoid activation to output values between 0 and 1
        decoder_modules.append(nn.Sigmoid())

        # combine the module lists to a model
        self.encoder = nn.Sequential(*encoder_modules)
        self.decoder = nn.Sequential(*decoder_modules)

        # log the model info
        # logging.debug(self.get_model_summary_str(device="cpu"))

    def forward(self, x):
        """Forward pass."""
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self, x):
        """Only the encode part of the forward pass."""
        x = self.encoder(x)
        return x

    def decode(self, x):
        x = self.decoder(x)
        return x

    @property
    def latent_dim(self):
        return self.dense_layer_dim

    def get_model_params_dict(self):
        return self.model_params_dict

    def get_model_summary_str(self, device="cpu"):
        """Get the string of the summary of the model using the torchinfo package."""
        summmary_of_model = "Model Summary:\n"
        summmary_of_model += "ENCODER:\n"
        summmary_of_model += str(summary(self.encoder, input_size=(self.batch_size, self.in_channels, self.img_size_x, self.img_size_y), verbose=0, device=device))
        summmary_of_model += "\n\nDECODER:\n"
        summmary_of_model += str(summary(self.decoder, input_size=(self.batch_size, self.dense_layer_dim), verbose=0, device=device))
        summmary_of_model += "\n"
        return summmary_of_model
