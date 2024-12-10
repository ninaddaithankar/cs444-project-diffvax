import torch
import torch.nn as nn

from typing import Any, List, Optional, Union

import segmentation_models_pytorch as smp

# from segmentation_models_pytorch.base import (
#     ClassificationHead,
#     SegmentationHead,
#     SegmentationModel,
# )
# from segmentation_models_pytorch.encoders import get_encoder

# from .decoder import UnetPlusPlusDecoder


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # Define the layers of UNet

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder1 = self.conv_block(3, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)

        self.final_enc = self.conv_block(512, 1024)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = self.conv_block(1024, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = self.conv_block(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = self.conv_block(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = self.conv_block(128, 64)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
    
    def forward(self, input):
        batch = input.shape[0]

        input = input.reshape(batch, 1, 32, 32) #Reshape according to size of dataset used
        enc1 = self.encoder1(input)
        enc2 = self.encoder2(self.maxpool(enc1))
        enc3 = self.encoder3(self.maxpool(enc2))
        enc4 = self.encoder4(self.maxpool(enc3))

        final_enc = self.final_enc(self.maxpool(enc4))

        dec4 = self.decoder4(torch.cat((self.upconv4(final_enc), enc4), dim=1))
        dec3 = self.decoder3(torch.cat((self.upconv3(dec4), enc3), dim=1))
        dec2 = self.decoder2(torch.cat((self.upconv2(dec3), enc2), dim=1))
        dec1 = self.decoder1(torch.cat((self.upconv1(dec2), enc1), dim=1))

        outputs = self.final(dec1)

        outputs = outputs.reshape(batch, -1)

        return outputs
    

class ImmunizerModel(nn.Module):
    def __init__(self):
        super(ImmunizerModel, self).__init__()
        # self.UNet = UNet()
        #Classes = 2 for mask and background
        #Removed last argument of **kwargs since only applies to timm encoder models
        self.UnetPlusPlus = smp.UnetPlusPlus(encoder_name='resnet18', encoder_depth=5, encoder_weights=None, decoder_use_batchnorm='inplace', decoder_channels=(256, 128, 64, 32, 16), decoder_attention_type='scse', in_channels=3, classes=2, activation=None, aux_params=None)

    def forward(self, image, mask):
        # Generate immunization noise
        epsilon_im = self.UNet(image)
        
        # Apply noise to the masked region
        immunized_image = image + epsilon_im * mask
        
        # Ensure the image stays in the [0, 1] range
        immunized_image = torch.clamp(immunized_image, 0, 1)
        
        return immunized_image, epsilon_im