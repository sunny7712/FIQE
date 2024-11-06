import timm
import torch
import argparse
import torch.nn as nn
from stylesan.inference import *
from stylegan2_pytorch.stylegan2.models import Discriminator
from stylegan2_pytorch.stylegan2 import models

class ResnetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        model = timm.create_model("resnet34", pretrained = True)
        self.backbone = nn.Sequential(*list(model.children())[:-1])
        self.fc = nn.Linear(512, 19 * 512)  # Linear layer to expand dimensions
    
    def forward(self, x):
        # x -> (batch, 3, 1024, 1024)
        x = self.backbone(x)
        # x ->(batch, 512)
        x = self.fc(x)
        # x ->(batch, 19 * 512)
        x = x.view(-1, 19, 512)
        # x ->(batch, 19, 512)
        return x

# class PretrainedGenerator(torch.nn.Module):
#     def __init__(self, outdir, batch_size, truncation_psi):
#         super().__init__()
#         self.G =  load_model() # Pretrained generator model, assumed to be loaded externally
#         self.outdir = outdir # Directory where images will be saved
#         self.batch_size = batch_size
#         self.truncation_psi = truncation_psi

#     def forward(self, w: torch.Tensor):
#         # Generate images using the generate_images function
#         images = generate_images(self.G, batch_size=self.batch_size, w=w, truncation_psi=self.truncation_psi)
        
#         # Save the generated images using save_img
#         # save_img(images, self.outdir)
        
#         return images  # Optionally return the images if further processing is needed
    
#     def save_img(self, w: torch.Tensor):
#         images = generate_images(self.G, batch_size=self.batch_size, w=w, truncation_psi=self.truncation_psi)
#         save_img(images, self.outdir)