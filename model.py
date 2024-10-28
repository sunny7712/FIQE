import timm
import torch
import argparse
import torch.nn as nn
from stylegan2_pytorch.stylegan2.models import Discriminator
from stylegan2_pytorch.stylegan2 import models

class ResnetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        model = timm.create_model("resnet34", pretrained = True)
        self.backbone = nn.Sequential(*list(model.children())[:-1])
    
    def forward(self, x):
        # x -> (batch, 3, 1024, 1024)
        x = self.backbone(x)
        # x ->(batch, 512)
        return x

class PretrainedGenerator(nn.Module):
    def __init__(
        self,
        model_path: str,
        batch_size: int = 4,
        seed: int = 6000,
        latent_size: int = 512,
        pixel_min: float = -1,
        pixel_max: float = 1,
        truncation_psi: float = 0.5,
    ):
        super().__init__()
        args = argparse.Namespace(
            command='generate_images',
            batch_size=batch_size,
            seeds=[seed + i for i in range(0, batch_size)], 
            network=model_path,  
            output='./results',  
            pixel_min=pixel_min,        
            pixel_max=pixel_max,        
            gpu=[0],              # Use [] if you're not using a GPU, or specify your GPU ids if using one
            truncation_psi=truncation_psi   # Truncation trick parameter, useful for controlling diversity in GANs
        )
        
        self.G = models.load(args.network)
    
    # @torch.no_grad()
    def forward(self, x):
        # x -> (batch, 512)
        x = self.G(x, labels = None)
        # x -> (batch, 3, 1024, 1024)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.D = Discriminator()
    
    def forward(self, x):
        # x -> (batch, 3, 1024, 1024)
        x = self.D(x)
        # x -> (batch, 1)
        return x