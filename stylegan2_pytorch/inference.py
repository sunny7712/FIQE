import warnings
import argparse
import os
from PIL import Image
import numpy as np
import torch
from run_generator import generate_images
import stylegan2
from stylegan2 import utils


# Hardcode the args with the values you want to test
args = argparse.Namespace(
    command='generate_images',
    batch_size=4,
    seeds=[6000, 6001, 6002, 6003],  # You can modify the seeds if you want to generate more images
    network='G.pth',  # Path to your trained model (ensure this is correct)
    output='./results',  # Output folder where the generated images will be saved
    pixel_min=-1,        # Pixel normalization minimum
    pixel_max=1,         # Pixel normalization maximum
    gpu=[0],              # Use [] if you're not using a GPU, or specify your GPU ids if using one
    truncation_psi=0.5   # Truncation trick parameter, useful for controlling diversity in GANs
)

latent_size = 512  
latents = torch.randn((args.batch_size, latent_size))
# print(latents.shape)

G = stylegan2.models.load(args.network)
G.eval()
with torch.no_grad():
    generated = G(latents, labels=None)
    # print(generated.shape)

images = utils.tensor_to_PIL(
    generated, pixel_min=args.pixel_min, pixel_max=args.pixel_max)

    # Save the generated images
for idx, img in enumerate(images):
    img.save(os.path.join(args.output, 'latent_image_%04d.png' % idx))

print("Image generation from latents completed.")
