import os
import numpy as np
from PIL import Image
import torch
from typing import List, Tuple, Optional
import dnnlib
import legacy
from torch_utils import gen_utils

def load_model():
    network_pkl = "stylesan/stylesan-xl_ffhq256.pkl"
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema']
        G = G.eval().requires_grad_(False).to(device)
    return G

def generate_images(G, batch_size: int,  w: torch.Tensor, truncation_psi = 1.0):
    # Generate images.
    w_avg = G.mapping.w_avg.unsqueeze(0).unsqueeze(1).repeat(batch_size, G.mapping.num_ws, 1)
    w = w_avg + (w - w_avg) * truncation_psi
    x = gen_utils.w_to_img(G, w, to_np = False)
    return x

def save_img(x, outdir):
    os.makedirs(outdir, exist_ok=True)
    x = x.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8).cpu().numpy()
    for i in range(x.shape[0]):
        img = Image.fromarray(x[i], mode = "RGB")
        img.save(f'{outdir}/image_{i}.png')
        


    
class PretrainedGenerator(torch.nn.Module):
    def __init__(self, outdir, batch_size, truncation_psi):
        super().__init__()
        self.G =  load_model() # Pretrained generator model, assumed to be loaded externally
        self.outdir = outdir # Directory where images will be saved
        self.batch_size = batch_size
        self.truncation_psi = truncation_psi

    def forward(self, w: torch.Tensor):
        # Generate images using the generate_images function
        images = generate_images(self.G, batch_size=self.batch_size, w=w, truncation_psi=self.truncation_psi)
        
        # Save the generated images using save_img
        # save_img(images, self.outdir)
        
        return images  # Optionally return the images if further processing is needed
    
    def save_img(self, w: torch.Tensor):
        images = generate_images(self.G, batch_size=self.batch_size, w=w, truncation_psi=self.truncation_psi)
        save_img(images, self.outdir)
        
        
# if __name__ == "__main__":
    # x = torch.randn((4, 19, 512)).to("cuda")
    # m = PretrainedGenerator("out", 4, 1.0)
    # y = m(x)
    # print(y.shape)

    # network_pkl = "stylesan-xl_ffhq256.pkl"
    # print('Loading networks from "%s"...' % network_pkl)
    # device = torch.device('cuda')
#     t = torch.randn((4, 19, 512)).to(device)
    # with dnnlib.util.open_url(network_pkl) as f:
        # G = legacy.load_network_pkl(f)['G_ema']
        # G = G.eval().requires_grad_(False).to(device)
#     x = generate_images(G, 4, t)
#     print(x.shape)
#     save_img(x, "out")