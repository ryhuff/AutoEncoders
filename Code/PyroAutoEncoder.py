#%%
import os

import numpy as np
import pyro
import pyro.contrib.examples.util  # patches torchvision
import pyro.distributions as dist
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from pyro.contrib.examples.util import MNIST
from pyro.infer import SVI, JitTrace_ELBO, Trace_ELBO
from pyro.optim import Adam

# %%

class Encoder(nn.Module):
    
    def __init__(self, output_dim=16, input_dim=1024):
        super().__init__()
        # setup the three linear transformations used
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.GELU(),
            nn.Linear(512, 128),
            nn.GELU()
        )
        self.means = nn.Linear(128, self.output_dim)
        self.stds = nn.Linear(128, self.output_dim)
        
    
    def forward(self, x):
        x = x.reshape(-1, self.input_dim)
        x = self.net(x)
        z_loc = self.means(x)
        z_scale = torch.exp(self.stds(x))
        return z_loc, z_scale
    

class Decoder(nn.Module):
    
    def __init__(self, input_dim=16, output_dim=1024):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.net = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.GELU(),
            nn.Linear(128, 512),
            nn.GELU(),
            nn.Linear(512, self.output_dim),
            nn.Sigmoid(),
        )

    def forward(self, z):
        z = z.reshape((-1, self.input_dim))
        z = self.net(z)
        # z = (z - torch.min(z)) / (torch.max(z) - torch.min(z))
        return z
    
    
class VAE(nn.Module):
   
    def __init__(self, input_dim=1024, latent_dim=32, use_cuda=False):
        super().__init__()
        # create the encoder and decoder networks
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = Encoder(input_dim=input_dim, output_dim=latent_dim)
        self.decoder = Decoder(input_dim=latent_dim, output_dim=input_dim)

        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()
        self.use_cuda = use_cuda
        self.z_dim = latent_dim

    # define the model p(x|z)p(z)
    def model(self, x):
        # register PyTorch module `decoder` with Pyro
        x = x.reshape(-1, self.input_dim)
        pyro.module("decoder", self.decoder)
        with pyro.plate("data", x.shape[0]):
            # setup hyperparameters for prior p(z)
            z_loc = x.new_zeros(torch.Size((x.shape[0], self.z_dim)))
            z_scale = x.new_ones(torch.Size((x.shape[0], self.z_dim)))
            # sample from prior (value will be sampled by guide when computing the ELBO)
            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
            # decode the latent code z
            loc_img = self.decoder(z)
            # score against actual images
            pyro.sample("obs", dist.Bernoulli(loc_img.reshape(-1,self.input_dim)).to_event(1), obs=x.reshape(-1, self.input_dim))

    # define the guide (i.e. variational distribution) q(z|x)
    def guide(self, x):
        # register PyTorch module `encoder` with Pyro
        x = x.reshape(-1, 1, self.input_dim)
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.shape[0]):
            # use the encoder to get the parameters used to define q(z|x)
            z_loc, z_scale = self.encoder(x)
            z_loc = z_loc.reshape(-1, self.latent_dim)
            z_scale = z_scale.reshape(-1, self.latent_dim)
            # sample the latent code z
            pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1)).reshape(-1, self.latent_dim)

    # define a helper function for reconstructing images
    def reconstruct_img(self, x):
        # encode image x
        z_loc, z_scale = self.encoder(x)
        # sample in latent space
        z = dist.Normal(z_loc, z_scale).sample()
        # decode the image (note we don't sample in image space)
        loc_img = self.decoder(z)
        return loc_img
# %%

pvae32 = VAE(latent_dim=32)

optimizer = Adam({"lr": 1.0e-4})
svi = SVI(pvae32.model, pvae32.guide, optimizer, loss=Trace_ELBO())
# %%
svi.step(d)
# %%
