#%%
import matplotlib.pyplot as plt
import numpy as np
# import pyro
# import pyro.distributions as dist
import pytorch_lightning as pl
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import wandb
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset

#%%
wandb.login()
# %%
class ExDataset(Dataset):
    def __init__(self, ID=0):
        self.ID = ID
        path_dict = {
            0:'../Data/other/dataset.npz',
            1:'../Data/other/dataset2.npz',
            2:'../Data/other/dataset3.npz'
        }
        arr = np.load(path_dict[ID])
        if ID == 0:
            self.dataset = arr.f.arr_0
            self.labels = None
        else:
            self.dataset = arr.f.dataset
            self.labels = arr.f.IDs
        
        self.data = torch.from_numpy(self.dataset).type(torch.FloatTensor)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        d = self.data[idx]
        # lab = self.labels[idx]
        return d


def load_dataset(idx):
    dataset = ExDataset(idx)
    ldata = len(dataset)
    train_size = int(ldata*0.8)
    val_size = int(ldata*0.1)
    test_size = ldata - train_size - val_size

    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    train_loader = data.DataLoader(train_set, batch_size=512, shuffle=True, drop_last=True, pin_memory=True, num_workers=4)
    val_loader = data.DataLoader(val_set, batch_size=256, shuffle=False, drop_last=True, num_workers=4)
    test_loader = data.DataLoader(test_set, batch_size=256, shuffle=False, drop_last=True, num_workers=4)
    return dict(dataset=dataset, loaders=[train_loader, val_loader, test_loader], sets=[train_set, val_set, test_set])
        

data_dict = load_dataset(2)
dataset = data_dict['dataset']
train_loader, val_loader, test_loader = data_dict['loaders']
train_set, val_set, test_set = data_dict['sets']
#%%

class Encoder(nn.Module):
    
    def __init__(self, dims=[1024, 256, 16]):
        super().__init__()
        # setup the three linear transformations used
        self.dims = dims.copy()
        self.input_dim = self.dims[0]
        self.output_dim = self.dims.pop()
        
        layers = []
        for i in range(len(self.dims)-1):
            layers.append(nn.Linear(self.dims[i], self.dims[i+1]))
            layers.append(nn.GELU())
        
        self.net = nn.Sequential(*layers)
        self.final = nn.Linear(self.dims[-1], self.output_dim)
        self.final_nonl = nn.GELU()
           
    def forward(self, x):
        x = x.reshape(-1, self.input_dim)
        x = self.net(x)
        x = self.final_nonl(self.final(x))
        return x
 
    
class VEncoder(nn.Module):
    
    def __init__(self, dims=[1024, 256, 16]):
        super().__init__()
        # setup the three linear transformations used
        self.dims = dims.copy()
        self.input_dim = self.dims[0]
        self.output_dim = self.dims.pop()
        
        layers = []
        for i in range(len(self.dims)-1):
            layers.append(nn.Linear(self.dims[i], self.dims[i+1]))
            layers.append(nn.GELU())
        
        self.net = nn.Sequential(*layers)
        self.mu = nn.Linear(self.dims[-1], self.output_dim)
        self.logvar = nn.Linear(self.dims[-1], self.output_dim)
        # self.final_nonl = nn.GELU()
           
    def forward(self, x):
        x = x.reshape(-1, self.input_dim)
        x = self.net(x)
        mu = self.mu(x)
        logvar = self.logvar(x)
        return mu, logvar
        
        
class Decoder(nn.Module):
    
    def __init__(self, dims=[16, 256, 1024]):
        super().__init__()
        # setup the three linear transformations used
        self.dims = dims.copy()
        self.input_dim = self.dims[0]
        self.output_dim = self.dims.pop()
        
        layers = []
        for i in range(len(self.dims)-1):
            layers.append(nn.Linear(self.dims[i], self.dims[i+1]))
            layers.append(nn.GELU())
        
        self.net = nn.Sequential(*layers)
        self.final = nn.Linear(self.dims[-1], self.output_dim)
        self.final_nonl = nn.Sigmoid()
        
    def forward(self, x):
        x = x.reshape(-1, self.input_dim)
        x = self.net(x)
        x = self.final_nonl(self.final(x))
        return x


class VAE(pl.LightningModule):
    
    def __init__(self, dims=[1024, 256, 16]):
        super().__init__()
        self.dims = dims
        self.encoder = VEncoder(dims=dims)
        dims.reverse()
        self.decoder = Decoder(dims=dims)
        dims.reverse()
        self.input_dim = dims[0]
        self.example_input_array = torch.zeros(1, self.input_dim)
        self.batch_size = 2**12
        self.lr = 5e-5
        self.name = f'VAE'
        for d in dims:
            self.name += f'_{d}'
        self.save_hyperparameters()
        
    def forward(self, x):
        self.mu, self.logvar = self.encoder(x)
        z = self.reparam()
        rec = self.decoder(z)
        return rec

    def reparam(self):
        std = torch.exp(0.5*self.logvar)
        eps = torch.randn_like(std)
        return self.mu + (eps*std)

    def _get_reconstruction_loss(self, batch):
        x = batch
        xhat = self.forward(x)
        MSE = F.mse_loss(xhat.reshape(-1, self.input_dim), x.reshape(-1, self.input_dim), reduction='sum')
        KL = -0.5 * torch.sum(1 + self.logvar - self.mu.pow(2) - self.logvar.exp())
        self.log("MSE_loss", MSE)
        self.log("KL_loss", KL)
        return MSE + KL

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=20, min_lr=1e-7)
        return dict(optimizer=optimizer, lr_scheduler=scheduler, monitor='val_loss')

    def train_dataloader(self):
        loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        return loader

    def training_step(self, batch, batch_nb):
        loss = self._get_reconstruction_loss(batch)
        self.log("train_loss", loss)
        return {'loss': loss, 'log': {'train_loss': loss}}

    def val_dataloader(self):
        loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False)
        return loader

    def validation_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("val_loss", loss)
        # return {'loss': loss, 'log': {'val_loss': loss}}

    def test_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("test_loss", loss)


class AE(pl.LightningModule):
    
    def __init__(self, dims=[1024, 256, 16]):
        super().__init__()
        self.dims = dims.copy()
        self.rdims = dims.copy()
        self.rdims.reverse()
        self.encoder = Encoder(dims=self.dims)
        dims.reverse()
        self.decoder = Decoder(dims=self.rdims)
        self.input_dim = self.dims[0]
        self.example_input_array = torch.zeros(1, self.input_dim)
        self.batch_size = 2**12
        self.lr = 1e-3
        self.name = 'AE'
        for d in self.dims:
            self.name += f'_{d}'

        self.save_hyperparameters()

    def forward(self, x):
        x = self.encoder(x)
        z = self.decoder(x)
        return z

    def _get_reconstruction_loss(self, batch):
        x = batch
        xhat = self.forward(x)
        MSE = F.mse_loss(xhat.reshape(-1, self.input_dim), x.reshape(-1, self.input_dim), reduction='sum')
        return MSE 

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=20, min_lr=1e-7)
        return dict(optimizer=optimizer, lr_scheduler=scheduler, monitor='val_loss')

    def train_dataloader(self):
        loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        return loader

    def training_step(self, batch, batch_nb):
        loss = self._get_reconstruction_loss(batch)
        self.log("train_loss", loss)
        return {'loss': loss, 'log': {'train_loss': loss}}

    def val_dataloader(self):
        loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False)
        return loader

    def validation_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("val_loss", loss)
        # return {'loss': loss, 'log': {'val_loss': loss}}

    def test_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("test_loss", loss)
    
    

# def train_model(model, date, train_loader, val_loader, test_loader, max_epoch=500):
#     # Create a PyTorch Lightning trainer with the generation callback
#     wandb_logger = WandbLogger(project=f'{model.name}', log_model="all")

#     trainer = pl.Trainer(
#         default_root_dir=f'../Results/Models/{date}/{model.name}/',
#         gpus=1,
#         max_epochs=max_epoch,
#         auto_scale_batch_size=True,
#         callbacks=[
#             ModelCheckpoint(save_weights_only=True, save_last=True, every_n_epochs=10),
#             LearningRateMonitor("epoch"),
#         ],
#         logger=wandb_logger
#     )
#     trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard
#     trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need
#     trainer.tune(model)

#     trainer.fit(model, train_loader, val_loader)
#     # Test best model on validation and test set
#     val_result = trainer.test(model, dataloaders=val_loader, verbose=False)
#     test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
#     result = {"test": test_result, "val": val_result}
#     return model, result


# %%


def train_models_28Jul():

    ldims = [
             [1024, 256, 16],
             [1024, 256, 32],
             [1024, 256, 64],
             [1024, 512, 16],
             [1024, 512, 32],
             [1024, 512, 64],
        ]

    out = {}
    out['bad'] = []

    # model_types = [AE, VAE]
    model_types = [VAE]
    for m in model_types:
        for dim in ldims:
            try:
                model = m(dim)

                out_dir = '25Jul'
                GPU = 1

                config = {
                    'structure': model.name,
                    'GPU': GPU,
                    'n_layers': len(model.dims),
                }

                for i, d in enumerate(model.dims):
                    config[f'layer_{i}'] = d

                wandb_logger = WandbLogger(
                    project=f'{out_dir}', 
                    name=model.name,
                    log_model="all"
                )

                wandb_logger.experiment.config.update(config)

                trainer = pl.Trainer(
                    default_root_dir=f'../Results/Models/{out_dir}',
                    gpus=GPU,
                    max_epochs=1000,
                    auto_scale_batch_size=True,
                    callbacks=[
                        ModelCheckpoint(save_weights_only=True, save_last=True, every_n_epochs=10),
                        LearningRateMonitor("epoch"),
                    ],
                    logger=wandb_logger
                )

                # log gradients and model topology
                wandb_logger.watch(model)

                trainer.tune(model)

                trainer.fit(model)
                # Test best model on validation and test set
                val_result = trainer.test(model, dataloaders=val_loader, verbose=False)
                test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
                result = {"test": test_result, "val": val_result}
                out[model.name] = result
                wandb.finish()
            except Exception as e:
                out['bad'].append((m, dim))
                try:
                    wandb.finish()
                except Exception:
                    pass

    return out

# %%
train_models_28Jul()
# %%

def load_model():
    pass