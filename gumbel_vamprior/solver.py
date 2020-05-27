#%%
from os import environ
environ['MLFLOW_TRACKING_URI'] = 'http://tracking.olympus.nintorac.dev:9001/'

import torch

from pathlib import Path
from importlib import import_module
from itertools import starmap

from ray import tune
from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset
from torch.optim.lr_scheduler import StepLR
from torch.nn import functional as F
from torch import nn
from torch.optim import Adadelta
from torch.nn import Module
from agoge import AbstractModel, AbstractSolver, TrainWorker as Worker
from agoge.utils import uuid, trial_name_creator as trial_name_creator, experiment_name_creator, get_logger


import torch
from torch.nn import functional as F
from importlib import import_module
from torch.optim import AdamW
from torch.distributions.kl import kl_divergence
from torch.distributions import Normal, Bernoulli
from agoge import AbstractSolver

import torch

from gumbel_vamp import VampAutoEncoder

import logging

logging.basicConfig(level=logging.INFO)

logger = get_logger(__name__)


class MNISTDataset():

    def __init__(self, data_path='~/datasets', transform=None):
        
        transform = transforms.Compose([
                           transforms.ToTensor(),
                        #    transforms.Normalize((0.1307,), (0.3081,))
                       ])

        if not isinstance(data_path, Path):
            data_path = Path(data_path).expanduser()

        train_dataset = datasets.MNIST(data_path.as_posix(), train=True, download=True,transform=transform)
        test_dataset = datasets.MNIST(data_path.as_posix(), train=False, download=True, transform=transform)

        self.dataset = ConcatDataset((train_dataset, test_dataset))

    def __getitem__(self, i):

        x, _ = self.dataset[i]
        x = x.flatten(-2, -1)
        x = Bernoulli(probs=x).sample()
        
        return {'x': x}

    def __len__(self):

        return len(self.dataset)


def sigmoidal_annealing(iter_nb, t=1e-4, s=-6):
    """

    iter_nb - number of parameter updates completed
    t - step size
    s - slope of the sigmoid
    """
    
    t, s = torch.tensor(t), torch.tensor(s).float()
    x0 = torch.sigmoid(s)
    value = (torch.sigmoid(iter_nb*t + s) - x0)/(1-x0) 

    return value

class Solver(AbstractSolver):
 
    def __init__(self, model,
        Optim=AdamW, optim_opts=dict(lr= 1e-4),
        max_beta=0.5,
        beta_temp=1e-4,
        **kwargs):

        if isinstance(Optim, str):
            Optim = import_module(Optim)

        self.optim = Optim(params=model.parameters(), **optim_opts)
        self.max_beta = max_beta
        self.model = model

        self.iter = 0
        self.beta_temp = beta_temp

    def loss(self, x, x_hat, z, posterior, prior):

        # calculate vampprior likelihood
        n_inputs, *_ = z.shape
        p_z = prior.log_prob(z.unsqueeze(-2)).sum(-1) - torch.log(torch.Tensor([n_inputs]))
        p_z = p_z.logsumexp(-1)

        # calculate posterior likelihood
        q_z = posterior.log_prob(z).sum(-1)

        assert q_z.shape == p_z.shape
        kl = (q_z - p_z).mean()


        beta = sigmoidal_annealing(self.iter, self.beta_temp).item()

        reconstruction_loss = F.binary_cross_entropy_with_logits(x_hat, x)

        loss = reconstruction_loss + self.max_beta * beta * kl

        return loss, {
            'reconstruction_loss': reconstruction_loss,
            'kl': kl,
            'beta': beta,
            'q_z': q_z.mean(),
            'p_z': p_z.mean(),
            'loss': loss
            # 'iter': self.iter // self.
        }
        

    def solve(self, X, **kwargs):
        
        Y = self.model(**X)
        loss, L = self.loss(**X, **Y)

        if loss != loss:
            raise ValueError('Nan Values detected')

        if self.model.training:
            self.iter += 1
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

        return L

    
    def step(self):

        pass


    def state_dict(self):
        
        state_dict = {
            'optim': self.optim.state_dict(),
            'iter': self.iter
        }

        return state_dict

    def load_state_dict(self, state_dict):
        
        self.optim.load_state_dict(state_dict['optim'])
        self.iter = state_dict['iter']



#########################
def config(Model, Solver, experiment_name, trial_name, batch_size=128, param_gumbel=True, param_t=1., **kwargs):

    data_handler = {
        'Dataset': MNISTDataset,
        'dataset_opts': {'data_path': '~/audio/artifacts/'},
        'loader_opts': {
            'batch_size': batch_size,
            'shuffle': True,
            'num_workers': 1
        },
    }

    model = {
        'Model': Model,
        'gumbel': gumbel if param_t >= 0 else 0,
        't': param_t
    }

    solver = {
        'Solver': Solver,
        'max_beta': 1,
        'beta_temp': 1,
    }

    tracker = {
        'metrics': [
            'reconstruction_loss',
            'kl',
            'beta',
            'q_z',
            'p_z',
            'loss',
        ],
        'experiment_name': experiment_name,
        'trial_name': trial_name
    }

    return {
        'data_handler': data_handler,
        'model': model,
        'solver': solver,
        'tracker': tracker,
    }

if __name__=='__main__':
    from mlflow.tracking import MlflowClient

    gumbel = 1 #tune.grid_search([1, 0])
    param_t = tune.grid_search(torch.linspace(1e-4, 1, 10).numpy().tolist()+[-1])
    # client = MlflowClient(tracking_uri='localhost:5000')
    experiment_name = 'vamp-autoencoder-grid-t'#+experiment_name_creator()
    # experiment_id = client.create_experiment(experiment_name)

    tune.run(
        Worker, 
        config={
            'config_generator': config,
            'experiment_name': experiment_name,
            'Model': VampAutoEncoder,
            'Solver': Solver,
            'param_gumbel': gumbel,
            'param_t': param_t
        },
        trial_name_creator=trial_name_creator,
        resources_per_trial={
            'cpu': 2
        },
        num_samples=10,
        stop={
            "training_iteration": 2
        }
    )
# points_per_epoch
# %%
bool(1), bool(0)
