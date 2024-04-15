import torch
from torch import nn

from config import CondGANConfig


class GenericMLP(nn.Module):
    def __init__(self,sequence,dropout=0.0,finishSigmoid = False):
        super().__init__()
        Q = list()
        for i,j in zip(sequence[:-2],sequence[1:-1]):
            Q.append(nn.Linear(i, j))
            Q.append(nn.ReLU())
            if dropout>0:
                Q.append(nn.Dropout(dropout))
        Q.append(nn.Linear(sequence[-2], sequence[-1]))
        if finishSigmoid:
            Q.append(nn.Sigmoid())
        self.model = nn.Sequential(*Q)
    def forward(self, x):
        output = self.model(x)
        return output


def create_mlps_from_config(config:CondGANConfig):
    params = config.get_generator_hyperparams()
    generator = GenericMLP(params['sequence'], params['dropout'], params['finishSigmoid'])
    params = config.get_critic_hyperparams()
    critic = GenericMLP(params['sequence'], params['dropout'], params['finishSigmoid'])
    return generator,critic

def load_generator_from_state(config:CondGANConfig,filename:str):
    params = config.get_generator_hyperparams()
    generator = GenericMLP(params['sequence'], params['dropout'], params['finishSigmoid'])
    generator.load_state_dict(torch.load(filename))

def load_critic_from_state(config:CondGANConfig,filename:str):
    params = config.get_critic_hyperparams()
    generator = GenericMLP(params['sequence'], params['dropout'], params['finishSigmoid'])
    generator.load_state_dict(torch.load(filename))