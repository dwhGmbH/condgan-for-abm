import torch
from torch import nn
from condgan_config import CondGANConfig


class GenericMLP(nn.Module):
    """
    Generic multilayer perceptron. Wraps the torch.nn.Sequential class and simplfies its initialization
    """

    def __init__(self, sequence: list[int], dropout: float = 0.0, finish_sigmoid: bool = False):
        """
        Wraps the torch.nn.Sequential class to simplify the initialization process.
        The network will have len(sequence) fully and linearly connected layers whereas layer i has the corresponding number sequence[i] of nodes
        ReLu activation functions are used for all layers.
        if dropout > 0, then the corresponding dropout is added to each layer.
        The output layer adds a linear function, and, if finishSigmoid, also a sigmoidal function.
        For example, ([1,3,1],0.0, False) would return the network:
        Sequential(Linear(1, 3),ReLu(),Linear(3,1),Linear())

        :param sequence: sequence of integers referring to the layer sizes of the multilyer perceptron
        :param dropout: number between 0 and 1 specifying whether and how much dropout is used
        :param finish_sigmoid: if the network is finished with sigmoid activation function
        """
        super().__init__()
        Q = list()
        for i, j in zip(sequence[:-2], sequence[1:-1]):
            Q.append(nn.Linear(i, j))
            Q.append(nn.ReLU())
            if dropout > 0:
                Q.append(nn.Dropout(dropout))
        Q.append(nn.Linear(sequence[-2], sequence[-1]))
        if finish_sigmoid:
            Q.append(nn.Sigmoid())
        self.model = nn.Sequential(*Q)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the model.
        :param x: input to the network
        :return: output of the network
        """
        output = self.model(x)
        return output


def create_mlps_from_config(config: CondGANConfig) -> (GenericMLP, GenericMLP):
    """
    Creates untrained new networks accroding to the configuration file
    :param config: configuration of the condGAN training experiment
    :return: Generator and Critic network
    """
    params = config.get_generator_hyperparams()
    generator = GenericMLP(params['sequence'], params['dropout'], params['finishSigmoid'])
    params = config.get_critic_hyperparams()
    critic = GenericMLP(params['sequence'], params['dropout'], params['finishSigmoid'])
    return generator, critic


def load_generator_from_state(config: CondGANConfig, filename: str) -> GenericMLP:
    """
    Loads a generator from a saved state.
    :param config: configuration of the condGAN training experiment to specify structure of the generator
    :param filename: full path to the .state file
    :return: Generator network
    """
    params = config.get_generator_hyperparams()
    generator = GenericMLP(params['sequence'], params['dropout'], params['finishSigmoid'])
    generator.load_state_dict(torch.load(filename))
    return generator


def load_critic_from_state(config: CondGANConfig, filename: str):
    """
    Loads a critic from a saved state.
    :param config: configuration of the condGAN training experiment to specify structure of the generator
    :param filename: full path to the .state file
    :return: Critic network
    """
    params = config.get_critic_hyperparams()
    critic = GenericMLP(params['sequence'], params['dropout'], params['finishSigmoid'])
    critic.load_state_dict(torch.load(filename))
    return critic
