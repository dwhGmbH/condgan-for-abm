import os
import pickle
from typing import Callable
import numpy as np


def create_synthetic_training_data(dataid: str, parameterspace: list[tuple[float, float]], function: Callable, N: int,
                                   seed: int = 12345):
    """
    Method which creates synthetic training data for training of a conditional GAN and saves them to the pickle format.

    In the first step, the method creates N random parameter vectors u_1...u_N.
    The vectors are sampled uniformly iid from the specified parameterspace

    For every sampled vector u_i, the method computes y_i = function(u_i,X_i) whereas X_i refers to a U(0,1) random noise
    The resulting y_i,i=1,...,N poses the value vector.

    Together, (u_i,y_i),i=1,...,N form the training data for the condGAN and will be saved to a pickle file

    :param dataid: some identifyer for the synthetic data
    :param parameterspace: Defines the parameter space. The length of the list specifies the length of the input parameter vector. The tuple entries refer to the lower and upper bound of the parameter.
    :param function: function imitating a complex decision process and which the condGAN should try to learn. The input dimension must be one larger than the dimension of the parameterspace, sicne a U(0,1) noise is added.
    :param N: number of training data sets to be created synthetically
    :param seed: optional, seed for the PRNG to make the process reproducible
    """
    np.random.seed(seed)
    dim = len(parameterspace)
    rands = np.random.rand(N, dim)
    lowers = np.array([x[0] for x in parameterspace])
    span = np.array([x[1] for x in parameterspace]) - lowers
    parameters = rands * span + lowers
    rands2 = np.random.rand(N)
    try:
        _temp = function(*parameters[0], rands2[0])
    except:
        raise AssertionError(
            "dimension mismatch: input dimension of the function must be equal to the size of parameterspace plus one")
    values = np.array([function(*x, y) for x, y in zip(parameters, rands2)])
    picklefile = os.path.join('..', 'data', 'trainingSet_' + dataid + '.pickle')
    with open(picklefile, 'wb') as f:
        pickle.dump((parameters, values), f)


if __name__ == '__main__':
    """
    This script creates a synthetic training data from a sinewave-shifted U(0,1) distribution
    """
    function = lambda a, b, X: a + np.sin(b * 2 * np.pi) + (X - 0.5)
    parameterspace = [(-1, 1), (0, 1)]
    N = 2 ** 16
    create_synthetic_training_data('synthetic_sinewave', parameterspace, function, N)
