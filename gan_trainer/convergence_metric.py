import numpy as np
from torch import nn


class ConvergenceMetric:
    """
    Base class for computing convergence metrics to check the training progress of the condGAN game.
    Should be treated as an abstract class, i.e. it should not be used as it is.
    """

    def __init__(self, params: np.array, vals: np.array, threshold: float = None, use_cuda: bool = False):
        """
        Constructor of a convergence metrics calculator object.
        Since convergence metrics need to compare the training data with the synthetic data created by the Generator, the constructor needs to be given the training data.
        Moreover, a threshold value must be passed to specify, if and when the training process should be stopped.
        :param params: parameter vectors in the training data
        :param vals: values in the training data
        :param threshold: threshold to decide if training can be stopped
        :param use_cuda: specifies, if a cuda device is used for training
        """
        self.use_cuda = use_cuda
        self.threshold = threshold

    def eval_generator(self, generator: nn.Module) -> (dict[str, float], bool):
        """
        Evaluates the metrics for the Generator network in its current training status.
        :param generator: Generator network
        :return: dict object with current statistics and a bool indicating whether training should be stopped
        """
        return {}, False

    def get_statistics_names(self) -> list[str]:
        """
        :return: names of the statistics evaluated in the class. Match the keys of the dict returned by :func: `gan_trainer.convergence_metric.eval_generator`
        """
        return []
