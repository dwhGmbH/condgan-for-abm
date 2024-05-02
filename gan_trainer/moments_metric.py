import numpy as np
import torch
from torch import nn

from convergence_metric import ConvergenceMetric

class MomentsMetric(ConvergenceMetric):
    """
    Class to compute the epsilon metrics w.r. to equivalence of moments
    """
    def __init__(self, params:np.array, vals:np.array, threshold:float=None, use_cuda:bool=False, k:int = 4, mu:np.array = None):
        """
        Constructor of the metric calculator. Computes the epsilon_i, i=1,...,k metrics and the sum-metric epsilon_(1,k).
        Termination is decided based on the value of the sum-metric.
        :param params: parameter vectors in the training data
        :param vals: values in the training data
        :param threshold: threshold to decide if training can be stopped
        :param use_cuda: specifies, if a cuda device is used for training
        :param k: number of epsilons to compute (1 for epsilon_1, 2 for epsilon_1 and epsilon_2, ...)
        :param mu: weights when computing the epsilon_(1,k) metric from the individual epsilon_i metrics
        """
        super().__init__(params, vals, threshold, use_cuda)
        self.momentsCount = k
        if mu==None:
            mu = [1.0 for x in range(self.momentsCount)]
        else:
            assert len(mu) == k, AssertionError('length of weights must match the number of moments used')
        self.momentsWeights = np.array(mu)
        self.statistic_names = ['$\\epsilon_'+str(k+1)+'$' for k in range(self.momentsCount)]
        self.statistic_names.append('$\\epsilon_{1,'+str(self.momentsCount)+'}$')
        self.threshold = threshold
        self.vals = vals
        self.latents = torch.concat([torch.tensor(params),torch.rand((len(vals),1))],dim=1)
        if use_cuda:
            self.latents = self.latents.cuda()


    def eval_generator(self,generator:nn.Module) -> (dict[str,float],bool):
        """
        Evaluates the metrics for the Generator network in its current training status.
        Termination is decided based on the value of the sum-metric epsilon_(1,k).
        :param g: Generator network
        :return: dict object with current statistics and a bool indicating whether training should be stopped
        """
        generated= np.array([x[0].numpy() for x in generator(self.latents).detach().cpu()])
        Xs = [0 for i in range(self.momentsCount)]
        for k in range(self.momentsCount):
            diff = generated ** (k + 1) - (self.vals) ** (k + 1)
            Xs[k] += np.sum(diff) / len(diff)
        Xs = [abs(x) for x in Xs]
        Xs.append(sum([x * y for x, y in zip(Xs, self.momentsWeights)]))
        stats = {x: y for x, y in zip(self.statistic_names, Xs)}
        if self.threshold==None:
            return stats, False
        else:
            return stats,Xs[-1]<self.threshold

    def get_statistics_names(self):
        """
        :return: names of the statistics evaluated in the class. Match the keys of the dict returned by :func: `gan_trainer.moments_metric.MomentsMetric.eval_generator`
        """
        return self.statistic_names


