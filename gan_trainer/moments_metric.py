import numpy as np
import torch
from torch import nn

from convergence_metric import ConvergenceMetric

class MomentsMetric(ConvergenceMetric):
    def __init__(self, params, vals, threshold=None, momentsCount = 4, momentsWeights = None, use_cuda=False):
        super().__init__(params, vals, threshold, use_cuda)
        self.momentsCount = momentsCount
        if momentsWeights==None:
            momentsWeights = [1.0 for x in range(self.momentsCount)]
        self.momentsWeights = np.array(momentsWeights)
        self.statistic_names = ['$\\epsilon_'+str(k+1)+'$' for k in range(self.momentsCount)]
        self.statistic_names.append('$\\epsilon_{1,'+str(self.momentsCount)+'}$')
        self.threshold = threshold
        self.vals = vals
        self.latents = torch.concat([torch.tensor(params),torch.rand((len(vals),1))],dim=1)
        if use_cuda:
            self.latents = self.latents.cuda()


    def eval_generator(self,generator:nn.Module):
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
            return self.statistic_names


