import numpy as np
import torch
from scipy.stats import qmc, kstest
from torch import nn

from gan_trainer.range_finder import RangeFinder
from validation_statistic import ValidationStatistic
from sklearn import cluster as cl


class KSTestStatistic(ValidationStatistic):
    def __init__(self, params, vals, threshold = None, points = 1000, use_cuda=False):
        super().__init__(params, vals, threshold, use_cuda)
        dim = np.size(params, 1)
        sampler = qmc.Halton(d=dim, scramble=False)
        sampled = sampler.random(n=points)
        self.threshold = threshold
        self.reference = dict()
        self.latents = dict()
        self.range_finder = RangeFinder(params)

        self.cubes = dict()
        self.vals = vals
        self.params = params

        target_count = len(vals) / points

        for sample in sampled:
            radius, logicalInds, pts = self.range_finder.find_in_range(sample, target_count)
            # print(pts)
            vs = vals[logicalInds]
            self.cubes[tuple(sample)] = [sample, radius]
            self.reference[tuple(sample)] = vs
            self.latents[tuple(sample)] = torch.concat([torch.tensor(params[logicalInds]), torch.rand((pts, 1))], dim=1)
            if self.use_cuda:
                self.latents[tuple(sample)] = self.latents[tuple(sample)].cuda()

    def plot_overlap(self):
        from matplotlib import pyplot as plt
        for k in self.reference.keys():
            mid = self.cubes[k][0]
            rad = self.cubes[k][1]
            plt.plot([mid[0] - rad, mid[0] - rad, mid[0] + rad, mid[0] + rad, mid[0] - rad],
                     [mid[1] - rad, mid[1] + rad, mid[1] + rad, mid[1] - rad, mid[1] - rad], linewidth=0.1, zorder=0)
        plt.scatter([x[0] for x in self.params[::1000]], [x[1] for x in self.params[::1000]], s=0.1, color='k',
                    marker='.', zorder=1)
        
    def eval_generator(self,generator:nn.Module) -> (dict[str,float],bool):
        generated=dict()
        for p in self.reference.keys():
            gen = generator(self.latents[p]).detach().cpu()
            generated[p]=np.array([x[0].numpy() for x in gen])
        stats1 = list()
        stats2 = list()
        for p in self.reference.keys():
            st = kstest(self.reference[p], generated[p])
            stats1.append(st[0])
            # st = ttest_ind(self.reference[p], generated[p])[0]
            stats2.append(st[1])
        s1 = sum(stats1) / len(stats1)
        s2 = sum(stats2) / len(stats2)
        if self.threshold != None:
            return {'kstest': s1, 'pvalue': s2}, s1 < self.threshold
        else:
            return {'kstest': s1, 'pvalue': s2}, False

    def get_statistics_names(self):
            return ['kstest','pvalue']


