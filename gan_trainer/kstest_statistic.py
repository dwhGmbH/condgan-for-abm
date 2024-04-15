from typing import Any

import numpy as np
import torch
from scipy.stats import qmc, kstest
from torch import nn
from validation_statistic import ValidationStatistic
from sklearn import cluster as cl


class KSTestStatistic(ValidationStatistic):
    def __init__(self, params, vals, threshold = None, pointsPerCluster = 1000, use_cuda=False):
        super().__init__(params, vals, threshold, use_cuda)
        N = len(vals)
        self.vals = vals
        self.params = params

        clusters = max(1,int(N/pointsPerCluster))
        kmeans = cl.KMeans(clusters).fit(params)
        indices = kmeans.labels_
        centers = kmeans.cluster_centers_

        self.threshold = threshold
        self.reference = dict()
        self.latents = dict()
        self.cubes = dict()

        self.weights = dict()
        for i in range(clusters):
            logicalInds = indices==i
            pts = np.count_nonzero(logicalInds)
            vs = vals[logicalInds]
            self.reference[tuple(centers[i,:])] = vs
            self.latents[tuple(centers[i,:])] = torch.concat([torch.tensor(params[logicalInds,:]),torch.rand((pts,1))],dim=1)
            if self.use_cuda:
                self.latents[tuple(centers[i,:])] =self.latents[tuple(centers[i,:])].cuda()
            self.weights[tuple(centers[i,:])] = pts
            self.cubes[tuple(centers[i,:])] = [params[logicalInds,:],centers[i]]

        sm = sum(self.weights.values())
        #print([min(self.weights.values()),sm/clusters,max(self.weights.values())])
        self.weights = {k:v/sm for k,v in self.weights.items()}

    def plot_overlap(self):
        from matplotlib import pyplot as plt
        xs = list()
        ys = list()
        cols = list()
        ss = list()
        for pts,ctr in self.cubes.values():
            col = [np.random.random(),np.random.random(),np.random.random()]
            xs.extend([x[0] for x in pts[::100]])
            ys.extend([x[1] for x in pts[::100]])
            cols.extend([col for x in pts[::100]])
            ss.extend([1 for x in pts[::100]])
            xs.append(ctr[0])
            ys.append(ctr[1])
            cols.append(col)
            ss.append(5)
        plt.scatter(xs,ys,s=ss,c=cols,marker='.',zorder = 1)
        
    def eval_generator(self,generator:nn.Module) -> (dict[str,float],bool):
        generated=dict()
        for p in self.reference.keys():
            gen = generator(self.latents[p]).detach().cpu()
            generated[p]=np.array([x[0].numpy() for x in gen])
        stats1 = list()
        stats2 = list()
        for p in self.reference.keys():
            st = kstest(self.reference[p], generated[p])
            stats1.append(st[0]*self.weights[p])
            #st = ttest_ind(self.reference[p], generated[p])[0]
            stats2.append(st[1]*self.weights[p])
        s1 = sum(stats1)
        s2 = sum(stats2)
        if self.threshold!=None:
            return {'kstest':s1,'pvalue':s2},s1<self.threshold
        else:
            return {'kstest': s1, 'pvalue': s2}, False

    def get_statistics_names(self):
            return ['kstest','pvalue']


