import numpy as np
import torch
from scipy.stats import qmc, kstest
from torch import nn

from gan_trainer.range_finder import RangeFinder
from convergence_metric import ConvergenceMetric


class KSTestMetric(ConvergenceMetric):
    def __init__(self, params:np.array, vals:np.array, threshold:float = None, use_cuda:bool=False, m:int = 1000, radius:float = None, s:int = None):
        """
        Constructor of the metric calculator.
        Computes the metrics based on a series Kolmogorov-Smirnoff tests for a series of selected points within the parameterset.
        The idea given as follows: Let (u_i,y_i),i=1,...,N stand for traing data and define J_x subset of {1,...,N}, so that
        all u_j, j=J_x lie within a certain Epsilon Ball U_eps around point x.
        Since epsilon is chosen small, we assume that all y_j,j in J follow the same unknown and unconditional distribution D_x.
        To evaluate the metric for a generator G we compute y*_j = G(u_j,X_j), j=J_x with an iid U(0,1) noise X, and evaluate whether y*_j is also D_x disrtibuted
        This is done using a classic Kolmogorov Smirnoff test.

        The strategy is implemented as follows: total of m points are selected from the hypercube using a Halton sequence.
        First, m points x_i,i=1..m, are selected from the [0,1]^d hypercube using a Halton sequence to fill the parameter space properly.
        For each of these points x_i we define the epsilon ball using either of two strategies:
        - if the radius parameter is not None, the epsilon ball is defined by the given radius
        - if the radius parameter is None, and the s parameter is not None, the epsilon ball is defined by the nearest s parameter vectors in the training data around x_j
        - if both parameters are unspecified, the epsilon ball is defined by the nearest N/m parameter vectors in the training data around x_j
        The corresponding set u_j, j=J_(x_i), together with a random noise X_j, j=J_(x_i) furthermore defines a latent set for the evaluation of the generator
        For every point x_i, i=1,..,m we evaluate KSTest(y_j, G(u_j,X_j)), j=J_(x_i) and take the average of both, KSTest statistic and p-value.

        Note, that for performance reasons, an epsilon ball is regarded w.r. to the L1 norm!

        :param params: parameter vectors in the training data
        :param vals: values in the training data
        :param threshold: threshold to decide if training can be stopped
        :param use_cuda: specifies, if a cuda device is used for training
        :param m: number of points in which the latent-set is divided
        :param radius: radius of the epsilon balls
        :param s: number of parameter vectors included in each epsilon ball. Ununsed, if radius is not equal to None.
        """
        super().__init__(params, vals, threshold, use_cuda)
        dim = np.size(params, 1)
        self.threshold = threshold
        self.reference = dict()
        self.latents = dict()
        self.range_finder = RangeFinder(params)

        self.spheres = dict()
        self.vals = vals
        self.params = params
        if radius == None and s==None:
            s = int(len(vals) / m)
        sampler = qmc.Halton(d=dim, scramble=False)
        sampled = sampler.random(n=m)
        for sample in sampled: #iterate over sampled points
            if radius !=None:
                r = radius
                inds, pts = self.range_finder.find_in_radius(sample, radius) #find all parameter vectors in epsilon ball
            else:
                inds, r = self.range_finder.find_nearest_s(sample, s) #find all parameter vectors in epsilon ball
            vs = vals[inds] #find corresponding values
            self.spheres[tuple(sample)] = [sample, r] #bookkeeping for validation
            self.reference[tuple(sample)] = vs
            self.latents[tuple(sample)] = torch.concat([torch.tensor(params[inds]), torch.rand((len(inds), 1))], dim=1)
            if self.use_cuda:
                self.latents[tuple(sample)] = self.latents[tuple(sample)].cuda()

    def plot_overlap(self):
        """
        Plot of how the epsilon balls cover the parameter space.
        Used for validation only.
        """
        from matplotlib import pyplot as plt
        for k in self.reference.keys():
            mid = self.spheres[k][0]
            rad = self.spheres[k][1]
            circle = plt.Circle((mid[0],mid[1]),rad,linewidth=0.1, zorder=0, fill= False)
            plt.gca().add_patch(circle)
        plt.scatter([x[0] for x in self.params[::1000]], [x[1] for x in self.params[::1000]], s=0.1, color='k',
                    marker='.', zorder=1)
        
    def eval_generator(self,generator:nn.Module) -> (dict[str,float],bool):
        """
        Evaluates the metric for the Generator network in its current training status.
        See :func: `gan_trainer.kstest_metric.KSTestMetric.__init__` for methodological details.
        Termination is decided upon the average value of all KSTest statistics
        :param g: Generator network
        :return: dict object with current statistics and a bool indicating whether training should be stopped
        """
        stats1 = list()
        stats2 = list()
        for p in self.reference.keys(): # iterate over all epsion balls
            gen = generator(self.latents[p]).detach().cpu() # generate from latents
            gen = np.array(gen).reshape([len(self.latents[p])]) # reshape for kstest
            st = kstest(self.reference[p], gen) # compute KS Test
            stats1.append(st[0])
            stats2.append(st[1])
        s1 = sum(stats1) / len(stats1) # take average
        s2 = sum(stats2) / len(stats2) # take average
        if self.threshold != None:
            return {'kstest': s1, 'pvalue': s2}, s1 < self.threshold
        else:
            return {'kstest': s1, 'pvalue': s2}, False

    def get_statistics_names(self) -> list[str]:
        """
        :return: names of the statistics evaluated in the class. Match the keys of the dict returned by :func: `gan_trainer.kstest_metric.KSTestMetric.eval_generator`
        """
        return ['kstest','pvalue']


