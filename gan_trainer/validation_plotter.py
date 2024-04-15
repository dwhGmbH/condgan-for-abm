import math
import numpy as np
import torch
from matplotlib import pyplot as plt
from range_finder import RangeFinder

class ValidationPlotter:
    def __init__(self,parameters,values,metrics,views=None,use_cuda = False):
        self.metrics = metrics
        self.use_cuda = use_cuda
        if views==None:
            l = 9+len(metrics)
        else:
            l = len(views)+len(metrics)
        a = int(math.ceil(l**0.5))
        b = int(math.ceil(l/a))
        self.rows = a
        self.columns = b

        self.training_parameters = parameters
        self.training_values = values
        self.dim = np.size(self.training_parameters,1)
        self.boundary_densities = [tuple(x['parameters']) for x in views]
        self.fig = plt.figure(figsize=(16,9))
        self.plothandles = dict()
        self.latents = dict()
        i = 1
        rng = np.array(range(self.dim))
        range_finder = RangeFinder(self.training_parameters)
        self.dxs = dict()
        for key in self.boundary_densities:
            plt.subplot(self.rows,self.columns,i)
            plt.title(str(key))
            values = np.array(key)
            if None in key:
                target = 100
            else:
                target = 100
            dx, logical_indices, pts = range_finder.find_in_range(values, target)
            self.latents[key] = torch.concat(
                [torch.tensor(self.training_parameters[logical_indices, :]), torch.rand((pts, 1))], dim=1)
            Y = self.training_values[logical_indices]
            if None in key:
                dimension = key.index(None)
                X=[x[dimension] for x in self.latents[key]]
                plt.plot(X,Y, 'r.', zorder=1,markersize=2, alpha=0.3)
                pl, = plt.plot([None],[None],'b.',zorder=2,markersize=2)
                pl = [plt.gca(),pl]
            else:
                pl =[plt.gca(),Y]
            self.dxs[key] = dx
            if self.use_cuda:
                self.latents[key] = self.latents[key].cuda()
            self.plothandles[key] = pl
            i+=1

        for m in self.metrics:
            plt.subplot(self.rows, self.columns, i)
            pl, = plt.plot([None], [None])
            self.plothandles[m] = [plt.gca(),pl]
            i += 1
            plt.title(m)
            if not 'loss' in m:
                plt.yscale('log')
            plt.grid()

    def update(self,epoch,generatedX,generatedY,metricsTimeLine):
        for key in self.boundary_densities:
            if None in key:
                self.plothandles[key][1].set_xdata(generatedX[key])
                self.plothandles[key][1].set_ydata(generatedY[key])
                ax = self.plothandles[key][0]
                ax.relim()
                ax.autoscale_view()
            else:
                ax = self.plothandles[key][0]
                ax.cla()
                Y = self.plothandles[key][1]
                if len(Y)>0:
                    ax.hist(Y, color='r', zorder=1, alpha=0.3, density=True)
                if len(generatedY[key]) > 0:
                    ax.hist(generatedY[key], color='b', zorder=2, alpha=0.3, density=True)
                ax.relim()
                ax.autoscale_view()
            ax.set_title('{} (+/- {:.03f})'.format(key,self.dxs[key]))

        timelines = [metricsTimeLine[m] for m in self.metrics]
        for key,[times,values] in zip(self.metrics,timelines):
            ax = self.plothandles[key][0]
            self.plothandles[key][1].set_xdata(times)
            self.plothandles[key][1].set_ydata(values)
            ax.relim()
            ax.autoscale_view()
        plt.suptitle('epoch {}'.format(epoch))
        plt.tight_layout()

    def update_from_generator(self,epoch,generator,metricsTimeLine):
        generatedX = dict()
        generatedY = dict()
        for key in self.boundary_densities:
            latent = self.latents[key]
            for i in range(self.dim):
                if key[i]==None:
                    generatedX[key] = latent[:,i].cpu().numpy()
            generatedY[key] = [x[0].numpy() for x in generator(latent).detach().cpu()]
        self.update(epoch,generatedX,generatedY,metricsTimeLine)


