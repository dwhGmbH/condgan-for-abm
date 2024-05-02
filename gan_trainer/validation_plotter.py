import math
import numpy as np
import torch
from torch import nn
from matplotlib import pyplot as plt
from range_finder import RangeFinder


class BoundaryDensityPlot:
    """
    Plots and updates a boundary density view of the training process
    """

    def __init__(self, view_specification: dict, range_finder: RangeFinder, training_parameters: np.array,
                 training_values: np.array, use_cuda: bool = False):
        """
        Initializes a scatter or histogram plot for the boundary density specified by the "view_specification"
        :param view_specification: See :func:`gan_trainer.validation_plotter.BoundaryDensityPlot.__init__` for details
        :param range_finder: RangeFinder object to compute the epsilon ball
        :param training_parameters: parameter vectors of the training data
        :param training_values: values of the training data
        :param use_cuda: whether the update can use GPU
        """
        self.ax = plt.gca()  # important - save current axis
        assert 'parameters' in view_specification.keys(), 'View specification must containt a "parameters" key'
        assert 'radius' in view_specification.keys() or 'nof_points' in view_specification.keys(), 'View specification must containt either "radius" or "nof_points" key'
        self.parameters = view_specification['parameters']
        self.dim = len(self.parameters)
        self.use_cuda = use_cuda
        if 'radius' in view_specification.keys():
            self.radius = float(view_specification['radius'])
            indices, self.nof_points = range_finder.find_in_radius(np.array(self.parameters), self.radius)
            if self.nof_points == 0:
                self.nof_points = 100
                indices, self.radius = range_finder.find_nearest_s(np.array(self.parameters), self.nof_points)
        else:
            self.nof_points = int(view_specification['nof_points'])
            indices, self.radius = range_finder.find_nearest_s(np.array(self.parameters), self.nof_points)
        self.Udata = training_parameters[indices, :]
        self.Ydata = training_values[indices]
        self.latents = torch.concat(
            [torch.tensor(self.Udata), torch.rand((len(indices), 1))], dim=1)
        if self.use_cuda:
            self.latents = self.latents.cuda()
        if None in self.parameters:
            k = [x for x in self.parameters if x == None]
            assert len(k) == 1, 'Only one None argument allowed per view specification'
            self.viewDim = self.parameters.index(None)
            self.isScatter = True
            X = [x[self.viewDim] for x in self.Udata]
            Y = self.Ydata
            self.plothandleData, = plt.plot(X, Y, 'r.', zorder=1, markersize=2,
                                            alpha=0.3)  # scatterplot for original data
            self.plothandleLatent, = plt.plot(X, [0 for x in X], 'b.', zorder=2,
                                              markersize=2)  # empty scatterplot for synthetic data
        else:
            self.isScatter = False
        plt.title('{:}, (r:{:.02f}, pts:{:.0f})'.format(self.parameters, self.radius, self.nof_points))

    def update(self, generator: nn.Module):
        """
        Updates the plot
        :param generator: current status of the generator network
        """
        generated = [x[0].numpy() for x in generator(self.latents).detach().cpu()]
        if self.isScatter:
            self.plothandleLatent.set_ydata(generated)
            self.ax.relim()
            self.ax.autoscale_view()
        else:
            self.ax.cla()
            self.ax.hist(self.Ydata, color='r', zorder=1, alpha=0.3, density=True)
            self.ax.hist(generated, color='b', zorder=2, alpha=0.3, density=True)
            self.ax.relim()
            self.ax.autoscale_view()
            plt.title('{:}, (r:{:.02f}, pts:{:.0f})'.format(self.parameters, self.radius, self.nof_points))


class ConvergenceMetricsPlot:
    """
    Plots and updates a convergence metric visualization of the training process
    """

    def __init__(self, metricsName: str):
        """
        Initializes a plot for convergence tracking.
        Automatically uses log-scale for all metrics which do not include the term "loss"
        :param metricsName: name of the metric to display
        """
        self.ax = plt.gca()
        self.name = metricsName
        self.plothandle1, = plt.plot([None], [None], color='k')  # empty plot for metric
        self.plothandle2, = plt.plot([None], [None], color='r', linewidth=2)  # empty plot for moving average
        if 'loss' in metricsName:
            plt.yscale('log')  # log scale is usually the better option for something that "converges" somewhere
        plt.grid()
        plt.title(metricsName)

    def get_exp_moving_average(self, times: list[int], values: list, memory: int = 50) -> list:
        """
        Helper routine to compute the moving average in epochs
        :param times: timesteps in epochs (integers)
        :param values: corresponding values
        :param memory: memory of the exponential moving average in epochs
        :return: list with equal format and length as "values"
        """
        expma = [values[0]]
        for i in range(1, len(values)):
            dt = times[i] - times[i - 1]
            fac = min(dt / memory, 1)
            expma.append(fac * values[i] + (1 - fac) * expma[i - 1])
        return expma

    def update(self, metric_timelines: dict[str, tuple[list, list]]):
        """
        Updates the plot from a metric_timelines dictionary, as it is internally used by the gan trainer
        :param metric_timelines: dict object with timeseries of the metric
        """
        times = metric_timelines[self.name][0]
        values = metric_timelines[self.name][1]
        self.plothandle1.set_xdata(times)
        self.plothandle1.set_ydata(values)
        expma = self.get_exp_moving_average(times, values, memory=50)
        self.plothandle2.set_xdata(times)
        self.plothandle2.set_ydata(expma)
        self.ax.relim()
        self.ax.autoscale_view()


class ValidationPlotter:
    """
    Class to plot important features of the training process of a congGAN
    """

    def __init__(self, parameters: np.array, values: np.array, metrics: list[str], views: list[dict] = None,
                 use_cuda: bool = False):
        """
        The validation plotter is used to visually track the convergence process of the congGAN trainer
        The visualisation it creates cnsists of multiple view panels, which can be split into two classes.

        The first class are visualizations of certain boundary densities of training data and the generator output.
        The two are compared visually using histograms or scatterplots.
        The corresponding boundary-density is specified in the entries of the optional "views" variable.
        Each entry "view" in the list results defines a plot in the following way:

        The corresponding dict object must have a "parameters" entry, which is a list of numbers matching in length with the dimension of the parameter-space, and a scalar "radius" or "nof_points" entry.
        In the following, we define S = (U,Y) as the overall parameter-vector set and value set given in the training data.
        We distinguish four cases:

        - C1: the "parameters" entry is a full vector of numbers and the "radius" entry is defined as a postive float
          In this case, we use c:=view["parameters"] as the center of an epsilon ball with radius "radius",
          i.e. (U',Y') := {(y,u) in (Y,U): ||u-c||_2<radius}
          The plot futhermore compares a histogram of the original data (i.e. hist(Y')) with a histogram of the generated data using U' as latents (i.e. hist(G(U',X)) with X being a U(0,1) noise)

        - C2: the "parameters" entry is a full vector of numbers and a positive integer "nof_points" is defined instead of "radius"
          In this case, we use c:=view["parameters"] as the center of an epsilon ball whereas the radius is defined so that the epsilon ball contains precisely "nof_points" points.
          With the computed radius, the view is analogous to C1

        - C3: the "parameters" entry contains one "None" entry and the "radius" entry is defined as a postive float
          Let i refer to the index of the None entry. Now, instead of view["parameters"], we use c=view["parameters"][:i,i+1:] as the center of the epsilon ball and consider.
          (U',Y') := {(y,u) in (Y,U): ||u[:i,i+1:]-c||_2<radius}
          That means the epslion ball's dimension is by one smaller than in C1 and C2
          The view shows the i-th dimension of U' over the corresponding values Y' as a scatter plot and compares it with a scatter plot of the correspondent generated values (i.e. G(U',X) with X being a U(0,1) noise)

        - C4: the "parameters" entry contains one "None" entry and a positive integer "nof_points" is defined instead of "radius"
          C4 combines the view of C3 with the ideas of C2

        :param parameters: training parameter vectors
        :param values: training values
        :param metrics: list of convergence metrics to plot
        :param views: list of views according to the specification
        :param use_cuda: whether to use a GPU for computing the statistics
        """
        if views == None:
            views = []
        l = len(views) + len(metrics)
        a = int(math.ceil(l ** 0.5))
        b = int(math.ceil(l / a))
        self.rows = a
        self.columns = b

        self.plots = list()
        self.fig = plt.figure(figsize=(16, 9))
        i = 1
        range_finder = RangeFinder(parameters)
        for view_specification in views:
            plt.subplot(self.rows, self.columns, i)
            i += 1
            self.plots.append(
                BoundaryDensityPlot(view_specification, range_finder, parameters, values, use_cuda=use_cuda))
        for m in metrics:
            plt.subplot(self.rows, self.columns, i)
            i += 1
            self.plots.append(ConvergenceMetricsPlot(m))

    def update_from_generator(self, epoch: int, generator: nn.Module, metric_timelines: dict[str, tuple[list, list]]):
        """
        Updates the views using the current status of the generator network
        :param epoch: current training epoch (only for the sup-title)
        :param generator: generator network
        :param metrics_time_line: dict object containing the times and values of the convergence metrics
        """
        for plot in self.plots:
            if isinstance(plot, BoundaryDensityPlot):
                plot.update(generator)
            elif isinstance(plot, ConvergenceMetricsPlot):
                plot.update(metric_timelines)
        plt.suptitle('epoch {}'.format(epoch))
        plt.tight_layout()
