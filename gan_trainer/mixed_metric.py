from torch import nn

from kstest_metric import KSTestMetric
from moments_metric import MomentsMetric
from convergence_metric import ConvergenceMetric


class MixedMetric(ConvergenceMetric):
    """
    Mixed metric combining the two classes :class:`gan_trainer.KSTestMetric` and :class:`gan_trainer.MomentsMetric`.
    The KSTestMetric is used for evaluating whether the training process can be stopped.
    Technically, the two classes are simply wrapped into one.
    """
    def __init__(self, params, vals, threshold=None, m=1000, k=4, mu=None,use_cuda=False):
        super().__init__(params, vals, threshold, use_cuda)
        self.kstestStatistic = KSTestMetric(params, vals, threshold=threshold, m=m, use_cuda=use_cuda)
        self.momentsStatistic = MomentsMetric(params, vals, threshold=None, k=k, mu=mu, use_cuda=use_cuda)

    def eval_generator(self, generator: nn.Module) -> (dict[str,float], bool):
        """
        Evaluates the metrics for the Generator network in its current training status.
        Essentially calls the corresponding methods of the two wrapped classes.
        Termination is decided based on the KSTest statistic.
        :param generator: Generator network
        :return: dict object with current statistics and a bool indicating whether training should be stopped
        """
        stats, stop  = self.momentsStatistic.eval_generator(generator)
        stats2, stop = self.kstestStatistic.eval_generator(generator)
        stats.update(stats2)
        return stats, stop

    def get_statistics_names(self) -> list[str]:
        """
        :return: names of the statistics evaluated in the class. Match the keys of the dict returned by :func: `gan_trainer.mixed_metric.eval_generator`
        """
        lst = self.kstestStatistic.get_statistics_names()
        lst.extend(self.momentsStatistic.get_statistics_names())
        return lst