from torch import nn

from kstest_statistic import KSTestStatistic
from moments_statistic import MomentsStatistic
from validation_statistic import ValidationStatistic


class MixedStatistic(ValidationStatistic):
    def __init__(self, params, vals, threshold=None, points=1000, momentsCount=4, momentsWeights=None,
                 use_cuda=False):
        super().__init__(params, vals, threshold, use_cuda)
        self.kstestStatistic = KSTestStatistic(params, vals, threshold=threshold,points=points,use_cuda=use_cuda)
        self.momentsStatistic = MomentsStatistic(params, vals, threshold=None, momentsCount=momentsCount, momentsWeights=momentsWeights,use_cuda=use_cuda)

    def eval_generator(self, g: nn.Module) -> (dict[str,float],bool):
        stats, stop  = self.momentsStatistic.eval_generator(g)
        stats2, stop = self.kstestStatistic.eval_generator(g)
        stats.update(stats2)
        return stats, stop

    def get_statistics_names(self) -> list[str]:
        lst = self.kstestStatistic.get_statistics_names()
        lst.extend(self.momentsStatistic.get_statistics_names())
        return lst