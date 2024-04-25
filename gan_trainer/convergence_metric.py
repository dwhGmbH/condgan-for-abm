import numpy as np
from torch import nn

class ConvergenceMetric:
    def __init__(self,params:np.array, vals:np.array, threshold:float, use_cuda=False):
        self.use_cuda = use_cuda
        self.threshold = threshold

    def eval_generator(self,g:nn.Module) -> (dict[str,float],bool):
        return {},False

    def get_statistics_names(self) -> list[str]:
        return []


