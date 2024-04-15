import pickle
import numpy as np
from config import CondGANConfig


class TraingSetLoader:
    def __init__(self,config:CondGANConfig):
        self.config = config

    def load(self):
        filenamePickle = self.config.get_traing_data_file()
        with open(filenamePickle, 'rb') as f:
            params, values = pickle.load(f)
        return [np.array(params), np.array(values)]