import pickle
import numpy as np
from condgan_config import CondGANConfig


class TraingSetLoader:
    """
    Loads a training set from a given pickle file
    """

    def __init__(self, config: CondGANConfig = None):
        """
        Constructor of the training data loader.
        If the config is None, only the load_from_file method can be used.
        :param config: configuration of the experiment to get the correct filename.
        """
        self.config = config

    def load_from_file(self, filename_pickle: str) -> (np.array, np.array):
        """
        Loads a training set from a given pickle file
        :param filename_pickle: full path to the pickle file
        :return: array of training parameter vectors and corresponding array of training values
        """
        with open(filename_pickle, 'rb') as f:
            params, values = pickle.load(f)
        return np.array(params), np.array(values)

    def load(self) -> (np.array, np.array):
        """
        Loads a training set from the file specified in the config instance.
        :return: array of training parameter vectors and corresponding array of training values
        """
        assert self.config is not None, AssertionError(
            'need to construct the class from a config instance to use this method or use "load_from_file" instead')
        filename_pickle = self.config.get_traing_data_file()
        return self.load_from_file(filename_pickle)
