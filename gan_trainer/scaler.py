import numpy as np


class Scaler:
    """
    Class to scale parameter and values up and down for normalization
    """
    def __init__(self,space=None,array:np.array=None):
        """
        Constructor for the scaler object.
        Can either be initialised with an array, in which case the value space will be detected from the max and min of the array entries,
        or by stating the value space directly.

        :param space: must be either a numpy array of (min,max) tuples or a single (min,max) tuple
        :param array: must be a numpy array
        """
        if isinstance(space, tuple):
            self.left = space[0]
            self.right = space[1]
        elif isinstance(space, np.ndarray):
            self.left = space[:, 0]
            self.right = space[:, 1]
        elif isinstance(space,list):
            space = np.array(space)
            self.left = space[:, 0]
            self.right = space[:, 1]
        elif isinstance(array,np.ndarray):
            if array.ndim ==1:
                self.left = np.min(array)
                self.right = np.max(array)
            elif array.ndim==2:
                self.left = np.min(array,0)
                self.right = np.max(array,0)
        else:
            raise TypeError(f'inputs {space} or {array} are not an accepted format')
        self.span = self.right-self.left
    def upscale(self,normed:np.array) -> np.array:
        """
        Upscale a normed value to the original space
        :param normed: normed value, should be between 0 and 1
        :return: upscaled value
        """
        return (normed*self.span)+self.left

    def downscale(self,data:np.array) -> np.array:
        """
        Downscale a given value within the original space to a value betwen 0 and 1
        :param data: original value, should be within the specified parameter space
        :return: normed value
        """
        return (data-self.left)/self.span
        
    def clamp(self, data:np.array,minimum:float=0.0,maximum:float=1.0) -> np.array:
        """
        Clamps a value to [min,max]
        :param data: value
        :param minimum: lower clamp bound (0.0 by default)
        :param maximum: upper clamp bound (1.0 by default)
        :return: clamped value
        """
        return np.maximum(np.minimum(data,maximum),minimum)