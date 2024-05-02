import numpy as np
from scipy import spatial


class RangeFinder:
    """
    This class is an auxiliary to quickly find points within a given range.
    Capable of neglecting dimensions
    """

    def __init__(self, data: np.array):
        """
        Creates an empty RangeFinder object with given data.
        Internally uses a variety of KDTrees to find points within a given range
        :param data: numpy array of points
        """
        self.data = data
        self.kdtrees = dict()
        self.kdtree = spatial.cKDTree(self.data)

    def _find_or_create_tree(self, center: np.array) -> (np.array, spatial.cKDTree):
        """
        Dependent on the number and position of None entries in x, this function creates or returns a KDTree object from the initial data which neglects the dimensions with None.
        :param x: center point of a query
        :return: cleaned center point with the None-dimensions left out, corresponding KDTree of the initial data ith the None-dimensions left out
        """
        key = tuple(center != None)
        center_clean = center[center != None]
        if not key in self.kdtrees.keys():
            self.kdtrees[key] = spatial.cKDTree(self.data[:, center != None])
        return center_clean, self.kdtrees[key]

    def find_in_radius(self, center: np.array, radius: float) -> (np.array, int):
        """
        Returns coordinates of all points in the orginal data which are within a given radius around the passed-on center.
        Dimensions for which center=None are left out of the query.
        :param center: center of the epsilon ball
        :param radius: radius of the epsilon ball
        :return: indices of the found points, number of found points within radius
        """
        center_clean, tree = self._find_or_create_tree(center)
        indices = tree.query_ball_point(x=center_clean, r=radius)
        return indices, len(indices)

    def find_nearest_s(self, center: np.array, s: int) -> (np.array, float):
        """
        Returns nearest s coordinates of all points in the orginal data to the passed-on center.
        Dimensions for which center=None are left out of the query.
        :param center: center of the epsilon ball
        :param s: number of coordinates to return
        :return: indices of the found points, radius of the epsilon ball
        """
        center_clean, tree = self._find_or_create_tree(center)
        dists, indices = tree.query(x=center_clean, k=s)
        radius = max(dists)
        return indices, radius
