import numpy as np

class RangeFinder:
    def __init__(self,data:np.array):
        self.data = data
        self.data_points = np.size(data,0)
        self.dim = np.size(data,1)

    def find_in_radius(self,center:np.array,dx:float):
        checkInds = center!=None
        left = center[checkInds] - dx
        right = center[checkInds] + dx
        logicInds = np.logical_and(np.all(self.data[:,checkInds] >= left, axis=1),np.all(self.data[:,checkInds] <= right, axis=1))
        pts = np.count_nonzero(logicInds)
        return logicInds, pts

    def find_in_range(self,center:np.array,target:int):
        checkInds = center!=None

        dxA = 0.0
        dxB = 1.0
        for i in range(20):
            dxC = (dxA+dxB)/2.0
            left = center[checkInds] - dxC
            right = center[checkInds] + dxC
            logicInds = np.logical_and(np.all(self.data[:,checkInds] >= left, axis=1),np.all(self.data[:,checkInds] <= right, axis=1))
            pts = np.count_nonzero(logicInds)
            if pts==target:
                break
            elif pts>target:
                dxB = dxC
            else:
                dxA = dxC
        if pts==0: #make sure to always return something positive
            dxC = dxB
            left = center[checkInds] - dxC
            right = center[checkInds] + dxC
            logicInds = np.logical_and(np.all(self.data[:, checkInds] >= left, axis=1),
                                       np.all(self.data[:, checkInds] <= right, axis=1))
            pts = np.count_nonzero(logicInds)
        return dxC, logicInds, pts
