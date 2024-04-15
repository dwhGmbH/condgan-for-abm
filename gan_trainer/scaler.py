import numpy as np


class Scaler:
    def __init__(self,values,space=None):
        arr = np.array(values)
        if space==None:
            if arr.ndim ==1:
                self.left = np.min(values )
                self.right = np.max(values )
            elif arr.ndim==2:
                self.left = np.min(values,0)
                self.right = np.max(values,0)
        else:
            space =np.array(space)
            if arr.ndim == 1:
                self.left=space[0]
                self.right = space[1]
            else:
                self.left = space[:,0]
                self.right = space[:,1]
        self.span = self.right-self.left
    def upscale(self,normed):
        return (normed*self.span)+self.left

    def downscale(self,data):
        return (data-self.left)/self.span
        
    def norm(self,data):
        return np.maximum(np.minimum(data,1.0),0.0)