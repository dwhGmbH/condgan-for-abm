import os
import pickle
from typing import Callable
import numpy as np


def create_synthetic_training_data(dataid:str,parameterspace:list[tuple[float,float]],function:Callable,N:int,seed:int=12345):
    np.random.seed(seed)
    dim = len(parameterspace)
    rands = np.random.rand(N,dim)
    lowers = np.array([x[0] for x in parameterspace])
    span = np.array([x[1] for x in parameterspace])-lowers
    parameters = rands*span+lowers
    rands2 = np.random.rand(N)
    values = np.array([function(*x,y) for x,y in zip(parameters,rands2)])
    picklefile = os.path.join('..','data','trainingSet_'+dataid+'.pickle')
    with open(picklefile,'wb') as f:
        pickle.dump((parameters,values),f)

if __name__ == '__main__':
    function = lambda a,b,X:a+np.sin(b*2*np.pi) + (X-0.5)
    parameterspace = [(-1,1),(0,1)]
    N = 2**16
    create_synthetic_training_data('synthetic_sinewave',parameterspace,function,N)





