import numpy as np
import torch

from config import CondGANConfig
from loader import TraingSetLoader
from scaler import Scaler
from trainer import CondGANTrainer
import sys
import os

if __name__=='__main__':
    if len(sys.argv)>1:
        if os.path.isfile(sys.argv[1]):
            filenames =[sys.argv[1]]
        elif os.path.isdir(sys.argv[1]):
            filenames = [sys.argv[1]+'/'+x for x in os.listdir(sys.argv[1]) if x.endswith('.json')]
        else:
            raise RuntimeError('Cannot find path or file specified')
    else:
        filenames = ['../configs/config_weibull_2e-7.json']

    useCuda = torch.cuda.is_available()
    print('cuda: {}'.format(useCuda))

    for filename in filenames:
        print(f'run {filename}')
        config = CondGANConfig(filename)
        trainingParams, trainingValues = TraingSetLoader(config).load()
        #trainingParams = trainingParams[::100, :]
        #trainingValues = trainingValues[::100]
        print(np.size(trainingValues, 0))

        ParamScaler = Scaler(trainingParams, config.get_parameter_space())
        trainingParamsNormed = ParamScaler.downscale(trainingParams)
        ValueScaler = Scaler(trainingValues, config.get_value_space())
        trainingValuesNormed = ValueScaler.norm(ValueScaler.downscale(trainingValues))

        trainer = CondGANTrainer(config,useCuda)
        trainer.train(trainingParamsNormed,trainingValuesNormed)