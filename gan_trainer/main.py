import numpy as np
import torch

from config import CondGANConfig
from loader import TraingSetLoader
from scaler import Scaler
from trainer import CondGANTrainer

if __name__=='__main__':
    config = CondGANConfig('../configs/config_synthetic_sinewave.json')
    trainingParams, trainingValues = TraingSetLoader(config).load()
    #trainingParams = trainingParams[::100, :]
    #trainingValues = trainingValues[::100]
    print(np.size(trainingValues, 0))

    ParamScaler = Scaler(trainingParams, config.get_parameter_space())
    trainingParamsNormed = ParamScaler.downscale(trainingParams)
    ValueScaler = Scaler(trainingValues, config.get_value_space())
    trainingValuesNormed = ValueScaler.norm(ValueScaler.downscale(trainingValues))

    useCuda = torch.cuda.is_available()
    print('cuda: {}'.format(useCuda))

    trainer = CondGANTrainer(config,useCuda)
    trainer.train(trainingParamsNormed,trainingValuesNormed)