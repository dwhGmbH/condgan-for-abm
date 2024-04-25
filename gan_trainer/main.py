import numpy as np
import torch

from config import CondGANConfig
from loader import TraingSetLoader
from scaler import Scaler
from trainer import CondGANTrainer
import sys
import os

if __name__=='__main__':
    """
    This script runs one or a series of congGAN training experiments.
    If the runtime argument points towards a JSON file, this script executes one training experiment with the corresponding configuration.
    If it points towards a folder with JSON files in it, the script executes them all sequentially.
    If no runtime argument is given, an default config is used.
    """
    if len(sys.argv)>1:
        if os.path.isfile(sys.argv[1]):
            filenames =[sys.argv[1]]
        elif os.path.isdir(sys.argv[1]):
            filenames = [sys.argv[1]+'/'+x for x in os.listdir(sys.argv[1]) if x.endswith('.json')]
        else:
            raise RuntimeError('Cannot find path or file specified')
    else:
        filenames = ['../configs/config_weibull_2e-7.json']

    useCuda = torch.cuda.is_available() #use cuda if available
    print(f'cuda: {useCuda}')

    for filename in filenames:
        print(f'run {filename}')
        try:
            config = CondGANConfig(filename) #create configuration instance
            trainingParams, trainingValues = TraingSetLoader(config).load() #load training data
            #trainingParams = trainingParams[::100, :]
            #trainingValues = trainingValues[::100]
            print(np.size(trainingValues, 0))

            ParamScaler = Scaler(trainingParams, config.get_parameter_space())
            trainingParamsNormed = ParamScaler.downscale(trainingParams) #downscale the training parameters
            ValueScaler = Scaler(trainingValues, config.get_value_space())
            trainingValuesNormed = ValueScaler.clamp(ValueScaler.downscale(trainingValues)) #downscale and clamp the training values
            trainer = CondGANTrainer(config,useCuda) #initialise trainer
            trainer.train(trainingParamsNormed,trainingValuesNormed) #run training
        except:
            # catch to continue with next config even if an error ocurred
            print(f'error ocurred for {filename}')