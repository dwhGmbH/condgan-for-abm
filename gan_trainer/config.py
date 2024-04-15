import json
import os
import shutil
from datetime import datetime
from enum import Enum

class LossFunctionID(Enum):
    BINARY=0,
    LOG = 1,
    WASSERSTEIN_GP = 2
    WASSERSTEIN_CLAMP = 3

class ConvergenceMetricsID(Enum):
    MOMENTS = 0,
    KSTEST = 1,
    MIXED = 2

class CondGANConfig:
    def __init__(self,filename:str):
        with open(filename,'r') as f:
            self.file_content = json.load(f)
        self.timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        resfolder = self.file_content['resultFolder']
        if not os.path.isdir(resfolder):
            os.mkdir(resfolder)
        self.resultfolder = os.path.join(resfolder,'Experiment_{}_{}'.format(self.file_content['scenario'],self.timestamp))
        if not os.path.isdir(self.resultfolder):
            os.mkdir(self.resultfolder)
        conff = os.path.split(filename)[-1]
        shutil.copyfile(filename,os.path.join(self.resultfolder,conff))

    def get_traing_data_file(self) -> str:
        return self.file_content['data']['dataFile']

    def get_resultfolder(self) -> str:
        return self.resultfolder

    def get_generator_hyperparams(self) -> dict:
        return self.file_content['generatorNetwork']

    def get_critic_hyperparams(self) -> dict:
        return self.file_content['criticNetwork']

    def get_timestamp(self):
        return self.timestamp

    def get_scenario_id(self):
        return self.file_content['scenario']

    def get_parameter_space(self):
        return self.file_content['data']['parameterSpace']

    def get_value_space(self):
        return self.file_content['data']['valueSpace']
    def get_loss_function_id(self):
        return LossFunctionID[self.file_content['learning']['lossFunction']['id']]

    def get_loss_function_hyperparams(self):
        return self.file_content['learning']['lossFunction']

    def get_learning_rate(self):
        return self.file_content['learning']['learningRate']['initial']

    def get_learning_rate_hyperparams(self):
        return self.file_content['learning']['learningRate']

    def get_batch_size(self):
        return int(self.file_content['learning']['batchSize'])

    def get_epochs(self):
        return int(self.file_content['learning']['epochs'])

    def get_snapshot_export_update_interval(self):
        return int(self.file_content['snapshotExport']['updateInterval'])

    def get_convergence_metric(self) -> ConvergenceMetricsID:
        return ConvergenceMetricsID[self.file_content['convergenceMetric']['id']]

    def get_convergence_metrics_update_interval(self):
        return int(self.file_content['convergenceMetric']['updateInterval'])

    def get_convergence_metrics_stopping_threshold(self):
        return float(self.file_content['convergenceMetric']['stoppingThreshold'])

    def get_console_log_update_interval(self):
        return int(self.file_content['consoleLog']['updateInterval'])

    def get_visualisation_update_interval(self):
        return int(self.file_content['visualisation']['updateInterval'])

    def get_views(self):
        return self.file_content['visualisation']['views']