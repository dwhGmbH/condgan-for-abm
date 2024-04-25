import json
import os
import shutil
from datetime import datetime
from enum import Enum

class LossFunctionID(Enum):
    """
    Supported types of loss functions
    """
    BINARY=0,
    LOG = 1,
    WASSERSTEIN_GP = 2
    WASSERSTEIN_CLAMP = 3

class ConvergenceMetricsID(Enum):
    """
    Supported types of outcome metrics
    """
    MOMENTS = 0,
    KSTEST = 1,
    MIXED = 2

class CondGANConfig:
    """
    Configuration for a conditional GAN training process
    """
    def __init__(self,filename:str):
        """
        Loads a configuration file in JSON format.
        The required and optional fields can be identified from the examples in the "configs" folder.
        :param filename: full path to the configuration file
        """
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
        """
        Returns the filepath to the training data.
        Must be a reference to a pickle file.
        :return: Full path to the training data file
        """
        return self.file_content['data']['dataFile']

    def get_resultfolder(self) -> str:
        """
        Returns the path to the folder in which the trainer should store all outcomes related to the training process.
        A new folder with a timestamp will be created there. So writing permissions are required.
        :return: Full path to the result folder
        """
        return self.resultfolder

    def get_generator_hyperparams(self) -> dict:
        """
        Returns the hyperparameters of the Generator.
        The dict object must have the fields "sequence", "dropout" and "finishSigmoid":
        - sequence: list of integers representing the nodes of the hidden layers of the network
        - dropout: float, specifying whether and how much dropout is applied between the layers
        - finishSigmoid: bool, specifying if the output of the final layer is additonally put through a sigmoid function or not
        :return: hyperparameters of the Generator
        """
        return self.file_content['generatorNetwork']

    def get_critic_hyperparams(self) -> dict:
        """
        Returns the hyperparameters of the Critic.
        The dict object must have the fields "sequence", "dropout" and "finishSigmoid":
        - sequence: list of integers representing the nodes of the hidden layers of the network
        - dropout: float, specifying whether and how much dropout is applied between the layers
        - finishSigmoid: bool, specifying if the output of the final layer is additonally put through a sigmoid function or not
        :return: hyperparameters of the Critic
        """
        return self.file_content['criticNetwork']

    def get_timestamp(self) -> str:
        """
        :return: stamp for the time when the experiment was started
        """
        return self.timestamp

    def get_scenario_id(self):
        """
        :return: name of the scenario as specified in the configuration file
        """
        return self.file_content['scenario']

    def get_parameter_space(self) -> list[tuple[float,float]]:
        """
        returns a list of tuples. The length of the list matches the length of the input parameter vector.
        The tuple entries refer to the lower and upper bound of the parameter.
        Used for normalization of the input.
        :return: parameter-space of the input vector
        """
        return [(x[0],x[1]) for x in self.file_content['data']['parameterSpace']]

    def get_value_space(self) -> tuple[float,float]:
        """
        The tuple entries refer to the lower and upper bound of the value space
        Used for normalization of the training data and de-normalization of the output.
        :return: value-space of the output
        """
        return tuple(self.file_content['data']['valueSpace'])

    def get_loss_function_id(self) -> LossFunctionID:
        """
        :return: id of the loss function specified in the configuration file
        """
        return LossFunctionID[self.file_content['learning']['lossFunction']['id']]

    def get_loss_function_hyperparams(self) -> dict:
        """
        returns potential hyperparameters of the loss function as dict object.
        Wasserstein loss must either have a "clampThreshold" or a "gradientPenaltyFactor"
        :return: hyperparams of the loss function
        """
        return self.file_content['learning']['lossFunction']

    def get_learning_rate(self) -> float:
        """
        :return: learning rate for the training process
        """
        return self.file_content['learning']['learningRate']['initial']

    def get_learning_rate_hyperparams(self) -> dict:
        """
        Optionally returns parameters for continuous learning rate reduction.
        If specified, the learning rate is rediced every "reductionInterval"-epochs by a fatcor of "reductionFactor" up to a certain "minimum"
        :return: hyperparameters of the learning process
        """
        return self.file_content['learning']['learningRate']

    def get_batch_size(self) -> int:
        """
        :return: batch size for training process
        """
        return int(self.file_content['learning']['batchSize'])

    def get_epochs(self) -> int:
        """
        :return: number of training epochs
        """
        return int(self.file_content['learning']['epochs'])

    def get_snapshot_export_update_interval(self) -> int:
        """
        returns how often a full snapshot of the trainin process (including network params and statistics) swhould be written to disk (in epochs)
        :return: interval in epochs
        """
        return int(self.file_content['snapshotExport']['updateInterval'])

    def get_convergence_metric(self) -> ConvergenceMetricsID:
        """
        :return: ID of the convergence metric used to track the training process
        """
        return ConvergenceMetricsID[self.file_content['convergenceMetric']['id']]

    def get_convergence_metrics_update_interval(self) -> int:
        """
        returns how often the convergence metrics should be updated (in epochs)
        :return: interval in epochs
        """
        return int(self.file_content['convergenceMetric']['updateInterval'])

    def get_convergence_metrics_stopping_threshold(self) -> float:
        """
        The trainig process stops as soon as the convergence metric returns a value smaller than this.
        If stopping treshold is set to zero, the training always stops after the epochs specified in :func: `gan_trainer.config.get_epochs`
        :return: stopping treshold value
        """
        return float(self.file_content['convergenceMetric']['stoppingThreshold'])

    def get_console_log_update_interval(self) -> int:
        """
        returns how often a statusupdate should be printed to the console (in epochs)
        :return: interval in epochs
        """
        return int(self.file_content['consoleLog']['updateInterval'])

    def get_visualisation_update_interval(self) -> int:
        """
        returns how often the visualization of the training process should be updated (in epochs)
        :return: interval in epochs
        """
        return int(self.file_content['visualisation']['updateInterval'])

    def get_views(self) -> list[dict]:
        """
        Specifies for which parts of the parameters space are displayed in the visualization of the training process.
        Each entry in the list defines a plot via a dict. The dict have the fields "parameters" and "radius"
        :return: specifies the upper part of the visualization of the training process
        """
        return self.file_content['visualisation']['views']