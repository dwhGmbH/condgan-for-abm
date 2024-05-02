# condgan-for-abm
Repository with various classes and scripts for training conditional GANs for use in agent-based simulation models.
We refer to the included `LICENSE` file for usage and sharing details.

## How to get this stuff running?
First of all, the code was developed under Python 3.12 and uses the packages specified in `requirements.txt`.

After successful installation, call the `main.py` script with your Python interpreter using the repository root as working directory.

### What happens then?
The script will start training of a conditional GAN as specified in `configs/config_weibull.json`. This JSON file contains various information related to how the training process is specified (hyperparameters, loss function, ...), which training data will be used and where outcomes should be saved.

1. As first step, the training data will be loaded from the specified `.pickle` file. The default config will load ~4M datasets from `data/trainingSet_synthetic_weibull.pickle`. Thsi might take a few moments.
2. Thereafter, a Trainer instance is created and the training process is started for, by default 3000 training epochs. It automatically detects whether a GPU (Cuda) is available for accelerating the process or if training must be done in your CPU.
3. The Trainer will create a new folder `results/Experiment_reaction-time_weibull<xxx>`, whereas xxx refers to a timestamp, will be created which will be the folder for the results carried out in the training process. As first step, the method copy-pastes the configuration file into the folder to increase reproducibility of the results.
4. While training a console messge is printed every epoch and a snapshot output will be produced and carried out every 10 epochs:
   - This includes a constantly updated picture `visualisation.png` showing details of the training process. The upper 8 subplots show boundary densities of data and GAN (Generator) output for different points in the parameterspace. The close the red and blue scatter plots / histograms align, the better the network is trained. Thereafter, two plots show the loss functions of critic and generator including an exponential moving average (to smooth out fluctuations). Finally, the last six plots display different convergence statistics.
   - For every snapshot a new folder will be created under `results/Experiment_reaction-time_weibull<xxx>/snapshots/<yyy>`, whereas yyy corresponds to the current epoch.
   - Inside this folder, the user will find exported versions (.torch and .state) of the critic and the generator network and a `metric.json` file. The latter contains the sourcedata for the visualised metrics.
5. The code terminates either if 3000 epochs have passed (standard case) or if the KSTest metric drops below 0.00001.

## How is the repository structured?
- `CondGANTrainer` is the key class of the repository. Its function `train` manages the whole training process of the networks and about 90% of all classes and functions in the repository are developed to support hereby.
- The trainer is initialized with a `CondGANConfig` instance, which wraps the contents of the `xxx_config.json` file into a class with getter functions.
- To run the `train` method, trainig data in the form of a list of parameter vectors and a list of output values has to be loaded first. The `TrainigSetLoader` class is defined precisely for this purpose. It loads a training set which was properly saved to a `pickle` file.
- Thereafter, the `Scaler` class is applied to down scale the training data to [0,1]
- To get a better understanding how the training data is structured, the helper-routine `create_synthetic_training_data` offers a method to create your own synthetic training data using an analytically known stochastic process.
- After calling the `train` method the `CondGANTrainer` initialises an "empty" generator and critic network as instances of `GenericMLP` using the method  `create_mlps_from_config` both found in `networks.py`.  Note, tha the script also offers methods to load exported networks into Python.
- Every few epochs the outcomes statistics (convergence metrics) and the visualization are updated. The `ConvergenceMetric` and the `ValidationPlotter` class take care of that. The project currently supports three different convergence metrics: `MomentsMetric`,`KSTestMetric` and `MixedMetric`.
- Both, plotting and metrics update require a logic to find points in a certain range. The `RangeFinder` wraps an efficient KDTree structure from `SciPy`.

## Customizing?
The repository content offers a variety of options for customization to run own condGAN training processes. Most of the customization can be done by modifying or creating a `xxxx_config.json` file. The default file `configs/config_weibull.json` includes comments using the `#` sign as prefix. Beyond, the mentioned `create_synthetic_training_data` function shows how own training data can be included into the framework.
