from datetime import datetime
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.autograd import Variable
import os
import json

from config import CondGANConfig, LossFunctionID, ConvergenceMetricsID
from kstest_metric import KSTestMetric
from mixed_metric import MixedMetric
from moments_metric import MomentsMetric
from networks import create_mlps_from_config
from validation_plotter import ValidationPlotter


class CondGANTrainer():
    """
    Class to train a condGAN model
    """

    def __init__(self, config: CondGANConfig, use_cuda=False):
        """
        Constructor for a condGAN trainer
        :param config: configuration for the training process.
        :param use_cuda: boolean to decide if training should/can happen on GPU or not
        """
        self.config = config
        assert config.get_critic_hyperparams()['sequence'][0] == config.get_generator_hyperparams()['sequence'][
            0], 'input layers of critic and generator must match'
        self.G, self.C = create_mlps_from_config(self.config)  # initialise networks
        self.use_cuda = use_cuda
        if self.use_cuda:
            self.G.cuda()  # transfer to cuda
            self.C.cuda()  # transfer to cuda
        self.G_opt = torch.optim.Adam(self.G.parameters(), lr=self.config.get_learning_rate())  # initialise optimiser
        self.C_opt = torch.optim.Adam(self.C.parameters(), lr=self.config.get_learning_rate())  # initialise optimiser

        self.lossfunctionID = self.config.get_loss_function_id()
        if self.lossfunctionID == LossFunctionID.BINARY:  # switch case over available loss functions
            self.loss_generator = self.loss_generator_base
            self.loss_critic = self.loss_critic_base
            self.loss_fct_binary = nn.BCELoss()
        elif self.lossfunctionID == LossFunctionID.LOG:
            self.loss_generator = self.loss_generator_log
            self.loss_critic = self.loss_critic_log
        elif self.lossfunctionID == LossFunctionID.WASSERSTEIN_GP:
            self.loss_generator = self.loss_generator_wasserstein
            self.loss_critic = self.loss_critic_wasserstein_gp
            self.wassersteinGPFactor = self.config.get_loss_function_hyperparams()['gradientPenaltyFactor']
        else:
            self.loss_generator = self.loss_generator_wasserstein
            self.loss_critic = self.loss_critic_wasserstein_wc
            self.wassersteinClamp = self.config.get_loss_function_hyperparams()['clampThreshold']

        self._lossGeneratorName = 'loss Generator'  # name of the validation metric which is printed out
        self._lossCriticName = 'loss Critic'
        self._ticks = list()

    def train(self, parameters: np.array, values: np.array):
        """
        Starts training with given training data i.e. parameter vectors and corresponding values.
        :param parameters: list of input parameter vectors
        :param values: list of reference values to learn from. Length must match with number of rows in parameters
        """
        assert parameters.shape[0] == len(
            values), f'dimension mismatch between training parameters ({parameters.shape[0]}) and values ({len(values)})'
        self.trainingParameters = np.array(parameters, dtype=np.float32)
        self.trainingValues = np.array(values, dtype=np.float32)
        self.inputDimension = np.size(parameters, 1)
        assert self.inputDimension == self.config.get_critic_hyperparams()['sequence'][
            0] - 1, f'dimension mismatch between training parameters ({parameters.shape[0]}) and input layer ({self.config.get_critic_hyperparams()['sequence'][0]}), whereas the latter should be by one larger than the prior.'

        dataCount = np.size(parameters, 0)
        train_data = torch.concat(
            [torch.tensor(self.trainingParameters), torch.tensor(self.trainingValues).view(dataCount, 1)], 1)
        train_labels = torch.zeros(dataCount)
        train_set = [(x, y) for x, y in zip(train_data, train_labels)]
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=self.config.get_batch_size(), shuffle=True
        )  # object to draw samples for training

        self.metrics = [self._lossGeneratorName,
                        self._lossCriticName]  # specify tracked metrics. Always track loss values
        if self.config.get_convergence_metrics_update_interval() > 0:
            self.convergenceMetricID = self.config.get_convergence_metric()
            if self.convergenceMetricID == ConvergenceMetricsID.MOMENTS:  # switch case over available convergence metrics
                self.convergenceMetric = MomentsMetric(self.trainingParameters, self.trainingValues,
                                                       threshold=self.config.get_convergence_metrics_stopping_threshold(),
                                                       use_cuda=self.use_cuda)
            elif self.convergenceMetricID == ConvergenceMetricsID.KSTEST:
                self.convergenceMetric = KSTestMetric(self.trainingParameters, self.trainingValues,
                                                      threshold=self.config.get_convergence_metrics_stopping_threshold(),
                                                      use_cuda=self.use_cuda)
            elif self.convergenceMetricID == ConvergenceMetricsID.MIXED:
                self.convergenceMetric = MixedMetric(self.trainingParameters, self.trainingValues,
                                                     threshold=self.config.get_convergence_metrics_stopping_threshold(),
                                                     use_cuda=self.use_cuda)
            self.metrics.extend(self.convergenceMetric.get_statistics_names())  # add names to track
        self.metric_values = {x: 0 for x in self.metrics}  # current values of metrics
        self.metric_timelines = {x: [list(), list()] for x in self.metrics}  # "historic" values of metrics
        if self.config.get_visualisation_update_interval() > 0:  # create a plotter object
            self.plotHandler = ValidationPlotter(self.trainingParameters, self.trainingValues, self.metrics,
                                                 views=self.config.get_views(), use_cuda=self.use_cuda)

        for epoch in range(1, self.config.get_epochs() + 1):  # start training
            tick = datetime.now()  # track real length of training step to evaluate computation time
            self._ticks.append(tick)
            print(f'Epoch: {epoch}')

            # do training
            self._train_epoch(train_loader)

            # do some statistics and printing
            if self.config.get_convergence_metrics_update_interval() > 0 and epoch % self.config.get_convergence_metrics_update_interval() == 0:
                stop = self.update_metrics(epoch)
                if stop:
                    print('statistics threshold reached')
                    break
            if self.config.get_console_log_update_interval() > 0 and epoch % self.config.get_console_log_update_interval() == 0:
                self.print_message(epoch)
            if self.config.get_visualisation_update_interval() > 0 and epoch % self.config.get_visualisation_update_interval() == 0:
                self.plotHandler.update_from_generator(epoch, self.G, self.metric_timelines)
                plt.savefig(self.config.get_resultfolder() + '/visualisation.png', dpi=300)
            if 'reductionInterval' in self.config.get_learning_rate_hyperparams().keys() and \
                    self.config.get_learning_rate_hyperparams()['reductionInterval'] > 0 and epoch % int(
                    self.config.get_learning_rate_hyperparams()['reductionInterval']) == 0:
                self.reduce_learning_rate()

            # update metrics
            self.metric_timelines[self._lossGeneratorName][0].append(epoch)
            self.metric_timelines[self._lossGeneratorName][1].append(self.metric_values[self._lossGeneratorName])
            self.metric_timelines[self._lossCriticName][0].append(epoch)
            self.metric_timelines[self._lossCriticName][1].append(self.metric_values[self._lossCriticName])

            # save snapshot
            if self.config.get_snapshot_export_update_interval() > 0 and epoch % self.config.get_snapshot_export_update_interval() == 0:
                self.save_snapshot(epoch)
        if self.config.get_visualisation_update_interval() > 0:
            self.plotHandler.update_from_generator(epoch, self.G, self.metric_timelines)
            plt.savefig(self.config.get_resultfolder() + '/visualisation.png', dpi=300)
        self.save_snapshot(epoch, final=True)
        if epoch == self.config.get_epochs():
            print('finished due to epoch maximum')
        else:
            print('finished due to threshold')

    def reduce_learning_rate(self):
        """
        Method called internally to reduce the learning rate parameters as specified in the configuration (loss function hyperparameters: 'reductionFactor' and 'minimum')
        """
        for g in self.G_opt.param_groups:
            if g['lr'] > self.config.get_loss_function_hyperparams()['minimum']:
                g['lr'] *= self.config.get_loss_function_hyperparams()['reductionFactor']
        for g in self.C_opt.param_groups:
            if g['lr'] > self.config.get_loss_function_hyperparams()['minimum']:
                g['lr'] *= self.config.get_loss_function_hyperparams()['reductionFactor']

    def save_snapshot(self, epoch: int, final: bool = False):
        """
        Saves a snapshot of the current status of the training process. This includes
        - a .state export of the generator and the critic network
        - a .torch export of the generator and the critic network
        - a .json file with time-series of the tracked metrics
        - (if plotting is active) a .png image with a visualisation of the training process
        If the final parameter is checked, the files will be exported to the overall result folder of the experiment,
        otherwise to a newly generated folder according to the epoch.
        :param epoch: current epoch of the training process
        :param final: if true, the output will be exported to the main experiment folder. Otherwise to a newly generated subfolder.
        """
        if final:
            folder = self.config.get_resultfolder()
        else:
            snapfolder = os.path.join(self.config.get_resultfolder(), 'snapshots')
            if not os.path.isdir(snapfolder):
                os.mkdir(snapfolder)
            folder = os.path.join(snapfolder, str(epoch))
            if not os.path.isdir(folder):
                os.mkdir(folder)
        torch.save(self.G.state_dict(), folder + '/generator_params_{:}.state'.format(epoch))
        model_scripted = torch.jit.script(self.G)
        model_scripted.save(folder + '/generator_torchScript_{:}.pt'.format(epoch))
        torch.save(self.C.state_dict(), folder + '/critic_params_{:}.state'.format(epoch))
        model_scripted = torch.jit.script(self.C)
        model_scripted.save(folder + '/critic_torchScript_{:}.pt'.format(epoch))
        snapshot = dict()
        snapshot['metrics'] = self.metric_timelines
        with open(folder + '/metrics.json', 'w') as f:
            json.dump(snapshot, f, indent=4)

    def print_message(self, epoch: int):
        """
        Prints out a console message with the most important informations
        :param epoch: current training epoch
        """
        for key in self.metrics:
            print("{}: {}".format(key, self.metric_values[key]))
        avg = (self._ticks[-1] - self._ticks[0]) / len(self._ticks)
        togo = avg * (self.config.get_epochs() - epoch)
        past = epoch * avg
        print('epochs - finished / to-go:  {:}/{:}'.format(epoch, self.config.get_epochs() - epoch))
        d0 = datetime(2020, 1, 1)
        print('time - finished / to-go:  {:%H:%M:%S}/{:%H:%M:%S}'.format(d0 + past, d0 + togo))

    def update_metrics(self, epoch: int):
        """
        Updates the internal convergence metrics using the current status of the generator network
        :param epoch: current training epoch
        """
        statistics, isStop = self.convergenceMetric.eval_generator(self.G)
        for k, v in statistics.items():  # update tracker variables to create output
            self.metric_values[k] = v
            self.metric_timelines[k][0].append(epoch + 1)
            self.metric_timelines[k][1].append(v)
        return isStop

    def _train_epoch(self, data_loader: torch.utils.data.DataLoader):
        """
        create one training iteration of critic and generator network using the training data provided in the data_loader
        :param data_loader: container for the trainig data
        """
        for i, data in enumerate(data_loader):  # iterate over the batches
            self._critic_train_iteration(data[0])
            self._generator_train_iteration(data[0])

    def _critic_train_iteration(self, data: torch.Tensor):
        """
        One trainig iteration for the critic network
        :param data: one batch of training data
        """
        batch_size = data.size()[0]
        generated = self.sample_generator(batch_size)
        if self.use_cuda:
            output_data = self.C(data.cuda())
        else:
            output_data = self.C(data)
        output_generated = self.C(generated)
        loss = self.loss_critic(data, generated, output_data, output_generated)
        self.C_opt.zero_grad()
        if self.lossfunctionID == LossFunctionID.WASSERSTEIN_CLAMP:  # clamp parameters after training if Wasserstein Clamping is active
            for p in self.C.parameters():
                p.data.clamp_(-self.wassersteinClamp, self.wassersteinClamp)
        loss.backward()  # compute gradients
        self.C_opt.step()  # make optimizer step
        self.metric_values[self._lossCriticName] = float(loss.cpu().mean().data.numpy())  # update tracked metric

    def _generator_train_iteration(self, data: torch.Tensor):
        """
        One trainig iteration for the generator network
        :param data: one batch of training data
        """
        batch_size = data.size()[0]
        generated_data = self.sample_generator(batch_size)
        d_generated = self.C(generated_data)
        g_loss = self.loss_generator(d_generated)
        self.G_opt.zero_grad()
        g_loss.backward()  # compute gradients
        self.G_opt.step()  # make optimizer step
        self.metric_values[self._lossGeneratorName] = float(g_loss.cpu().mean().data.numpy())  # update tracked metric

    def loss_generator_base(self, output_latent: torch.Tensor) -> torch.Tensor:
        """
        Basic binary loss function for the generator.
        For every synthetic C(u,G(u,X)) =: ol (i.e. latent output), we return 1-ol as loss.
        I.e. if the critic (discriminator) correctly predicted a synthetic value (ol~0), the generator is punished by a large loss
        Note: use only if the critic output is normed to [0,1], e.g. via sidmoid
        :param output_latent: output of the critic for the synthetic generated values
        :return: loss for the generator network
        """
        if self.use_cuda:
            return self.loss_fct_binary(output_latent, torch.ones((output_latent.size(0), 1)).cuda())
        else:
            return self.loss_fct_binary(output_latent, torch.ones((output_latent.size(0), 1)))

    def loss_generator_log(self, output_latent: torch.Tensor) -> torch.Tensor:
        """
        Logarithmic loss function for the generator.
        For every synthetic C(u,G(u,X)) =: ol (i.e. latent output), we return -log(ol) as loss.
        I.e. if the critic (discriminator) correctly predicted a synthetic value (ol~0), the generator is punished by a large loss
        Note: use only if the critic output is normed to [0,1], e.g. via sidmoid
        :param output_latent: output of the critic for the synthetic generated values
        :return: loss for the generator network
        """
        return -torch.log(output_latent).mean()

    def loss_generator_wasserstein(self, output_latent: torch.Tensor) -> torch.Tensor:
        """
        Wasserstein loss function for the generator.
        For every synthetic C(u,G(u,X)) =: ol (i.e. latent output), we return -ol as loss.
        I.e. for a low critic rating for a synthetic value (i.e. ol is small), the generator is punished by a large loss
        :param output_latent: output of the critic for the synthetic generated values
        :return: loss for the generator network
        """
        return -output_latent.mean()

    def loss_critic_base(self, values_data: torch.Tensor, values_latent: torch.Tensor, output_data: torch.Tensor,
                         output_latent: torch.Tensor) -> torch.Tensor:
        """
        Basic binary loss function for the critic.
        We evaluate whether the critic (discriminator) correctly distinguishes wright from wrong:
        For every synthetic C(u,G(u)) = ol and every actual C(u,y)=oa, we return ol +(1-oa).
        I.e. the worse the critic recognizes the actual training values y, (i.e. oa~0)
        and the worse the critic recognizes the generated value G(u) (i.e. ol~1),
        the larger the loss for the critic (discrminiator).
        Note: use only if the critic output is normed to [0,1], e.g. via sidmoid
        :param values_data: values from the actual training set
        :param values_latent: synthetic values from the generator
        :param output_data: output of the critic for values_data
        :param output_latent: output of the critic for values_latent
        :return: loss for the critic network
        """
        batch_size = values_data.size()[0]
        critic_output = torch.concat((output_data, output_latent))
        labelled = torch.concat((torch.ones((batch_size, 1)), torch.zeros((batch_size, 1))))
        if self.use_cuda:
            return self.loss_fct_binary(critic_output.cuda(), labelled.cuda())
        else:
            return self.loss_fct_binary(critic_output, labelled)

    def loss_critic_log(self, values_data: torch.Tensor, values_latent: torch.Tensor, output_data: torch.Tensor,
                        output_latent: torch.Tensor) -> torch.Tensor:
        """
        Logarithmic loss function for the critic.
        We evaluate whether the critic (discriminator) correctly distinguishes wright from wrong:
        For every synthetic C(u,G(u)) = ol and every actual C(u,y)=oa, we return -log(oa)-log(1-ol).
        I.e. the worse the critic recognizes the actual training values y, (i.e. oa~0)
        and the worse the critic recognizes the generated value G(u) (i.e. ol~1),
        the larger the loss for the critic (discrminator).
        Note: use only if the critic output is normed to [0,1], e.g. via sidmoid
        :param values_data: values from the actual training set
        :param values_latent: synthetic values from the generator
        :param output_data: output of the critic for values_data
        :param output_latent: output of the critic for values_latent
        :return: loss for the critic network
        """
        m1 = torch.log(output_data)
        m2 = torch.log(1 - output_latent)
        return -m1.mean() - m2.mean()

    def loss_critic_wasserstein_wc(self, values_data: torch.Tensor, values_latent: torch.Tensor,
                                   output_data: torch.Tensor, output_latent: torch.Tensor) -> torch.Tensor:
        """
        Wasserstein loss function for the critic for weight clipping.
        We evaluate whether the critic correctly distinguishes wright from wrong:
        For every synthetic C(u,G(u)) = ol and every actual C(u,y)=oa, we return -oa+ol.
        I.e. the worse the critic scores the actual training values y, (i.e. oa is small)
        and the worse the critic scores the generated value G(u) (i.e. ol is large),
        the larger the loss for the critic.
        :param values_data: values from the actual training set
        :param values_latent: synthetic values from the generator
        :param output_data: output of the critic for values_data
        :param output_latent: output of the critic for values_latent
        :return: loss for the critic network
        """
        return - output_data.mean() + output_latent.mean()

    def loss_critic_wasserstein_gp(self, values_data: torch.Tensor, values_latent: torch.Tensor,
                                   output_data: torch.Tensor, output_latent: torch.Tensor) -> torch.Tensor:
        """
        Wasserstein loss function for the critic with gradient penalty.
        We evaluate whether the critic correctly distinguishes wright from wrong:
        For every synthetic C(u,G(u)) = ol and every actual C(u,y)=oa, we return -oa+ol+alpha*sqrt(1-GradC^2)
        I.e. the worse the critic scores the actual training values y, (i.e. oa is small),
        the worse the critic scores the generated value G(u) (i.e. ol is large),
        the higher the gradient penalty factor alpha, and
        the larger the difference between the computed gradients GradC of the critic network and one (compare Kantonovic-Rubinstein Duality),
        the larger the loss for the critic.
        :param values_data: values from the actual training set
        :param values_latent: synthetic values from the generator
        :param output_data: output of the critic for values_data
        :param output_latent: output of the critic for values_latent
        :return: loss for the critic network
        """
        batch_size, dim = values_data.size()
        alpha = torch.rand(batch_size, dim)
        if self.use_cuda:
            values_data = values_data.cuda()
            alpha = alpha.cuda()
        values_interpolated = alpha * values_data + (
                    1 - alpha) * values_latent  # compute some random but representative point on the value-space
        output_interpolated = self.C(values_interpolated)  # evaluate critic there
        if self.use_cuda:
            output_interpolated = output_interpolated.cuda()
        gradout = torch.ones_like(output_interpolated)
        gradients = torch.autograd.grad(outputs=output_interpolated, inputs=values_interpolated,
                                        grad_outputs=gradout,
                                        create_graph=True, retain_graph=True, only_inputs=True)[
            0]  # compute the gradients w.r. to the input
        gradient_norm = gradients.view(len(gradients), -1).norm(2, dim=1)
        gradient_penalty = (gradient_norm - 1) ** 2  # compute the discrepancy of the gradients from 1
        return -output_data.mean() + output_latent.mean() + self.wassersteinGPFactor * gradient_penalty.mean()

    def sample_generator(self, num_samples: int) -> torch.Tensor:
        """
        Creates a set of synthetic values
        :param num_samples: number of values
        :return: latent parameter vectors concatenated with latent values generated via the generator network
        """
        params, rands = self.sample_latent(num_samples)
        latent_samples = Variable(torch.concat([params, rands], 1))
        if self.use_cuda:
            latent_samples = latent_samples.cuda()
            params = params.cuda()
        generated_data = self.G(latent_samples)
        data = torch.concat([params, generated_data.view((num_samples, 1))], 1)
        return data

    def sample_latent(self, num_samples: int) -> (torch.Tensor, torch.Tensor):
        """
        Creates a synthetic input for the generator, i.e. takes a ranodom set of parameter vectors from the training data and adds a random noise.
        :param num_samples: number of generator inputs to create
        :return: parameter vectors and corresponding random noise for the generator
        """
        random_source = torch.rand((num_samples, 1))
        parameters = torch.tensor(np.random.default_rng().choice(self.trainingParameters, num_samples))
        return parameters, random_source
