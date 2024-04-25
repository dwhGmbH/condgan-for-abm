from datetime import datetime
from enum import Enum

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.autograd import Variable
from torch.optim import Optimizer
import os
import json

from config import CondGANConfig, LossFunctionID, ConvergenceMetricsID
from kstest_metric import KSTestMetric
from mixed_metric import MixedMetric
from moments_metric import MomentsMetric
from networks import create_mlps_from_config
from validation_plotter import ValidationPlotter

class CondGANTrainer():
    def __init__(self, config:CondGANConfig,use_cuda=False):
        self.config = config
        self.G,self.C = create_mlps_from_config(self.config)
        self.use_cuda = use_cuda
        if self.use_cuda:
            self.G.cuda()
            self.C.cuda()
        self.G_opt = torch.optim.Adam(self.G.parameters(), lr=self.config.get_learning_rate())
        self.C_opt = torch.optim.Adam(self.C.parameters(), lr=self.config.get_learning_rate())

        self.lossfunctionID = self.config.get_loss_function_id()
        if self.lossfunctionID == LossFunctionID.BINARY:
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

        self._lossGeneratorName = 'loss Generator'
        self._lossCriticName = 'loss Critic'
        self._ticks = list()

    def train(self, parameters:np.array,values:np.array):
        self.trainingParameters = np.array(parameters,dtype=np.float32)
        self.trainingValues = np.array(values,dtype=np.float32)
        self.inputDimension = np.size(parameters,1)
        dataCount = np.size(parameters,0)

        train_data = torch.concat([torch.tensor(self.trainingParameters), torch.tensor(self.trainingValues).view(dataCount, 1)], 1)
        train_labels = torch.zeros(dataCount)
        train_set = [(x,y) for x,y in zip(train_data,train_labels)]
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=self.config.get_batch_size(), shuffle=True
        )
        self.metrics = [self._lossGeneratorName, self._lossCriticName]
        if self.config.get_convergence_metrics_update_interval() > 0:
            self.convergenceMetricID =self.config.get_convergence_metric()
            if self.convergenceMetricID == ConvergenceMetricsID.MOMENTS:
                self.convergenceMetric = MomentsMetric(self.trainingParameters, self.trainingValues, threshold=self.config.get_convergence_metrics_stopping_threshold(), use_cuda=self.use_cuda)
            elif self.convergenceMetricID == ConvergenceMetricsID.KSTEST:
                self.convergenceMetric= KSTestMetric(self.trainingParameters, self.trainingValues, threshold=self.config.get_convergence_metrics_stopping_threshold(), use_cuda=self.use_cuda)
            elif self.convergenceMetricID == ConvergenceMetricsID.MIXED:
                self.convergenceMetric= MixedMetric(self.trainingParameters, self.trainingValues, threshold=self.config.get_convergence_metrics_stopping_threshold(), use_cuda=self.use_cuda)
            self.metrics.extend(self.convergenceMetric.get_statistics_names())
        self.metricValues = {x:0 for x in self.metrics}
        self.metricsTimeline = {x:[list(),list()] for x in self.metrics}
        if self.config.get_visualisation_update_interval() > 0:
            self.plotHandler = ValidationPlotter(self.trainingParameters,self.trainingValues,self.metrics,views=self.config.get_views(),use_cuda=self.use_cuda)

        for epoch in range(1,self.config.get_epochs()+1):
            tick = datetime.now()
            self._ticks.append(tick)
            print('Epoch: {}'.format(epoch))

            # do training
            self._train_epoch(train_loader)

            # do some statistics and printing
            if self.config.get_convergence_metrics_update_interval()>0 and epoch % self.config.get_convergence_metrics_update_interval() == 0:
                stop = self.update_statistics(epoch)
                if stop:
                    print('statistics threshold reached')
                    break
            if self.config.get_console_log_update_interval()>0 and epoch % self.config.get_console_log_update_interval() == 0:
                self.print_message(epoch)
            if  self.config.get_visualisation_update_interval() > 0 and epoch%self.config.get_visualisation_update_interval() == 0:
                self.plotHandler.update_from_generator(epoch, self.G, self.metricsTimeline)
                plt.savefig(self.config.get_resultfolder()+'/visualisation.png', dpi=300)
            if 'reductionInterval' in self.config.get_learning_rate_hyperparams().keys() and self.config.get_learning_rate_hyperparams()['reductionInterval']>0 and epoch%int(self.config.get_learning_rate_hyperparams()['reductionInterval']) == 0:
                self.reduce_learning_rate()

            #update metrics
            self.metricsTimeline[self._lossGeneratorName][0].append(epoch)
            self.metricsTimeline[self._lossGeneratorName][1].append(self.metricValues[self._lossGeneratorName])
            self.metricsTimeline[self._lossCriticName][0].append(epoch)
            self.metricsTimeline[self._lossCriticName][1].append(self.metricValues[self._lossCriticName])

            # save snapshot
            if self.config.get_snapshot_export_update_interval()>0 and epoch % self.config.get_snapshot_export_update_interval() == 0:
                self.save_snapshot(epoch)
            tock = datetime.now()

        self.plotHandler.update_from_generator(epoch, self.G, self.metricsTimeline)
        plt.savefig(self.config.get_resultfolder() + '/visualisation.png', dpi=300)
        self.save_snapshot(epoch,final=True)
        if epoch == self.config.get_epochs():
            print('finished due to epoch maximum')
        else:
            print('finished due to threshold')

    def reduce_learning_rate(self):
        for g in self.G_opt.param_groups:
            if g['lr'] > self.config.get_loss_function_hyperparams()['minimum']:
                g['lr'] *= self.config.get_loss_function_hyperparams()['reductionFactor']
        for g in self.C_opt.param_groups:
            if g['lr'] > self.config.get_loss_function_hyperparams()['minimum']:
                g['lr'] *= self.config.get_loss_function_hyperparams()['reductionFactor']
        
    def save_snapshot(self,epoch,final=False):
        if final:
            folder = self.config.get_resultfolder()
        else:
            snapfolder = os.path.join(self.config.get_resultfolder(),'snapshots')
            if not os.path.isdir(snapfolder):
                os.mkdir(snapfolder)
            folder = os.path.join(snapfolder,str(epoch))
            if not os.path.isdir(folder):
                os.mkdir(folder)
        torch.save(self.G.state_dict(), folder+'/generator_params_{:}.state'.format(epoch))
        model_scripted = torch.jit.script(self.G)
        model_scripted.save(folder+'/generator_torchScript_{:}.pt'.format(epoch))
        torch.save(self.C.state_dict(), folder + '/critic_params_{:}.state'.format(epoch))
        model_scripted = torch.jit.script(self.C)
        model_scripted.save(folder+'/critic_torchScript_{:}.pt'.format(epoch))
        snapshot = dict()
        snapshot['metrics'] = self.metricsTimeline
        with open(folder+'/metrics.json','w') as f:
            json.dump(snapshot,f,indent=4)

    def print_message(self,epoch:int):
        for key in self.metrics:
            print("{}: {}".format(key,self.metricValues[key]))
        avg = (self._ticks[-1]-self._ticks[0])/len(self._ticks)
        togo = avg*(self.config.get_epochs()-epoch)
        past = epoch*avg
        print('epochs - finished / to-go:  {:}/{:}'.format(epoch, self.config.get_epochs()-epoch))
        d0= datetime(2020,1,1)
        print('time - finished / to-go:  {:%H:%M:%S}/{:%H:%M:%S}'.format(d0+past,d0+togo))



    def update_statistics(self,epoch):
        statistics,isStop = self.convergenceMetric.eval_generator(self.G)
        for k,v in statistics.items():
            self.metricValues[k] = v
            self.metricsTimeline[k][0].append(epoch + 1)
            self.metricsTimeline[k][1].append(v)
        return isStop

    def _train_epoch(self, data_loader):
        for i, data in enumerate(data_loader):
            self._critic_train_iteration(data[0])
            self._generator_train_iteration(data[0])

    def _critic_train_iteration(self, data):
        batch_size = data.size()[0]
        generated = self.sample_generator(batch_size)
        if self.use_cuda:
            output_data = self.C(data.cuda())
        else:
            output_data = self.C(data)
        output_generated = self.C(generated)
        loss = self.loss_critic(data, generated, output_data, output_generated)
        self.C_opt.zero_grad()
        if self.lossfunctionID==LossFunctionID.WASSERSTEIN_CLAMP:
            for p in self.C.parameters():
                p.data.clamp_(-self.wassersteinClamp, self.wassersteinClamp)
        loss.backward()
        self.C_opt.step()
        self.metricValues[self._lossCriticName] = float(loss.cpu().mean().data.numpy())

    def _generator_train_iteration(self, data):
        batch_size = data.size()[0]
        generated_data = self.sample_generator(batch_size)
        d_generated = self.C(generated_data)
        g_loss = self.loss_generator(d_generated)
        self.G_opt.zero_grad()
        g_loss.backward()
        self.G_opt.step()
        self.metricValues[self._lossGeneratorName] = float(g_loss.cpu().mean().data.numpy())

    def loss_generator_base(self,output_latent):
        if self.use_cuda:
            return self.loss_fct_binary(output_latent, torch.ones((output_latent.size(0), 1)).cuda())
        else:
            return self.loss_fct_binary(output_latent, torch.ones((output_latent.size(0),1)))

    def loss_generator_log(self,output_latent):
        return -torch.log(output_latent).mean()

    def loss_generator_wasserstein(self,output_latent):
        return -output_latent.mean()

    def loss_critic_base(self, data, latent, output_data, output_latent):
        batch_size = data.size()[0]
        critic_output = torch.concat((output_data,output_latent))
        labelled = torch.concat((torch.ones((batch_size,1)),torch.zeros((batch_size,1))))
        if self.use_cuda:
            return self.loss_fct_binary(critic_output.cuda(),labelled.cuda())
        else:
            return self.loss_fct_binary(critic_output,labelled)

    def loss_critic_log(self,data,latent,output_data,output_latent):
        m1 = torch.log(output_data)
        m2 = torch.log(1-output_latent)
        return -m1.mean()-m2.mean()

    def loss_critic_wasserstein_wc(self,data,latent,output_data,output_latent):
        return -(output_data.mean()-output_latent.mean())

    def loss_critic_wasserstein_gp(self,data,latent,output_data,output_latent):
        batch_size, dim = data.size()
        alpha = torch.rand(batch_size, dim)
        if self.use_cuda:
            data= data.cuda()
            alpha = alpha.cuda()
        interpolated = alpha*data+(1-alpha)*latent
        output_interpolated = self.C(interpolated)
        if self.use_cuda:
            output_interpolated = output_interpolated.cuda()
        gradout = torch.ones_like(output_interpolated)
        gradients = torch.autograd.grad(outputs=output_interpolated, inputs=interpolated,
                               grad_outputs=gradout,
                               create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradient_norm = gradients.view(len(gradients),-1).norm(2,dim=1)
        gradient_penalty = (gradient_norm - 1)**2
        return output_latent.mean()-output_data.mean()+self.wassersteinGPFactor*gradient_penalty.mean()

    def sample_generator(self, num_samples):
        rands,params = self.sample_latent(num_samples)
        latent_samples = Variable(torch.concat([params,rands],1))
        if self.use_cuda:
            latent_samples = latent_samples.cuda()
            params = params.cuda()
        generated_data = self.G(latent_samples)
        data = torch.concat([params,generated_data.view((num_samples,1))],1)
        return data

    def sample_latent(self,count):
        randomSource = torch.rand((count,1))
        parameters = torch.tensor(np.random.default_rng().choice(self.trainingParameters,count))
        return randomSource,parameters
