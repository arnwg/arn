import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import nn

from time import time
from tqdm import tqdm

from sklearn.metrics import roc_auc_score, average_precision_score,auc, roc_curve
from sklearn.metrics import precision_recall_fscore_support, auc, precision_recall_curve
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from sklearn.utils import shuffle

import pandas as pd
import os
import sys

seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

class Generator(nn.Module):
    def __init__(self, nc, z_dim):
        super(Generator, self).__init__()
        
        self.nc = nc
        self.z_dim = z_dim
        self.fc1 = nn.Linear(self.z_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, self.nc)
        
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.drop(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        x = self.drop(x)
        
        out = self.fc3(x)
        
        return out
    
    
class Discriminator(nn.Module):
    def __init__(self, nc):
        super(Discriminator, self).__init__()
        
        self.nc = nc
        self.fc1 = nn.Linear(self.nc, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 1)
        
        self.lrelu = nn.LeakyReLU(0.1)
        self.sigmoid = nn.Sigmoid()
        
        self.drop = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.lrelu(x)
        x = self.drop(x)
        
        x = self.fc2(x)
        x = self.lrelu(x)
        x = self.drop(x)
        
        x = self.fc3(x)
        x = self.lrelu(x)
        x = self.drop(x)
        
        x = self.fc4(x)
        out = self.sigmoid(x)
        
        return out.flatten()
    
    
class GeneratorLoss(nn.Module):
    def __init__(self, beta, _power):
        super(GeneratorLoss, self).__init__()
        
        self.beta = beta
        self._power = _power
        
        self.bce = nn.BCELoss()
        
    ### Average distance from the Center of Mass    
    def forward(self, generated_data, y_true, y_pred):
        # dispersion_loss
        loss_b = self.bce(y_pred, y_true)
        center = torch.mean(generated_data, dim = 0, keepdim = True)
        distance_xy = torch.pow(torch.sub(generated_data, center), 2)
        distance = torch.sum(distance_xy, 1)
        avg_distance = torch.mean(torch.sqrt(distance))
        loss_d = torch.reciprocal(avg_distance)
        
        loss = loss_b + self.beta*loss_d
        
        return loss
    
    
class DiscriminatorLoss(nn.Module):
    def __init__(self, gamma):
        super(DiscriminatorLoss, self).__init__()
        
        self.gamma = gamma
        self.bce = nn.BCELoss()
    
    def forward(self, y_true, true_pred, y_fake, fake_pred):
        loss_real = self.bce(true_pred, y_true)
        loss_gen = self.bce(fake_pred, y_fake)
        return loss_real + self.gamma * loss_gen
    
    
class FenceGAN(nn.Module):
    def __init__(self, nc, z_dim, gamma, alpha, beta, _power, v_freq, g_objective_anneal, repeat, baseline, device):
        super(FenceGAN, self).__init__()
        
        self.nc = nc
        self.z_dim = z_dim
        self.gamma = gamma
        self.beta = beta
        self.alpha = alpha
        self._power = _power
        self.v_freq = v_freq
        self.g_objective_anneal = g_objective_anneal
        self.repeat = repeat
        self.baseline = baseline
        self.device = device
        
        self.G = Generator(self.nc, self.z_dim).to(self.device)
        self.D = Discriminator(self.nc).to(self.device)
        
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), lr=1e-4, weight_decay=1e-3)
        self.d_optimizer = torch.optim.SGD(self.D.parameters(), lr=1e-4, weight_decay=1e-3, momentum = 0.9)
        
        self.disc_loss = DiscriminatorLoss(self.gamma)
        self.gen_loss = GeneratorLoss(self.beta, self._power)
        
    def D_step(self, x_true, y_true):
        self.D.zero_grad()
        bs = x_true.size(0)
        
        true_pred = self.D(x_true)
        
        noise = torch.normal(0, 1, size=(bs, self.z_dim)).to(self.device)
        generated_data = self.G(noise)
        
        fake_pred = self.D(generated_data)  
        y_fake = torch.zeros(bs).to(self.device)
        
        d_loss = self.disc_loss(y_true, true_pred, y_fake, fake_pred)
        
        d_loss.backward()
        self.d_optimizer.step()

        return d_loss
    
    def G_step(self, bs, epoch):
        self.G.zero_grad()
        
        noise = torch.normal(0, 1, size=(bs, self.z_dim)).to(self.device)
        generated_data = self.G(noise)
        
        fake_pred = self.D(generated_data)
        y_fake = torch.zeros(bs).to(self.device)
        y_fake[:] = self.label_annealing(epoch)
        
        g_loss = self.gen_loss(generated_data, y_fake, fake_pred)
        g_loss.backward()
        self.g_optimizer.step()
        
        return g_loss
    
    def label_annealing(self, epoch):
        x = (epoch + 1) % self.repeat
        if self.g_objective_anneal == 1:
            label = self.baseline
        else:
            label = torch.min(1, self.baseline + torch.pow(np.e, (-(self.g_objective_anneal)) * x))
        return label
         
        
    def pretrain(self, train_loader, batch_size = 32):
        try:
            count = 1
            self.D.train()
            d_losses = []
            for batch, label in train_loader:
                batch = batch.to(self.device)
                label = label.to(self.device)

                d_loss = self.D_step(batch, label)
                d_losses.append(d_loss.item())
            
            d_loss = np.mean(d_losses)
            print("Epoch #1: Loss: {:.4f}".format(d_loss))
            while d_loss > 7:
                count += 1
                d_losses = []
                for batch, label in train_loader:
                    batch = batch.to(self.device)
                    label = label.to(self.device)

                    d_loss = self.D_step(batch, label)
                    d_losses.append(d_loss.item())
            
                d_loss = np.mean(d_losses)
                
                sys.stdout.write("\r" + 'Epoch [{:>3}] | d_loss: {:.4f} |'
                                  .format(count, d_loss))
                sys.stdout.flush()
                
        except KeyboardInterrupt:
            print('*'*89)
            print('[INFO] Exiting from pre-training early')
            
    def plotDGLoss(self, d_losses, g_losses):
        num_epochs = len(d_losses)
        plt.figure()
        plt.xlim(0, num_epochs + 1)
        plt.plot(range(1, num_epochs + 1), d_losses[:num_epochs], label='d loss')
        plt.legend(loc = 'lower right', bbox_to_anchor=(.8, 0.3, 0.5, 0.5))
        plt.show()
        plt.plot(range(1, num_epochs + 1), g_losses[:num_epochs], label='g loss')    
        plt.legend(loc = 'lower right', bbox_to_anchor=(.8, 0.3, 0.5, 0.5))
        plt.show()
        
        
    def train(self, train_loader, test_loader, val_loader, batch_size = 32, n_epochs = 100):
        self.pretrain(train_loader, batch_size)
        d_losses = np.zeros(n_epochs)
        g_losses = np.zeros(n_epochs)
        
        print("[INFO] Starting training phase...")
        start = time()
        
        try: 
            for epoch in range(n_epochs):
                self.D.train()
                self.G.train()
                i = 0
                for batch, label in train_loader:
                    ### Train Discriminator ###
                    batch = batch.to(self.device)
                    label = label.to(self.device)

                    d_loss = self.D_step(batch, label)
                    
                    ### Train Generator ###
                    
                    g_loss = self.G_step(batch_size, epoch)
                    
                    d_losses[epoch] = d_losses[epoch]*(i/(i+1.)) + d_loss.item()*(1./(i+1.))
                    g_losses[epoch] = g_losses[epoch]*(i/(i+1.)) + g_loss.item()*(1./(i+1.))
                    
                    i += 1
                #if (epoch + 1) % self.v_freq == 0:
                #    mode_eval = 'auroc'
                #    val, test = self.compute_au(val_loader, test_loader, mode_eval)    
                #    print(f'\tGen. Loss: {g_losses[epoch]:.3f}\n\tDisc. Loss: {d_losses[epoch]:.3f}\n\t{mode_eval}: {val:.3f}')
                
                #else:
                sys.stdout.write("\r" + 'Epoch [{:>3}/{}] | d_loss: {:.3f} | g_loss: {:.3f}'
                                  .format(epoch+1, n_epochs, d_losses[epoch], g_losses[epoch]))
                sys.stdout.flush()
                              
        except KeyboardInterrupt:
            print('*'*89)
            print('[INFO] Exiting from training early')
        print(f'\n[INFO] Training phase... Elapsed time: {(time() - start):.0f} seconds\n')
        self.plotDGLoss(d_losses[:epoch], g_losses[:epoch])
        return d_losses[:epoch], g_losses[:epoch]
    
    def predict(self, test_loader):
        self.D.eval()

        x_normal = test_loader.dataset.tensors[0][test_loader.dataset.tensors[1] == 1]
        x_anomalous = test_loader.dataset.tensors[0][test_loader.dataset.tensors[1] == 0]

        y_anomalous = test_loader.dataset.tensors[1][test_loader.dataset.tensors[1] == 0]
        y_normal = test_loader.dataset.tensors[1][test_loader.dataset.tensors[1] == 1]
        label = np.concatenate((y_anomalous.cpu(), y_normal.cpu()))

        pred_normal = self.D(x_normal)
        pred_anomaly = self.D(x_anomalous)
        pred = np.concatenate((pred_anomaly.detach().cpu(), pred_normal.detach().cpu()))


        _auc = roc_auc_score(1-label, 1-pred)
        pr_auc = self.pr_auc(1-label, 1-pred)
        
        return _auc, pr_auc
    
    def plot_pr_curve(self, precision, recall):
        plt.figure()
        plt.plot(recall, precision, marker='.')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.show()
    
    def pr_auc(self, y_test, y_pred):
        precision, recall, _ = precision_recall_curve(y_test, y_pred)
        auc_score = auc(recall, precision)
        print(f'PR AUC: {auc_score:.2f}')
        self.plot_pr_curve(precision, recall)
        return auc_score
            