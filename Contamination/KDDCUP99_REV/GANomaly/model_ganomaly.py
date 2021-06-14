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

from scipy.optimize import brentq
from scipy.interpolate import interp1d

import pandas as pd
import os
import sys

seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

class GANomaly(nn.Module):
    def __init__(self, nc, nz, device):
        super(GANomaly, self).__init__()
        
        self.nc = nc
        self.nz = nz
        self.device = device
        
        self.G = netG(self.nc, self.nz).to(self.device)
        self.D = netD(self.nc).to(self.device)
        self.G.apply(weights_init)
        self.D.apply(weights_init)

        self.d_optimizer = torch.optim.Adam(self.D.parameters(), lr=2e-4)
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), lr=2e-4)
        
        self.l1_loss = nn.L1Loss()
        self.bce = nn.BCELoss() #nn.BCEWithLogitsLoss(reduction='mean')
        self.mse = nn.MSELoss()
        
    def plotLoss(self, d_losses, g_losses):
        num_epochs = len(d_losses)
        plt.figure()
        plt.xlim(0, num_epochs + 1)
        plt.plot(range(1, num_epochs + 1), d_losses[:num_epochs], label='d loss')
        plt.legend(loc = 'lower right', bbox_to_anchor=(.8, 0.3, 0.5, 0.5))
        plt.show()
        plt.plot(range(1, num_epochs + 1), g_losses[:num_epochs], label='g loss')    
        plt.legend(loc = 'lower right', bbox_to_anchor=(.8, 0.3, 0.5, 0.5))
        plt.show()

    def test(self, test_loader):
        self.G.eval()
        
        for i, (batch, label) in enumerate(test_loader, 0):
            bs = batch.size(0)

            with torch.no_grad():
                _, latent_i, latent_o = self.G(batch)

            error = torch.mean(torch.pow((latent_i - latent_o), 2), dim=1).to('cpu')

            if i == 0:
                scores = error.cpu()
                labels = label.cpu()
            else:
                scores = torch.cat((scores, error.cpu()))
                labels = torch.cat((labels, label.cpu()))

            #scores[i * bs : i * bs + error.size(0)] = error.reshape(error.size(0))
            #labels[i * bs : i * bs + error.size(0)] = label.reshape(error.size(0))


        # Normalize score
        scores = (scores - torch.min(scores)) / (torch.max(scores) - torch.min(scores))
        
        labels = labels.cpu()
        scores = scores.cpu()
        
        _auc = roc_auc_score(labels, scores)
        #auc = self.roc(labels, scores)
        pr_auc = self.pr_auc(labels, scores)

        return _auc, pr_auc

    def roc(self, labels, scores):
        """Compute ROC curve and ROC area for each class"""
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        labels = labels.cpu()
        scores = scores.cpu()
        
        # True/False Positive Rates.
        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)

        return roc_auc
    
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

    def train_and_test(self, train_loader, test_loader, args, num_epochs = 100, bs = 32):
        d_losses = np.zeros(num_epochs)
        g_losses = np.zeros(num_epochs)

        print("[INFO] Starting training phase...\n")
        start = time()

        try:
            for epoch in range(num_epochs):
                self.G.train()
                self.D.train()

                i = 0
                for batch, _ in train_loader:
                    batch = batch.to(self.device)
                    self.G.zero_grad()
                    self.D.zero_grad()

                    """
                    Forward Pass
                    """
                    gen_data, latent_input, latent_output = self.G(batch)
                    pred_real, feat_real = self.D(batch)
                    pred_fake, feat_fake = self.D(gen_data.detach())
                    
                    ### Backward pass
                    loss_g_adv = l2_loss(self.D(batch)[1], self.D(gen_data)[1])
                    loss_g_con = self.l1_loss(gen_data, batch)
                    loss_g_enc = l2_loss(latent_output, latent_input)
                    
                    w_adv = args['w_adv']
                    w_con = args['w_con']
                    w_enc = args['w_enc']
                    g_loss = w_adv * loss_g_adv + w_con * loss_g_con + w_enc * loss_g_enc
                    g_losses[epoch] = g_losses[epoch]*(i/(i+1.)) + g_loss.item()*(1./(i+1.))
                    g_loss.backward(retain_graph=True)
                    self.g_optimizer.step()


                    real_label = torch.ones(size=(bs,), dtype=torch.float32, device=self.device)
                    fake_label = torch.zeros(size=(bs,), dtype=torch.float32, device=self.device)

                    loss_d_real = self.bce(pred_real, real_label)
                    loss_d_fake = self.bce(pred_fake, fake_label)

                    d_loss = (loss_d_real + loss_d_fake) * 0.5
                    d_losses[epoch] = d_losses[epoch]*(i/(i+1.)) + d_loss.item()*(1./(i+1.))

                    d_loss.backward()
                    self.d_optimizer.step()

                    if d_loss.item() < 1e-5:
                        self.D.apply(weights_init)

                    i += 1

                sys.stdout.write("\r" + 'Epoch [{:>3}/{}] | d_loss: {:.3f} | g_loss: {:.3f}'
                                      .format(epoch+1, num_epochs, d_losses[epoch], g_losses[epoch]))
                sys.stdout.flush()


        except KeyboardInterrupt:
            print('*'*89)
            print('[INFO] Exiting from training early')
        print(f'\n[INFO] Training phase... Elapsed time: {(time() - start):.0f} seconds\n')
        self.plotLoss(d_losses, g_losses)
        _auc, pr_auc = self.test(test_loader)

        return _auc, pr_auc

def l1_loss(x, y):
    return torch.mean(torch.abs(x - y))


def l2_loss(x, y, size_average=True):
    if size_average:
        return torch.mean(torch.pow((x-y), 2))
    else:
        return torch.pow((x-y), 2)

def weights_init(mod):
    classname = mod.__class__.__name__
    
    if classname.find('Linear') != -1:
        mod.weight.data.normal_(0, 0.01)
        mod.bias.data.fill_(0)
        
    elif classname.find('BatchNorm') != -1:
        mod.weight.data.normal_(1.0, 0.02)
        mod.bias.data.fill_(0)
        
class Encoder(nn.Module):
    def __init__(self, nc, nz = 16):
        super(Encoder, self).__init__()
        
        self.nc = nc
        self.nz = nz
        
        self.main = nn.Sequential(
            nn.Linear(self.nc, 64),
            nn.LeakyReLU(0.2),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            
            nn.Linear(32, self.nz)
        )
        
    def forward(self, x):
        return self.main(x)
    
    
class Decoder(nn.Module):
    def __init__(self, nc, nz):
        super(Decoder, self).__init__()
        
        self.nc = nc
        self.nz = nz
        
        self.main = nn.Sequential(
            nn.Linear(self.nz, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.Linear(64, self.nc),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.main(x)

    
class netD(nn.Module):
    def __init__(self, nc):
        super(netD, self).__init__()
        
        self.nc = nc
        self.model = Encoder(self.nc, 1)
        self.layers = list(self.model.main.children())
        
        self.features = nn.Sequential(*self.layers[:-1])
        self.classifier = nn.Sequential(self.layers[-1])
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        features = self.features(x)
        classifier = self.classifier(features)
        out = self.sigmoid(classifier) #classifier.view(-1, 1).squeeze(1)
        
        return out.flatten(), features 


class netG(nn.Module):
    def __init__(self, nc, nz):
        super(netG, self).__init__()
        
        self.nc = nc
        self.nz = nz
        
        self.encoder1 = Encoder(nc, nz)
        self.decoder = Decoder(nc, nz)
        self.encoder2 = Encoder(nc, nz)
        
    def forward(self, x):
        latent_x = self.encoder1(x)
        gen_x = self.decoder(latent_x)
        latent_x1 = self.encoder2(gen_x)
        
        return gen_x, latent_x, latent_x1
    
    
