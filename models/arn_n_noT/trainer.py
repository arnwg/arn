import torch
import torch.nn as nn
import numpy as np

from time import time
import sys

from models.arn_n.model import Discriminator, Generator
from models.arn_n.loss import DiscriminatorLoss, GeneratorLoss
from utils.plot import plot_ARN_loss
from utils.util import save_arn_models

class ARN_N(nn.Module):
    def __init__(self, params):
        super(ARN_N, self).__init__()

        self.params = params
        self.device = self.params['device']
        self.nc = self.params['nc']

        self.D = Discriminator(nc = self.nc).to(self.device)
        self.G = Generator(nf_in = self.nc).to(self.device)

        self.d_optimizer = torch.optim.Adam(self.D.parameters(), lr=self.params['lr_D'])
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), lr=self.params['lr_G'])

        self.d_loss = DiscriminatorLoss(self.device)
        self.g_loss = GeneratorLoss(self.device)

        self.criterion = nn.BCELoss()


    def D_step(self, true_data, step):
        self.D.zero_grad()

        logits, _, _, sampled_data = self.G(true_data)
        true_pred = self.D(true_data)
        fake_pred = self.D(sampled_data.detach())

        d_loss_batch = self.d_loss(true_pred, fake_pred, step)
        d_loss_batch.backward()
        self.d_optimizer.step()

        return d_loss_batch, true_pred, fake_pred


    def G_step(self,true_data, step):
        self.G.zero_grad()

        logits, z_mean, z_logvar, sampled_data = self.G(true_data)
        fake_pred = self.D(sampled_data)

        gen_loss_batch, bce_loss, rec_loss, kl = self.g_loss(true_data, fake_pred, sampled_data,
                                                             z_mean, z_logvar)
        gen_loss_batch.backward()
        self.g_optimizer.step()

        return gen_loss_batch, bce_loss, rec_loss, kl


    def evaluation(self, val_loader):
        self.D.eval()

        d_l = []

        for batch, label in val_loader:
            batch = batch.to(self.device)
            label = label.to(self.device)

            with torch.no_grad():
                y_pred = self.D(batch)

            d_loss_batch = self.criterion(y_pred, label)
            d_l.append(d_loss_batch.item())

        return np.mean(d_l)


    def train(self, data_loader, val_loader, path_G, path_D, batch_size = 32, num_epochs = 10,
              step = 10, lowerbnd=5e-15, num_q_steps = 1, num_g_steps = 1, show_loss = True):

        d_losses = np.zeros(num_epochs)
        g_losses = np.zeros(num_epochs)
        real_scores = np.zeros(num_epochs)
        fake_scores = np.zeros(num_epochs)
        rec_losses = np.zeros(num_epochs)
        bce_losses = np.zeros(num_epochs)
        kldes = np.zeros(num_epochs)

        d_losses_val = np.zeros(num_epochs)

        total_steps = (len(data_loader.dataset) // batch_size)
        print("[INFO] Starting training phase...")
        start = time()

        bestLoss = np.inf

        try:

            step_count = 0
            for epoch in range(num_epochs):
                self.D.train()
                self.G.train()
                i = 0
                for batch in data_loader:

                    step_count += 1
                    batch = batch.to(self.device)

                    ### Train Discriminator ###
                    for _ in range(num_q_steps):
                        d_loss, real_score, fake_score = self.D_step(batch,step_count)

                    ### Train Generator ###
                    for _ in range(num_g_steps):
                        g_loss, bce_loss, rec_loss, kl = self.G_step(batch,step_count)

                    d_losses[epoch] = d_losses[epoch]*(i/(i+1.)) + d_loss.item()*(1./(i+1.))
                    g_losses[epoch] = g_losses[epoch]*(i/(i+1.)) + g_loss.item()*(1./(i+1.))

                    rec_losses[epoch] = rec_losses[epoch]*(i/(i+1.)) + rec_loss.item()*(1./(i+1.))
                    bce_losses[epoch] = bce_losses[epoch]*(i/(i+1.)) + bce_loss.item()*(1./(i+1.))
                    kldes[epoch] = kldes[epoch]*(i/(i+1.)) + kl.item()*(1./(i+1.))

                    real_scores[epoch] = real_scores[epoch]*(i/(i+1.)) + real_score.mean().item()*(1./(i+1.))
                    fake_scores[epoch] = fake_scores[epoch]*(i/(i+1.)) + fake_score.mean().item()*(1./(i+1.))

                    i += 1

                dLossVal = self.evaluation(val_loader)
                d_losses_val[epoch] = dLossVal

                # Save best Model
                if bestLoss > dLossVal:
                    bestLoss = dLossVal
                    save_arn_models(self.G, self.D, path_G, path_D)

                sys.stdout.write("\r" + 'Epoch [{:>3}/{}] | d_loss: {:.4f} | g_loss: {:.4f} ({:.2f}, {:.2f}, {:.2f}) | D(x): {:.2f} | D(G(x)): {:.2f} | d_loss_val: {:.4f}'
                                 .format(epoch+1, num_epochs, d_losses[epoch], g_losses[epoch], bce_loss.item(), rec_losses[epoch], kldes[epoch], real_scores[epoch], fake_scores[epoch], d_losses_val[epoch]))
                sys.stdout.flush()


        except KeyboardInterrupt:
            print('-' * 89)
            print('[INFO] Exiting from training early')
        print(f'\n[INFO] Training phase... Elapsed time: {(time() - start):.0f} seconds\n')

        if show_loss:
            plot_ARN_loss(d_losses[:epoch], g_losses[:epoch], d_losses_val[:epoch], bce_losses[:epoch], rec_losses[:epoch], kldes[:epoch], real_scores[:epoch], fake_scores[:epoch])

        results = {'d_losses': d_losses[:epoch], 'g_losses': g_losses[:epoch], 'rec_losses':rec_losses[:epoch],
                   'bce_losses': bce_losses[:epoch], ' kldes': kldes[:epoch], 'real_scores': real_scores[:epoch],
                   'fake_scores': fake_scores[:epoch], 'd_losses_val': d_losses_val[:epoch] }

        return results