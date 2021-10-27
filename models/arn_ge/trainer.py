import numpy as np
import torch
import torch.nn as nn

from time import time
import sys

from models.arn_ge.model import Generator, Discriminator
from models.arn_ge.loss import Loss
from utils.util import save_arn_models
from utils.plot import plot_ARN_GE_loss


class ARN_GE(nn.Module):
    def __init__(self, params):
        super(ARN_GE, self).__init__()
        self.params = params

        self.device = self.params['device']
        self.selected_columns = self.params['selected_columns']
        self.discreteCol = self.params['discreteCol']
        self.index = self.params['index']
        self.nc = self.params['nc']

        self.D = Discriminator(nc = self.nc).to(self.device)
        self.G = Generator(device = self.device, nf_in = self.nc).to(self.device)

        self.optimizer = torch.optim.Adam([
            {'params': self.D.parameters(), 'lr': self.params['lr_D']},
            {'params': self.G.parameters(), 'lr': self.params['lr_G']}
        ])

        self.loss = Loss(self.device, self.discreteCol, self.selected_columns, self.index)

        self.temperature = 1
        self.anneal = 0.9995

        self.criterion = nn.BCELoss()


    def anneal_temp(self, lowerbound=1e-5):
        if self.temperature > lowerbound:
            self.temperature = self.temperature*self.anneal


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


    def train(self, data_loader, val_loader, path_G, path_D, batch_size = 32,
              num_epochs = 10, step = 10, lowerbnd=5e-15, num_q_steps = 1, num_g_steps = 1, show_loss = True):

        losses = np.zeros(num_epochs)
        p_true = np.zeros(num_epochs)
        p_fake = np.zeros(num_epochs)
        rec_errors = np.zeros(num_epochs)
        kldes = np.zeros(num_epochs)
        real_scores = np.zeros(num_epochs)
        fake_scores = np.zeros(num_epochs)

        d_losses_val = np.zeros(num_epochs)

        self.temperature = 1.

        total_steps = (len(data_loader.dataset) // batch_size) #*num_epochs
        print("[INFO] Starting training phase...")
        start = time()

        minloss = np.Inf

        try:

            step_count = 0
            for epoch in range(num_epochs):
                self.D.train()
                self.G.train()
                i = 0
                for batch in data_loader:

                    step_count += 1
                    batch = batch.to(self.device)

                    logits, z_mean, z_logvar, sampled_data = self.G(batch, self.discreteCol, self.selected_columns, self.index, self.temperature)

                    true_pred = self.D(batch)
                    fake_pred = self.D(sampled_data)

                    _loss, log_p_true, log_p_fake, reconstruction, kld = self.loss(batch, logits, z_mean, z_logvar, true_pred, fake_pred, step_count)
                    _loss.backward()
                    self.optimizer.step()


                    losses[epoch] = losses[epoch]*(i/(i+1.)) + _loss.item()*(1./(i+1.))
                    rec_errors[epoch] = rec_errors[epoch]*(i/(i+1.)) + reconstruction.item()*(1./(i+1.))
                    kldes[epoch] = kldes[epoch]*(i/(i+1.)) + kld.item()*(1./(i+1.))
                    p_true[epoch] = p_true[epoch]*(i/(i+1.)) + log_p_true.item()*(1./(i+1.))
                    p_fake[epoch] = p_fake[epoch]*(i/(i+1.)) + log_p_fake.item()*(1./(i+1.))

                    real_scores[epoch] = real_scores[epoch]*(i/(i+1.)) + true_pred.mean().item()*(1./(i+1.))
                    fake_scores[epoch] = fake_scores[epoch]*(i/(i+1.)) + fake_pred.mean().item()*(1./(i+1.))


                    # Anneal the temperature along with training steps
                    self.anneal_temp(lowerbnd)

                    i += 1

                if losses[epoch] < minloss:
                    minloss = losses[epoch]
                    save_arn_models(self.G, self.D, path_G, path_D)

                dLossVal = self.evaluation(val_loader)
                d_losses_val[epoch] = dLossVal

                ### Save best models?

                sys.stdout.write("\r" + 'Epoch [{:>3}/{}] | loss: {:.4f} ({:.4f}, {:.4f}, {:.4f}, {:.4f}) | D(x): {:.2f} | D(G(x)): {:.2f} |  d_loss_val: {:.4f} |'
                                 .format(epoch+1, num_epochs, losses[epoch], p_true[epoch], p_fake[epoch], rec_errors[epoch], kldes[epoch], real_scores[epoch], fake_scores[epoch], d_losses_val[epoch]))
                sys.stdout.flush()


        except KeyboardInterrupt:
            print('-' * 89)
            print('[INFO] Exiting from training early')
        print(f'\n[INFO] Training phase... Elapsed time: {(time() - start):.0f} seconds\n')

        if show_loss:
            plot_ARN_GE_loss(losses[:epoch], p_true[:epoch], p_fake[:epoch], rec_errors[:epoch], kldes[:epoch], real_scores[:epoch], fake_scores[:epoch])

        results = {'losses': losses[:epoch], 'p_true': p_true[:epoch], 'p_fake': p_fake[:epoch],
                   'rec_errors': rec_errors[:epoch], 'kld': kldes[:epoch],
                   'real_scores': real_scores[:epoch], 'fake_scores': fake_scores[:epoch],
                   'd_losses_val': d_losses_val[:epoch]}

        return results
