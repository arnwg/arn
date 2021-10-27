import torch
import torch.nn as nn

from utils.util import generate_labels

class DiscriminatorLoss(nn.Module):
    def __init__(self, device):
        super(DiscriminatorLoss, self).__init__()

        self.device = device
        self.criterion = nn.BCELoss()

    def forward(self, true_preds, fake_preds, step):

        bs = true_preds.size(0)
        y_real = generate_labels(bs,0.3,0.7,1., step, up=True).to(self.device)
        y_fake = generate_labels(bs,0.3,0.,0.3, step, up=False).to(self.device)

        D_real_loss = self.criterion(true_preds, y_real)
        D_fake_loss = self.criterion(fake_preds, y_fake)

        return D_real_loss + D_fake_loss


class GeneratorLoss(nn.Module):
    def __init__(self, device):
        super(GeneratorLoss, self).__init__()

        self.device = device
        self.criterion = nn.BCELoss()
        self.mse = nn.MSELoss(reduction = 'mean')
        self.cel = nn.CrossEntropyLoss()


    def KLD(self,z_mean, z_logvar):
        return torch.mean(0.5 * (-0.5 * z_logvar + torch.exp(0.5 * z_logvar) + z_mean ** 2))

    def forward(self, true_data, fake_preds, sampled_data, z_mean, z_logvar, beta = 1, gamma = 1e-2):

        bs = fake_preds.size(0)

        y_fake = torch.ones(bs).to(self.device)
        log_p_y = self.criterion(fake_preds, y_fake)

        rec = self.mse(true_data, sampled_data)

        kld = self.KLD(z_mean, z_logvar)

        return gamma*log_p_y + rec + beta*kld, log_p_y, rec, kld

