import torch
import torch.nn as nn

from utils.util import generate_labels

class Loss(nn.Module):
    def __init__(self, device, discreteCol, selected_columns, index):
        super(Loss, self).__init__()
        self.device = device

        self.discreteCol = discreteCol
        self.selected_columns = selected_columns
        self.index = index

        self.bce = nn.BCELoss()
        self.mse = nn.MSELoss()
        self.cel = nn.CrossEntropyLoss()


    def KLD(self,z_mean, z_logvar):
        return torch.mean(0.5 * (-0.5 * z_logvar + torch.exp(0.5 * z_logvar) + z_mean ** 2))


    def reconstruction(self, true_data, sampled_data):
        if len(self.index) != 0:
            g1 = self.mse(true_data[:, self.index], sampled_data[:, self.index])
        else:
            g1 = 0
        g2 = 0
        for name in self.discreteCol:
            y = true_data[:, self.selected_columns[name]]
            y_p = sampled_data[:, self.selected_columns[name]]
            g2 += self.mse(y_p, y)
        g2 /= len(self.discreteCol)
        return g1 + 0.5 * g2

    def forward(self, true_data, sampled_data, z_mean, z_logvar, true_pred, fake_pred, step, alpha = 20):
        bs = true_pred.size(0)

        y_real = generate_labels(bs,0.3,0.7,1., step, up=True).to(self.device)
        y_fake = generate_labels(bs,0.3,0.,0.3, step, up=False).to(self.device)

        log_p_true = self.bce(true_pred, y_real)
        log_p_fake = self.bce(fake_pred, y_fake)

        reconstruction = self.reconstruction(true_data, sampled_data)
        kld = self.KLD(z_mean, z_logvar)

        return log_p_true + log_p_fake + alpha*reconstruction + kld, log_p_true, log_p_fake, reconstruction, kld

