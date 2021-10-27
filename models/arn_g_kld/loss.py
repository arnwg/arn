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
    def __init__(self, device, discreteCol, selected_columns, index):
        super(GeneratorLoss, self).__init__()

        self.device = device
        self.discreteCol = discreteCol
        self.selected_columns = selected_columns
        self.index = index

        self.criterion = nn.BCELoss()
        self.mse = nn.MSELoss(reduction = 'mean')
        self.cel = nn.CrossEntropyLoss()


    def reconstruction(self, true_data, sampled_data):
        if len(self.index)!= 0:
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

    def forward(self, true_data, fake_preds, sampled_data, beta = 1, gamma = 1e-2):

        bs = fake_preds.size(0)
        y_fake = torch.ones(bs).to(self.device)
        log_p_y = self.criterion(fake_preds, y_fake)
        rec = self.reconstruction(true_data, sampled_data)

        return gamma*log_p_y + rec, log_p_y, rec