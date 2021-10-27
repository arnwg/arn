import torch
import torch.nn as nn

from utils.util import gumbel_softmax

class Generator(nn.Module):
    def __init__(self, device, nf_in = 121, nf_out = 32, z_dim = 16):
        super(Generator, self).__init__()

        self.device = device
        self.nf_in = nf_in
        self.nf_out = nf_out
        self.z_dim = z_dim

        self.encoder = nn.Sequential(
            nn.Linear(self.nf_in, self.nf_out * 2),
            nn.BatchNorm1d(self.nf_out * 2, track_running_stats = False),
            nn.LeakyReLU(0.2),

            nn.Linear(self.nf_out * 2, self.nf_out),
            nn.BatchNorm1d(self.nf_out, track_running_stats = False),
            nn.LeakyReLU(0.2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.nf_out, self.nf_out * 2),
            nn.BatchNorm1d(self.nf_out * 2, track_running_stats = False),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(self.nf_out * 2, self.nf_in)
        )

        self.sigmoid = nn.Sigmoid()


        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def gumbel(self, logits, t):
        return gumbel_softmax(logits, self.device, t)

    def forward(self, x, discreteCol, selected_columns, index, t=1):
        enc = self.encoder(x)
        logits = self.decoder(enc)

        sampled_data = logits.clone()
        if len(index)!= 0:
            sampled_data[:, index] = self.sigmoid(logits[:, index])

        for name in discreteCol:
            sampled_data[:, selected_columns[name]] = self.gumbel(logits[:, selected_columns[name]], t)

        return logits, sampled_data

class Discriminator(nn.Module):

    def __init__(self, nc = 121, nc_out = 16, nout = 128):
        super(Discriminator, self).__init__()

        self.nc = nc
        self.nc_out = nc_out
        self.nout = nout

        self.main = nn.Sequential(
            # features extractor
            nn.Linear(self.nc, self.nout),
            nn.BatchNorm1d(self.nout, track_running_stats = False),
            nn.LeakyReLU(0.2),

            nn.Linear(self.nout, self.nout * 2),
            nn.BatchNorm1d(self.nout * 2, track_running_stats = False),
            nn.LeakyReLU(0.2),

            nn.Linear(self.nout * 2, self.nout * 4),
            nn.BatchNorm1d(self.nout * 4, track_running_stats = False),
            nn.LeakyReLU(0.2),

            # classifier
            nn.Linear(self.nout * 4, self.nout),
            nn.BatchNorm1d(self.nout, track_running_stats = False),
            nn.ReLU(),

            nn.Linear(self.nout, self.nc_out * 4),
            nn.BatchNorm1d(self.nc_out * 4, track_running_stats = False),
            nn.ReLU(),

            nn.Dropout(0.2),
            nn.Linear(self.nc_out * 4, self.nc_out * 2),
            nn.ReLU(),

            nn.Dropout(0.2),
            nn.Linear(self.nc_out * 2, self.nc_out),
            nn.ReLU(),

            nn.Dropout(0.2),
            nn.Linear(self.nc_out, 1),
            nn.Sigmoid()
        )

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        x = self.main(x)
        return x.flatten()