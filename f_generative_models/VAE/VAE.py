import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

class VAE(nn.Module):
    def __init__(self, input_dim, h_dim, z_dim):
        super().__init__()

        # encodeur
        self.fc1 = nn.Linear(input_dim, h_dim)
        self.fc21 = nn.Linear(h_dim, z_dim)
        self.fc22 = nn.Linear(h_dim, z_dim)

        # decodeur
        self.fc3 = nn.Linear(z_dim, h_dim)
        self.fc4 = nn.Linear(h_dim, input_dim)


    def encode(self, x):
        h = F.relu(self.fc1(x))
        mu = self.fc21(h)
        log_var = self.fc22(h)
        return mu, log_var

    def sample(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        distr = Normal(mu, std)
        z = distr.rsample()
        return z

    def decode(self, z):
        h = F.relu(self.fc3(z))
        x = torch.sigmoid(self.fc4(h))
        return x

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.sample(mu, log_var)
        return self.decode(z), mu, log_var

def loss_function(x_hat, x, mu, log_var):
    bce = F.binary_cross_entropy(x_hat, x, reduction='sum')
    kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return bce + kl

# on peut utiliser rsample
# .KL