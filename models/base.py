# models/base.py
import torch.nn as nn

class BaseAutoEncoder(nn.Module):
    def encode(self, x):
        raise NotImplementedError()

    def decode(self, z):
        raise NotImplementedError()

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)
