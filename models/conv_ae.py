from models.base import BaseAutoEncoder
import torch.nn as nn

class ConvAutoEncoder(BaseAutoEncoder):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.latent_dim = config["latent_dim"]
        channels = config["enc_channels"]

        # Encoder
        enc = []
        for in_ch, out_ch in zip(channels[:-1], channels[1:]):
            enc.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1))
            enc.append(nn.ReLU())
        self.encoder_cnn = nn.Sequential(*enc)

        # Flatten → latent
        self.flatten = nn.Flatten()
        self.to_latent = nn.Linear(channels[-1]*7*7, self.latent_dim)

        # Decoder
        self.from_latent = nn.Linear(self.latent_dim, channels[-1]*7*7)
        dec = []
        for in_ch, out_ch in zip(reversed(channels[1:]), reversed(channels[:-1])):
            dec.append(nn.ConvTranspose2d(in_ch, out_ch, 3, stride=2, padding=1, output_padding=1))
            dec.append(nn.ReLU())
        dec[-1] = nn.Sigmoid()  # 最後だけ出力化
        self.decoder_cnn = nn.Sequential(*dec)

    def encode(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        return self.to_latent(x)

    def decode(self, z):
        x = self.from_latent(z).view(-1, self.config["enc_channels"][-1], 7, 7)
        return self.decoder_cnn(x)
