import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvVAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.latent_dim = config["latent_dim"]
        self.enc_channels = config["enc_channels"]

        # --- Encoder ---
        encoder_layers = []
        in_channels = self.enc_channels[0]
        for out_channels in self.enc_channels[1:]:
            encoder_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1))
            encoder_layers.append(nn.ReLU())
            in_channels = out_channels
        self.encoder = nn.Sequential(*encoder_layers)

        # --- Flatten dimensions ---
        example_input = torch.zeros(1, 1, 28, 28)
        with torch.no_grad():
            h = self.encoder(example_input)
        self._enc_out_shape = h.shape[1:]
        enc_out_dim = h.numel()

        self.fc_mu = nn.Linear(enc_out_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(enc_out_dim, self.latent_dim)

        # --- Decoder ---
        self.fc_decode = nn.Linear(self.latent_dim, enc_out_dim)

        decoder_layers = []
        reversed_channels = list(reversed(self.enc_channels[1:]))

        # 1. 追加：最初に "拡張用" チャネルを増やす層
        decoder_layers.append(nn.ConvTranspose2d(reversed_channels[0], reversed_channels[0], kernel_size=3, stride=1, padding=1))
        decoder_layers.append(nn.LeakyReLU(0.2))

        # 2. 元の構成を続ける
        in_channels = reversed_channels[0]
        for out_channels in reversed_channels[1:]:
            decoder_layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1))
            decoder_layers.append(nn.LeakyReLU(0.2))
            in_channels = out_channels

        # 3. 最後の出力層
        decoder_layers.append(nn.ConvTranspose2d(in_channels, self.enc_channels[0], kernel_size=3, stride=2, padding=1, output_padding=1))
        decoder_layers.append(nn.Sigmoid())

        self.decoder = nn.Sequential(*decoder_layers)


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        h = self.encoder(x)
        h_flat = h.view(x.size(0), -1)
        mu = self.fc_mu(h_flat)
        logvar = self.fc_logvar(h_flat)
        return mu, logvar

    def decode(self, z):
        h = self.fc_decode(z).view(z.size(0), *self._enc_out_shape)
        x_hat = self.decoder(h)
        return x_hat

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar
