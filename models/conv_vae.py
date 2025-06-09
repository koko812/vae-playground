import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF


def center_crop_28(x):
    return TF.center_crop(x, [28, 28])


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

        # reversed_channels = list(reversed(self.enc_channels[1:]))  # たとえば [128, 64]

        decoder_layers = []

        # 入力チャンネル数は self._enc_out_shape[0]（= encoder の出力チャンネル数）
        in_channels = self._enc_out_shape[0]

        # 1. アップサンプリング層（例：1段のみ）
        decoder_layers.append(nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=64,  # 固定でもいいし、self.enc_channels[-2] にしてもOK
            kernel_size=3, stride=2, padding=1, output_padding=1))
        decoder_layers.append(nn.LeakyReLU(0.2))

        # 2. 出力層
        decoder_layers.append(nn.ConvTranspose2d(
            in_channels=64,
            out_channels=self.enc_channels[0],  # = 1
            kernel_size=3, stride=2, padding=1, output_padding=1))
        decoder_layers.append(nn.Sigmoid())  # 出力を [0, 1] に収める


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
        x_hat = center_crop_28(x_hat)
        return x_hat

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar


# --- 初期化関数の定義 ---
def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
