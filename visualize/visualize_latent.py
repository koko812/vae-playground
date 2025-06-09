import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent));

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.conv_vae import ConvVAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- モデル設定とロード ---
config = {
    "name": "free_bits_varbias_initialize_anealing_layer3_conv_b_32_avg_loss_5_epoch_dec_beta_1_latent2_large",
    "latent_dim": 2,
    "enc_channels": [1, 64, 128],
}
model = ConvVAE(config).to(device)
model.load_state_dict(torch.load(f"trained_models/{config['name']}_vae.pt", map_location=device))
model.eval()

# --- データローダー（同じ前処理）---
transform = transforms.ToTensor()
test_loader = DataLoader(
    datasets.MNIST(root="./data", train=False, transform=transform, download=True),
    batch_size=256, shuffle=False
)

# --- 潜在変数を取得 ---
latents, labels = [], []

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        mu, logvar = model.encode(x)
        z = mu  # ※ノイズなしで構造観察に向いてる
        latents.append(z.cpu())
        labels.append(y)

Z = torch.cat(latents)
Y = torch.cat(labels)
# visualize_latent.py の最後などに追加
np.save(f"samples/latent_vis/{config['name']}_scatter.npy", Z.numpy())
np.save(f"samples/latent_vis/{config['name']}_labels.npy", Y.numpy())


# --- プロット ---
plt.figure(figsize=(8, 6))
sc = plt.scatter(Z[:, 0], Z[:, 1], c=Y, cmap='tab10', s=10, alpha=0.7)
plt.colorbar(sc, ticks=range(10))
plt.title(f"Latent Space: {config['name']}")
plt.xlabel("z1")
plt.ylabel("z2")
plt.grid(True)
plt.savefig(f"samples/latent_vis/{config['name']}_latent.png")
plt.show()
