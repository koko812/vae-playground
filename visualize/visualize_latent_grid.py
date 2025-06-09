import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent));

import torch
import matplotlib.pyplot as plt
from models.conv_vae import ConvVAE  # 必要ならパス調整
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- モデル読み込み（latent_dim=2 用） ---
config = {
    "name": "free_bits_varbias_initialize_anealing_layer3_conv_b_32_avg_loss_5_epoch_dec_beta_1_latent2_large",  # 例: "free_bits_..._latent2_large"
    "latent_dim": 2,
    "enc_channels": [1, 64, 128],
}

model = ConvVAE(config).to(device)
model.load_state_dict(torch.load(f"trained_models/{config['name']}_vae.pt", map_location=device))
model.eval()

# --- グリッド生成 ---
grid_size = 20
z_range = 3  # [-3, 3] の範囲でグリッド
z1 = np.linspace(-z_range, z_range, grid_size)
z2 = np.linspace(-z_range, z_range, grid_size)
z_grid = torch.tensor(np.stack(np.meshgrid(z1, z2), axis=-1).reshape(-1, 2), dtype=torch.float32).to(device)

# --- 画像生成 ---
with torch.no_grad():
    recon = model.decode(z_grid).cpu()  # shape: [N², 1, 28, 28]

# --- 画像を並べる ---
canvas = np.zeros((28 * grid_size, 28 * grid_size))
for idx, img in enumerate(recon):
    i = idx // grid_size
    j = idx % grid_size
    canvas[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = img[0].numpy()

# --- 描画・保存 ---
plt.figure(figsize=(8, 8))
plt.imshow(canvas, cmap="gray")
plt.title("Latent Space Grid Decode")
plt.axis("off")
plt.tight_layout()
plt.savefig(f"samples/latent_vis/{config['name']}_grid.png")
plt.show()
