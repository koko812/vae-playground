import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent));

import torch
import numpy as np
from PIL import Image
import os
from models.conv_vae import ConvVAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 構成とモデル読み込み ---
config = {
    "name": "free_bits_varbias_initialize_anealing_layer3_conv_b_32_avg_loss_5_epoch_dec_beta_1_latent2_large",  # 例: "free_bits_..._latent2_large"
    "latent_dim": 2,
    "enc_channels": [1, 64, 128],
}

model = ConvVAE(config).to(device)
model.load_state_dict(torch.load(f"trained_models/{config['name']}_vae.pt", map_location=device))
model.eval()

# --- z_start (7), z_end (0) の座標（手動で指定）
# さっきの grid 可視化からだいたいの位置を見てセット（調整可能）
z_start = torch.tensor([[0, 2]], dtype=torch.float32).to(device)  # 7っぽい位置
z_end   = torch.tensor([[0, -2]], dtype=torch.float32).to(device)  # 0っぽい位置

# --- 補間 & 画像生成 ---
steps = 40
images = []
with torch.no_grad():
    for t in np.linspace(0, 1, steps):
        z = (1 - t) * z_start + t * z_end
        img = model.decode(z).cpu()[0][0].numpy() * 255  # shape: [1, 1, 28, 28]
        img_pil = Image.fromarray(img.astype(np.uint8), mode="L")
        images.append(img_pil.resize((96, 96), resample=Image.NEAREST))  # 拡大して見やすく

# --- 保存先ディレクトリ ---
out_path = f"samples/latent_vis/{config['name']}_walk_7_to_0.gif"
os.makedirs("samples/latent_vis", exist_ok=True)

# --- GIF 保存 ---
images[0].save(out_path, save_all=True, append_images=images[1:], duration=80, loop=0)
print(f"✅ Saved: {out_path}")
