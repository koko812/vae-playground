# Streamlit を使った VAE latent space explorer UI
import streamlit as st
st.set_page_config(layout="wide")  # ← 最初に必ず呼ぶ

import sys
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# ----- 設定 -----
CONFIG = {
    "name": "free_bits_varbias_initialize_anealing_layer3_conv_b_32_avg_loss_5_epoch_dec_beta_1_latent2_large",  # 例: "free_bits_..._latent2_large"
    "latent_dim": 2,
    "enc_channels": [1, 64, 128],
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = f"trained_models/{CONFIG['name']}_vae.pt"

# ----- モデルロード -----
@st.cache_resource
def load_model():
    sys.path.append(str(Path(__file__).resolve().parent.parent));
    from models.conv_vae import ConvVAE
    model = ConvVAE(CONFIG).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

model = load_model()

# ----- UI 描画 -----
st.title("VAE Latent Space Slider Explorer")

# スライダー配置（上部に）
st.markdown("### Move in latent space")
z1, z2 = st.columns(2)
with z1:
    val_z1 = st.slider("z1", -3.0, 3.0, 0.0, step=0.1)
with z2:
    val_z2 = st.slider("z2", -3.0, 3.0, 0.0, step=0.1)

z = torch.tensor([[val_z1, val_z2]], dtype=torch.float32).to(DEVICE)

# 2カラムレイアウト（左：散布図、右：生成画像）
col1, col2 = st.columns([1, 1], gap="large")

# 散布図
with col1:
    if os.path.exists(f"samples/latent_vis/{CONFIG['name']}_scatter.npy"):
        Z = np.load(f"samples/latent_vis/{CONFIG['name']}_scatter.npy")  # shape [N, 2]
        Y = np.load(f"samples/latent_vis/{CONFIG['name']}_labels.npy")    # shape [N]

        fig, ax = plt.subplots(figsize=(5, 5))
        scatter = ax.scatter(Z[:, 0], Z[:, 1], c=Y, cmap="tab10", s=10, alpha=0.4)
        ax.scatter([val_z1], [val_z2], color="red", s=100, marker="x")
        ax.set_title("Latent Space")
        ax.set_xlabel("z1")
        ax.set_ylabel("z2")
        st.pyplot(fig)
    else:
        st.warning("scatter.npy / labels.npy が存在しません。先に保存してください。")

# 生成画像
with col2:
    with torch.no_grad():
        img = model.decode(z).cpu()[0][0].numpy()
    fig2, ax2 = plt.subplots(figsize=(5, 5))
    ax2.imshow(img, cmap="gray")
    ax2.axis("off")
    st.pyplot(fig2)