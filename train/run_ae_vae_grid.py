import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os, json, time
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from models.ae import AutoEncoder
from models.vae import VariationalAutoEncoder

# デバイス
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# データ
transform = transforms.ToTensor()
train_loader = DataLoader(
    datasets.MNIST(root="./data", train=True, transform=transform, download=True),
    batch_size=128,
    shuffle=True
)

# ログ保存準備
os.makedirs("logs", exist_ok=True)
log_path = "logs/ae_vae_logs.json"
all_logs = []

latent_dims = [2, 8]
depths = [1, 2]
activations = [nn.ReLU, nn.LeakyReLU]
modes = ["AE", "VAE"]

def build_ae(latent_dim, depth, act_fn):
    enc = [nn.Flatten(), nn.Linear(28*28, 128), act_fn()]
    for _ in range(depth - 1):
        enc += [nn.Linear(128, 128), act_fn()]
    enc += [nn.Linear(128, latent_dim)]

    dec = [nn.Linear(latent_dim, 128), act_fn()]
    for _ in range(depth - 1):
        dec += [nn.Linear(128, 128), act_fn()]
    dec += [nn.Linear(128, 28*28), nn.Sigmoid(), nn.Unflatten(1, (1, 28, 28))]

    return AutoEncoder(nn.Sequential(*enc), nn.Sequential(*dec))

def build_vae(latent_dim, depth, act_fn):
    enc = [nn.Flatten(), nn.Linear(28*28, 128), act_fn()]
    for _ in range(depth - 1):
        enc += [nn.Linear(128, 128), act_fn()]

    dec = [nn.Linear(latent_dim, 128), act_fn()]
    for _ in range(depth - 1):
        dec += [nn.Linear(128, 128), act_fn()]
    dec += [nn.Linear(128, 28*28), nn.Sigmoid(), nn.Unflatten(1, (1, 28, 28))]

    return VariationalAutoEncoder(nn.Sequential(*enc), nn.Sequential(*dec), latent_dim)

def train(model, mode="AE"):
    model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    total_loss = 0
    for x, _ in tqdm(train_loader, desc=f"{mode} Training", leave=False):
        x = x.to(device)
        optimizer.zero_grad()
        if mode == "AE":
            x_hat = model(x)
            loss = criterion(x_hat, x)
        else:  # VAE
            x_hat, mu, logvar = model(x)
            bce = criterion(x_hat, x)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = bce + kl
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss

def save_image(model, mode, latent_dim, depth, act_name):
    model.eval()
    os.makedirs("samples", exist_ok=True)
    with torch.no_grad():
        test_batch = next(iter(train_loader))[0][:8].to(device)
        if mode == "AE":
            recon = model(test_batch)
        else:
            recon, _, _ = model(test_batch)

    fig, axs = plt.subplots(2, 8, figsize=(12, 3))
    for i in range(8):
        axs[0, i].imshow(test_batch[i][0].cpu(), cmap="gray")
        axs[0, i].axis("off")
        axs[1, i].imshow(recon[i][0].cpu(), cmap="gray")
        axs[1, i].axis("off")

    title = f"{mode}_latent{latent_dim}_depth{depth}_{act_name}.png"
    plt.savefig(f"samples/{title}")
    plt.close()

# グリッドサーチ
from itertools import product
start_all = time.time()
grid = list(product(modes, latent_dims, depths, activations))
print(f"🎛️ 実行総数: {len(grid)}")

for i, (mode, latent_dim, depth, act_fn) in enumerate(grid, 1):
    print(f"\n[{i}/{len(grid)}] {mode} latent={latent_dim}, depth={depth}, act={act_fn.__name__}")

    model = build_ae(latent_dim, depth, act_fn) if mode == "AE" else build_vae(latent_dim, depth, act_fn)

    # GPUメモリリセット
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start_time = time.time()
    loss = train(model, mode=mode)
    duration = round(time.time() - start_time, 2)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_mem = torch.cuda.max_memory_allocated(0) // (1024 ** 2)
        device_name = torch.cuda.get_device_name(0)
    else:
        peak_mem = 0
        device_name = "CPU"

    log = {
        "mode": mode,
        "latent_dim": latent_dim,
        "depth": depth,
        "activation": act_fn.__name__,
        "loss": round(loss, 4),
        "duration_sec": duration,
        "timestamp": datetime.now().isoformat(),
        "device": device_name,
        "peak_memory_MB": peak_mem
    }
    all_logs.append(log)
    save_image(model, mode, latent_dim, depth, act_fn.__name__)

# 結果保存
total_duration = round(time.time() - start_all, 2)
all_logs.append({"total_duration_sec": total_duration})
with open(log_path, "w") as f:
    json.dump(all_logs, f, indent=2)

print(f"\n✅ 全ログを {log_path} に保存")
print(f"⏱️ 全体時間: {total_duration} 秒")
