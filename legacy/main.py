import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os, json, time
from datetime import datetime
from model import AutoEncoder
import itertools
from tqdm import tqdm
import matplotlib.pyplot as plt

# デバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# データセット
transform = transforms.ToTensor()
train_loader = DataLoader(
    datasets.MNIST(root="./data", train=True, transform=transform, download=True),
    batch_size=128,
    shuffle=True
)

# ログファイル準備
os.makedirs("logs", exist_ok=True)
log_path = "logs/ae_logs.json"
all_logs = []

# グリッドサーチ設定
latent_dims = [2, 8, 32]
depths = [1, 2]
activations = [nn.ReLU, nn.LeakyReLU]

def build_autoencoder(latent_dim, depth, act_fn):
    enc = [nn.Flatten(), nn.Linear(28*28, 128), act_fn()]
    for _ in range(depth - 1):
        enc += [nn.Linear(128, 128), act_fn()]
    enc += [nn.Linear(128, latent_dim)]

    dec = [nn.Linear(latent_dim, 128), act_fn()]
    for _ in range(depth - 1):
        dec += [nn.Linear(128, 128), act_fn()]
    dec += [nn.Linear(128, 28*28), nn.Sigmoid(), nn.Unflatten(1, (1, 28, 28))]

    return AutoEncoder(nn.Sequential(*enc), nn.Sequential(*dec))

def save_reconstruction_image(model, latent_dim, depth, act_name, filename=None):
    model.eval()
    os.makedirs("samples", exist_ok=True)

    with torch.no_grad():
        test_batch = next(iter(train_loader))[0][:8].to(device)
        recon = model(test_batch)

    fig, axs = plt.subplots(2, 8, figsize=(12, 3))
    for i in range(8):
        axs[0, i].imshow(test_batch[i][0].cpu(), cmap="gray")
        axs[0, i].axis("off")
        axs[1, i].imshow(recon[i][0].cpu(), cmap="gray")
        axs[1, i].axis("off")

    title = f"latent{latent_dim}_depth{depth}_{act_name}"
    path = filename or f"samples/{title}.png"
    plt.savefig(path)
    plt.close()

def train(model):
    model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    total_loss = 0
    for x, _ in tqdm(train_loader, desc="Training", leave=False):
        x = x.to(device)
        optimizer.zero_grad()
        x_hat = model(x)
        loss = criterion(x_hat, x)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss

# 実行開始時刻
global_start_time = time.time()

# グリッド条件
grid = list(itertools.product(latent_dims, depths, activations))
total = len(grid)

print("🔍 実行するグリッドサーチ条件一覧:")
for i, (ld, d, act) in enumerate(grid, 1):
    print(f"  [{i}/{total}] latent_dim={ld}, depth={d}, activation={act.__name__}")
print("\n🚀 グリッドサーチ開始...\n")

for i, (latent_dim, depth, act_fn) in enumerate(grid, 1):
    model = build_autoencoder(latent_dim, depth, act_fn)

    # 個別の試行タイマー開始
    start_time = time.time()

    # GPUメモリリセット（初期）
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    loss = train(model)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        used_memory = torch.cuda.memory_allocated(0) // (1024 ** 2)
        peak_memory = torch.cuda.max_memory_allocated(0) // (1024 ** 2)
        device_name = torch.cuda.get_device_name(0)
        device_count = torch.cuda.device_count()
    else:
        used_memory = 0
        peak_memory = 0
        device_name = "CPU"
        device_count = 0

    duration = round(time.time() - start_time, 2)

    log = {
        "latent_dim": latent_dim,
        "depth": depth,
        "activation": act_fn.__name__,
        "loss": round(loss, 6),
        "duration_sec": duration,
        "timestamp": datetime.now().isoformat(),
        "gpu": {
            "device_name": device_name,
            "device_count": device_count,
            "used_memory_MB": used_memory,
            "peak_memory_MB": peak_memory
        }
    }
    all_logs.append(log)
    print(f"[{i}/{total}] latent={latent_dim}, depth={depth}, act={act_fn.__name__} → loss={log['loss']}, time={duration}s")
    save_reconstruction_image(model, latent_dim, depth, act_fn.__name__)

# 全体実行時間を追加
total_duration = round(time.time() - global_start_time, 2)
all_logs.append({"total_duration_sec": total_duration})

# JSON保存
with open(log_path, "w") as f:
    json.dump(all_logs, f, indent=2)

print(f"\n✅ ログを書き出しました → {log_path}")
print(f"🕒 全体実行時間: {total_duration} 秒")
