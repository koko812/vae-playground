import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os, json, time
from datetime import datetime
from models.conv_ae import ConvAutoEncoder
from tqdm import tqdm
import matplotlib.pyplot as plt

# ---------- Config List (プリセット) ----------
conv_configs = [
    {"name": "small", "latent_dim": 8, "enc_channels": [1, 8, 16]},
    {"name": "medium", "latent_dim": 16, "enc_channels": [1, 16, 32]},
    {"name": "large", "latent_dim": 32, "enc_channels": [1, 32, 64]},
]

# ---------- Setup ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.ToTensor()
train_loader = DataLoader(
    datasets.MNIST(root="./data", train=True, transform=transform, download=True),
    batch_size=128,
    shuffle=True
)
os.makedirs("logs", exist_ok=True)
os.makedirs("samples", exist_ok=True)
log_path = "logs/conv_ae_logs.json"
all_logs = []

# ---------- Utilities ----------
def save_reconstruction_image(model, config_name):
    model.eval()
    with torch.no_grad():
        test_batch = next(iter(train_loader))[0][:8].to(device)
        recon = model(test_batch)

    fig, axs = plt.subplots(2, 8, figsize=(12, 3))
    for i in range(8):
        axs[0, i].imshow(test_batch[i][0].cpu(), cmap="gray")
        axs[0, i].axis("off")
        axs[1, i].imshow(recon[i][0].cpu(), cmap="gray")
        axs[1, i].axis("off")

    plt.savefig(f"samples/{config_name}.png")
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

# ---------- Main Grid Loop ----------
print("\U0001F50D 実行するConvAE構成一覧:")
for i, cfg in enumerate(conv_configs, 1):
    print(f"  [{i}/{len(conv_configs)}] {cfg}")

global_start_time = time.time()
print("\n\U0001F680 ConvAEグリッドサーチ開始...\n")

for i, cfg in enumerate(conv_configs, 1):
    model = ConvAutoEncoder(cfg)
    config_name = cfg["name"]
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start_time = time.time()
    loss = train(model)
    duration = round(time.time() - start_time, 2)

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

    log = {
        "name": config_name,
        "latent_dim": cfg["latent_dim"],
        "enc_channels": cfg["enc_channels"],
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
    print(f"[{i}/{len(conv_configs)}] {config_name} → loss={log['loss']}, time={duration}s")
    save_reconstruction_image(model, config_name)

# ---------- Save Logs ----------
total_duration = round(time.time() - global_start_time, 2)
all_logs.append({"total_duration_sec": total_duration})

with open(log_path, "w") as f:
    json.dump(all_logs, f, indent=2)

print(f"\n✅ ログ書き出し完了 → {log_path}")
print(f"⏱️ 全体時間: {total_duration} 秒")
