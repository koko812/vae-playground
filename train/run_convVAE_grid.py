import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os, json, time
from datetime import datetime
from models.conv_ae import ConvAutoEncoder
from models.conv_vae import ConvVAE
from tqdm import tqdm
import matplotlib.pyplot as plt
from models.conv_vae import init_weights

# ---------- Config List (プリセット) ----------
conv_configs = [
    {
        "name": "free_bits_varbias_initialize_anealing_layer3_conv_b_32_avg_loss_5_epoch_dec_beta_1_latent2_large",
        "latent_dim": 2,
        "enc_channels": [1, 64, 128],
    },
    {
        "name": "free_bits_varbias_initialize_anealing_layer3_conv_b_32_avg_loss_5_epoch_dec_beta_1_latent2_medium",
        "latent_dim": 2,
        "enc_channels": [1, 16, 64],
    },
]

# ---------- Setup ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.ToTensor()
train_loader = DataLoader(
    datasets.MNIST(root="./data", train=True, transform=transform, download=True),
    batch_size=32,
    shuffle=True,
)
os.makedirs("logs", exist_ok=True)
os.makedirs("samples", exist_ok=True)
log_path = "logs/free_bits_varbias_initialize_anealing_layer3_conv_b_1_avg_loss_5_epoch_dec_plus_one_beta_1_latent_2_conv_ae_logs.json"
all_logs = []


# ---------- Utilities ----------
def save_reconstruction_image(model, config_name):
    model.eval()
    with torch.no_grad():
        test_batch = next(iter(train_loader))[0][:8].to(device)
        recon, *_ = model(test_batch)  # ← 出力が tuple の場合

    fig, axs = plt.subplots(2, 8, figsize=(12, 3))
    for i in range(8):
        axs[0, i].imshow(test_batch[i][0].cpu(), cmap="gray")
        axs[0, i].axis("off")
        print(recon[i][0].shape)
        axs[1, i].imshow(recon[i][0].cpu().numpy(), cmap="gray")
        axs[1, i].axis("off")

    plt.savefig(f"samples/conv_VAE/{config_name}.png")
    plt.close()

def compute_beta(step, beta_max=0.5, warmup_steps=1000):
    return min(beta_max, step / warmup_steps * beta_max)

def loss_fn_vae(x, x_hat, mu, logvar, beta=0.5, free_bits_eps=0.1):
    batch_size, latent_dim = mu.size()

    # ❶ 再構成損失（1サンプル平均）
    recon_loss = F.binary_cross_entropy(x_hat, x, reduction="sum") / batch_size

    # ❷ KL per dim per sample → shape: [batch, latent_dim]
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # shape: [batch, latent_dim]

    # ❸ 各次元の平均 KL（1次元あたり）→ shape: [latent_dim]
    kl_per_dim_mean = kl_per_dim.mean(dim=0)

    # ❹ Free bits を適用：min ε（例：0.1）を下限とする
    kl_per_dim_clipped = torch.clamp(kl_per_dim_mean, min=free_bits_eps)

    # ❺ 全体の KL（1サンプルあたり）
    kl_div = kl_per_dim_clipped.sum()

    # ❻ 合計ロス（1サンプルあたり）
    return recon_loss + beta * kl_div, recon_loss, kl_div



def train(model, dataloader, optimizer, device, global_step, warmup_steps=1000, beta_max=4.0, free_bits_eps=0.1):
    model.train()
    total_loss, total_recon, total_kl = 0, 0, 0

    for x, _ in tqdm(dataloader):
        x = x.to(device)
        optimizer.zero_grad()

        # forward
        x_hat, mu, logvar = model(x)

        # 💡 beta を更新
        beta = compute_beta(global_step, beta_max=beta_max, warmup_steps=warmup_steps)

        # 💡 Free Bits 対応 loss
        loss, recon_loss, kl_div = loss_fn_vae(x, x_hat, mu, logvar, beta=beta, free_bits_eps=free_bits_eps)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_div.item()
        global_step += 1

    return (
        total_loss / len(dataloader.dataset),
        total_recon / len(dataloader.dataset),
        total_kl / len(dataloader.dataset),
    )



# ---------- Main Grid Loop ----------
print("\U0001f50d 実行するConvVAE構成一覧:")
for i, cfg in enumerate(conv_configs, 1):
    print(f"  [{i}/{len(conv_configs)}] {cfg}")

global_start_time = time.time()
print("\n\U0001f680 ConvVAEグリッドサーチ開始...\n")


EPOCHS = 10  # ← お好きなエポック数に変更可

for i, cfg in enumerate(conv_configs, 1):
    model = ConvVAE(cfg).to(device)
    model.apply(init_weights)

    config_name = cfg["name"]
    optimizer = optim.Adam(model.parameters(), lr=1e-3)  # ← 必要に応じて学習率も調整

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start_time = time.time()

    global_step = 0  # 👈 新しく追加



    # --- エポックループ ---
    total_loss, total_recon, total_kl = 0, 0, 0
    for epoch in range(EPOCHS):
        loss, recon, kl = train(model, train_loader, optimizer, device, global_step)
        global_step += len(train_loader)  # 👈 各epoch後に step 更新
    
        total_loss += loss
        total_recon += recon
        total_kl += kl
        print(
            f"  Epoch {epoch + 1}/{EPOCHS} | loss={loss:.2f}, recon={recon:.2f}, kl={kl:.2f}"
        )

    # 平均を取る
    avg_loss = total_loss / EPOCHS
    avg_recon = total_recon / EPOCHS
    avg_kl = total_kl / EPOCHS

    duration = round(time.time() - start_time, 2)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        used_memory = torch.cuda.memory_allocated(0) // (1024**2)
        peak_memory = torch.cuda.max_memory_allocated(0) // (1024**2)
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
        "loss": round(avg_loss, 6),
        "recon_loss": round(avg_recon, 6),
        "kl_div": round(avg_kl, 6),
        "duration_sec": duration,
        "timestamp": datetime.now().isoformat(),
        "gpu": {
            "device_name": device_name,
            "device_count": device_count,
            "used_memory_MB": used_memory,
            "peak_memory_MB": peak_memory,
        },
    }

    all_logs.append(log)
    print(
        f"[{i}/{len(conv_configs)}] {config_name} → avg_loss={log['loss']}, time={duration}s"
    )
    save_reconstruction_image(model, config_name)
    torch.save(model.state_dict(), f"trained_models/{config_name}_vae.pt")



# ---------- Save Logs ----------
total_duration = round(time.time() - global_start_time, 2)
all_logs.append({"total_duration_sec": total_duration})

with open(log_path, "w") as f:
    json.dump(all_logs, f, indent=2)

print(f"\n✅ ログ書き出し完了 → {log_path}")
print(f"⏱️ 全体時間: {total_duration} 秒")
