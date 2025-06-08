import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

from model import AutoEncoder  # 別ファイルに分けたモデル定義

# デバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# データセット準備（MNIST）
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# モデル初期化
model = AutoEncoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 学習ループ
print("Training AutoEncoder...")
for epoch in range(5):
    model.train()
    total_loss = 0
    for x, _ in train_loader:
        x = x.to(device)
        optimizer.zero_grad()
        x_hat = model(x)
        loss = criterion(x_hat, x)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# テスト画像の再構成と保存
os.makedirs("samples", exist_ok=True)
model.eval()
with torch.no_grad():
    test_img = next(iter(train_loader))[0][:8].to(device)
    recon_img = model(test_img)

    # 並べて保存
    fig, axs = plt.subplots(2, 8, figsize=(12, 3))
    for i in range(8):
        axs[0, i].imshow(test_img[i][0].cpu(), cmap="gray")
        axs[0, i].axis("off")
        axs[1, i].imshow(recon_img[i][0].cpu(), cmap="gray")
        axs[1, i].axis("off")
    plt.savefig("samples/reconstruction.png")
    print("✅ 再構成画像を samples/reconstruction.png に保存しました！")
