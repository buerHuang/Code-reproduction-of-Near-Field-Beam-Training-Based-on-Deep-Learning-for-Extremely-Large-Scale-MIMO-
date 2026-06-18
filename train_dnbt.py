import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from net import DNBTNet   # 你保存的网络定义文件
from dataset_generator import NearFieldBeamDataset  # 你生成数据集的文件
import matplotlib.pyplot as plt
import numpy as np
import time

# =========================
# 训练主程序
# =========================
def train_dnbt(
    epochs=10,
    batch_size=100,
    lr=1e-3,
    save_model=True,
    model_path="dnbt_trained.pth",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # === 1️⃣ 加载数据集 ===
    ds = NearFieldBeamDataset(Nt=256, fc=100e9, N_samples=16000,
                              D_interval=16, delta_n=4, SNR_dB=0)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    # === 2️⃣ 定义模型、优化器、损失函数 ===
    model = DNBTNet(N=ds.D, S=ds.S, num_blocks=10,
                    dropout_p=0.2, use_bn=True, normalize_input=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"Model initialized with D={ds.D}, S={ds.S}, Q={ds.D*ds.S}")
    print(f"Dataset size={len(ds)}, Batch={batch_size}, Epochs={epochs}")

    # === 3️⃣ 训练循环 ===
    losses, accuracies = [], []
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        total_loss, total_correct, total_samples = 0.0, 0, 0

        for Yb, yb in dl:
            Yb = Yb.to(device)
            yb = yb.to(device)

            logits = model(Yb)
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = logits.argmax(dim=-1)
            total_correct += (pred == yb).sum().item()
            total_loss += loss.item() * Yb.size(0)
            total_samples += Yb.size(0)

        avg_loss = total_loss / total_samples
        acc = total_correct / total_samples
        losses.append(avg_loss)
        accuracies.append(acc)

        print(f"[Epoch {epoch+1:02d}/{epochs}] Loss={avg_loss:.4f}  Acc={acc:.3f}")

    elapsed = time.time() - start_time
    print(f"Training complete in {elapsed/60:.2f} min")

    # === 4️⃣ 可视化训练过程 ===
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(losses, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('CrossEntropy Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1,2,2)
    plt.plot(accuracies, label='Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # === 5️⃣ 保存模型 ===
    if save_model:
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    return model, losses, accuracies


if __name__ == "__main__":
    model, losses, accuracies = train_dnbt(
        epochs=100,
        batch_size=256,
        lr=1e-3,
        save_model=True
    )
