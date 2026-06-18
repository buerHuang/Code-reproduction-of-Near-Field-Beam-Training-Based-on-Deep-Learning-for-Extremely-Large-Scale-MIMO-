"""
基于 PyTorch 实现的深度神经网络（DNN），对应论文：
"Near-Field Beam Training Based on Deep Learning for Extremely Large-Scale MIMO"

主要功能：
- 输入：幅度矩阵 Y，形状为 (B, N, S)，其中
  N = 角度采样数（angle samples），S = 距离环数（distance rings）

- 主干结构（Backbone）：
  初始卷积层：3x3 卷积核，将通道数从 1 → 8
  接 10 个残差块（Residual Blocks）
    每个残差块包含：
        Conv(8→2, k=3) + BN + ReLU + Dropout
        Conv(2→8, k=3) + BN + Dropout
    残差连接（8通道恒等映射）后再经过 ReLU 激活

- 分类器（Classifier）：
  Flatten → Linear(8*N*S → Q)，其中 Q = N*S

- 输出：
  logits，形状为 (B, Q)
  （可通过 torch.softmax(logits, dim=-1) 得到概率分布）

说明：
- 本文件包含：模型定义、一个小型的合成示例数据集、以及最小化的训练循环示例。
- 实际应用中，应将合成数据集替换为你自己生成的近场信道 (Y, label) 数据。
"""
from typing import Tuple, Optional
import torch
from sympy import false

print(torch.__file__)
print(torch.__version__)
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    残差块（Residual Block），由两个 3x3 卷积组成。
    第一个卷积将通道数从 8 降至 2，第二个卷积再恢复到 8，
    符合论文中“每个残差块使用2个和8个卷积核”的描述。
    """
    def __init__(self, dropout_p: float = 0.2, use_bn: bool = True):
        super().__init__()
        self.use_bn = use_bn
        self.conv1 = nn.Conv2d(8, 2, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(2) if use_bn else nn.Identity()
        self.conv2 = nn.Conv2d(2, 8, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(8) if use_bn else nn.Identity()
        self.dropout1 = nn.Dropout2d(dropout_p) if dropout_p > 0 else nn.Identity()
        self.dropout2 = nn.Dropout2d(dropout_p) if dropout_p > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x  # 保存输入用于残差连接
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout2(out)

        # 残差连接：输出 + 输入
        out = out + identity
        out = F.relu(out, inplace=True)
        return out


class DNBTNet(nn.Module):
    """
    近场波束训练的深度学习主干网络（DNBTNet）。

    参数说明：
        N (int): 角度采样数（行数）
        S (int): 距离环数（列数）
        num_blocks (int): 残差块数量（默认10个）
        dropout_p (float): 残差块中Dropout概率
        use_bn (bool): 是否使用BatchNorm
        normalize_input (bool): 是否对输入Y进行归一化
    """
    def __init__(
        self,
        N: int,
        S: int,
        num_blocks: int = 10,
        dropout_p: float = 0.2,
        use_bn: bool = True,
        normalize_input: bool = True,
    ) -> None:
        super().__init__()
        self.N = N
        self.S = S
        self.Q = N * S  # 分类数量（角度×距离）
        self.normalize_input = normalize_input

        # 初始特征提取层：1通道 → 8通道
        self.stem = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
        )

        # 残差堆叠
        self.blocks = nn.Sequential(*[ResidualBlock(dropout_p, use_bn) for _ in range(num_blocks)])

        # 分类器
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * N * S, self.Q),
        )

        # 权重初始化
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward(self, Y: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。
        参数：
            Y: 输入张量，形状为 (B, N, S)，表示幅度矩阵；
               对未测得的码字位置可以填0。
        返回：
            logits: 输出张量，形状为 (B, Q)
        """
        if Y.dim() != 3 or Y.size(1) != self.N or Y.size(2) != self.S:
            raise ValueError(f"输入Y的形状应为 (B, {self.N}, {self.S})，实际为 {tuple(Y.shape)}")

        # 对每个样本做归一化（防止尺度差异）
        if self.normalize_input:
            Ymin = Y.amin(dim=(1, 2), keepdim=True)
            Ymax = Y.amax(dim=(1, 2), keepdim=True)
            Y = (Y - Ymin) / (Ymax - Ymin + 1e-8)

        x = Y.unsqueeze(1)  # 添加通道维度 (B, 1, N, S)
        x = self.stem(x)     # 初始卷积
        x = self.blocks(x)   # 残差堆叠
        logits = self.head(x)  # 分类器输出 (B, Q)
        return logits

    @torch.no_grad()
    def predict_proba(self, Y: torch.Tensor) -> torch.Tensor:
        """返回预测的softmax概率"""
        logits = self.forward(Y)
        return torch.softmax(logits, dim=-1)

    @torch.no_grad()
    def predict_topk(self, Y: torch.Tensor, k: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回top-k预测结果。
        返回：
            (topk_indices, topk_probs)，形状均为 (B, k)
        """
        probs = self.predict_proba(Y)
        topk_probs, topk_idx = probs.topk(k, dim=-1)
        return topk_idx, topk_probs


# -----------------------------
# 以下部分：最小训练示例（demo）
# -----------------------------
class TinySyntheticDataset(torch.utils.data.Dataset):
    """
    小型合成数据集，用于演示模型输入输出的维度与训练流程。
    实际使用时，应替换为真实的近场信道(Y, label)数据。
    """
    def __init__(self, N: int, S: int, size: int = 1024, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.N, self.S = N, S
        self.Q = N * S
        self.size = size

        # 随机生成幅度矩阵，并在真实标签位置添加一个强峰值
        Y = torch.rand(size, N, S, generator=g)
        idx = torch.randint(low=0, high=self.Q, size=(size,), generator=g)
        for i in range(size):
            n = idx[i] % N
            s = idx[i] // N
            Y[i, n, s] += 1.0  # 增强真实位置的幅度
        self.Y = Y
        self.labels = idx  # 标签范围 [0, Q)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, i: int):
        return self.Y[i], self.labels[i]


def demo_train_step():
    """
    简单的训练示例，用于验证模型是否能正常运行与反向传播。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N, S = 16, 6  # 示例尺寸，可根据码本维度调整
    model = DNBTNet(N=N, S=S, num_blocks=10, dropout_p=0, use_bn=false).to(device)

    ds = TinySyntheticDataset(N, S, size=50000)
    dl = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True)

    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for step, (Yb, yb) in enumerate(dl):
        Yb = Yb.to(device)
        yb = yb.to(device)
        logits = model(Yb)
        loss = criterion(logits, yb)

        optim.zero_grad()
        loss.backward()
        optim.step()

        # 每隔20步打印一次损失与准确率
        if step % 20 == 0:
            with torch.no_grad():
                pred = logits.argmax(dim=-1)
                acc = (pred == yb).float().mean().item()
            print(f"step={step:03d}  loss={loss.item():.4f}  acc={acc:.3f}")


if __name__ == "__main__":
    # 直接运行本文件时执行demo
    demo_train_step()
