import numpy as np
import torch
from torch.utils.data import Dataset
from near_field_channel import near_field_channel
from polar_codebook import polar_codebook
from training_near_exhaustive import training_near_exhaustive
from codeword_sampling import sample_codewords
import os


class NearFieldBeamDataset(Dataset):
    """
    近场波束训练数据集，支持 (r, θ) 随机采样缓存。
    若缓存文件存在但样本数量不匹配，将自动重新生成。
    """

    def __init__(self,
                 Nt=256,
                 fc=100e9,
                 B=0,
                 d=None,
                 M=1,
                 SNR_dB=-10,
                 N_samples=1024,
                 D_interval=8,
                 delta_n=1,
                 Rmin=10, Rmax=40,
                 theta_min=-np.pi/3, theta_max=np.pi/3,
                 normalize=True,
                 seed=0,
                 param_cache="NF_random_params.npz"):
        super().__init__()
        np.random.seed(seed)

        # ===============================
        # 1️⃣ 加载或生成随机采样参数 (r, θ)
        # ===============================
        regenerate = False  # 是否需要重新生成缓存
        if os.path.exists(param_cache):
            try:
                data = np.load(param_cache)
                theta_all = data["theta_all"]
                r_all = data["r_all"]

                # 自动检查样本数是否匹配
                if len(theta_all) != N_samples or len(r_all) != N_samples:
                    print(f"⚠️ 缓存文件 {param_cache} 的样本数 ({len(theta_all)}) 与当前 N_samples={N_samples} 不匹配。")
                    regenerate = True
                else:
                    self.theta_all = theta_all
                    self.r_all = r_all
                    print(f"🔹 从缓存加载采样参数: {param_cache} ({N_samples} 个样本)")
            except Exception as e:
                print(f"⚠️ 无法读取缓存文件 {param_cache}: {e}")
                regenerate = True
        else:
            regenerate = True

        if regenerate:
            print(f"⚙️ 正在重新生成随机采样参数 (N={N_samples})...")
            self.theta_all = np.random.uniform(theta_min, theta_max, N_samples)
            self.r_all = np.random.uniform(Rmin, Rmax, N_samples)
            np.savez(param_cache, theta_all=self.theta_all, r_all=self.r_all)
            print(f"✅ 已生成并缓存新的随机采样参数 → {param_cache}")

        # ===============================
        # 2️⃣ 初始化系统参数
        # ===============================
        c = 3e8
        if d is None:
            d = (c / fc) / 2

        self.Nt = Nt
        self.fc = fc
        self.B = B
        self.d = d
        self.M = M
        self.SNR_dB = SNR_dB
        self.SNR = 10 ** (SNR_dB / 10.0)
        self.Rmin, self.Rmax = Rmin, Rmax
        self.theta_min, self.theta_max = theta_min, theta_max
        self.N_samples = N_samples
        self.normalize = normalize

        # ===============================
        # 3️⃣ 极化域码本与采样索引
        # ===============================
        self.w, self.sin_theta_list, self.r_list = polar_codebook(
            Nt, s=1, d=d, lambda_c=c / fc,
            delta=1.2, rho_min=3,
            theta_min=theta_min, theta_max=theta_max
        )
        self.D, self.S = self.w.shape[1], self.w.shape[2]

        _, self.sampled_indices, self.sampled_flat_indices, self.mask = sample_codewords(
            self.w, D_interval=D_interval, delta_n=delta_n, return_mask=True
        )

        np.save('sampled_indices.npy', self.sampled_indices, allow_pickle=True)
        np.save('mask.npy', self.mask)
        print("✅ 已保存采样索引 sampled_indices.npy 与掩码 mask.npy")

        # ===============================
        # 4️⃣ 生成训练样本
        # ===============================
        self.Y_all, self.labels_all = self._generate_all_samples()

    def _generate_all_samples(self):
        Y_all = np.zeros((self.N_samples, self.D, self.S))
        labels_all = np.zeros(self.N_samples, dtype=int)

        # 添加多信噪比范围
        SNR_dB_range = np.linspace(-15, 30, 9)  # 例如 -10, -5, 0, 5, 10, 15, 20

        for i in range(self.N_samples):
            r = self.r_all[i]
            theta = self.theta_all[i]
            print(f"Sample {i}: r={r:.2f} m, θ={np.degrees(theta):.2f}°")

            # ✅ 随机选取信噪比
            SNR_dB_i = np.random.choice(SNR_dB_range)
            SNR_i = 10 ** (SNR_dB_i / 10.0)
            noise_power = 10 ** (-SNR_dB_i / 10.0)

            H, hc = near_field_channel(self.Nt, self.d, self.fc, self.B, self.M, r, theta)

            Y = np.zeros((self.D, self.S))
            for s, idx_list in enumerate(self.sampled_indices):
                for n in idx_list:
                    signal = np.vdot(hc, self.w[:, n, s])
                    signal_power = np.mean(np.abs(signal) ** 2)
                    noise_power = signal_power / (10 ** ((SNR_dB_i) / 10.0))

                    noise = np.sqrt(noise_power / 2) * (np.random.randn() + 1j * np.random.randn())
                    Y[n, s] = np.abs(signal + noise)

            # === 归一化
            if self.normalize:
                Ymin, Ymax = Y.min(), Y.max()
                Y = (Y - Ymin) / (Ymax - Ymin)

            # === 标签：穷举最优波束索引
            rate, _, _, _, idx_max = training_near_exhaustive(
                self.D, self.S, hc, self.w, SNR_i, SNR_dB_i,
                Q=16, mode=0, M=self.M, overhead_max=self.D * self.S,
                sin_theta_near_nb=self.sin_theta_list, r_near_nb=self.r_list
            )
            Y_all[i] = Y
            labels_all[i] = int(idx_max)

        return Y_all, labels_all

    def __len__(self):
        return self.N_samples

    def __getitem__(self, idx):
        Y = torch.tensor(self.Y_all[idx], dtype=torch.float32)
        label = torch.tensor(self.labels_all[idx], dtype=torch.long)
        return Y, label


# =============================
# Example usage
# =============================
if __name__ == "__main__":
    ds = NearFieldBeamDataset(
        Nt=256, fc=100e9, N_samples=50,
        D_interval=4, delta_n=4, SNR_dB=-10,
        param_cache="NF_params_fixed.npz"
    )

    print(f"\nDataset shape: Y={ds.Y_all.shape}, labels={ds.labels_all.shape}")
    print(f"Sample 0: r={ds.r_all[0]:.2f} m, θ={np.degrees(ds.theta_all[0]):.2f}°")
