import numpy as np
import torch
import matplotlib.pyplot as plt
from time import time

from near_field_channel import near_field_channel
from polar_codebook import polar_codebook
from training_near_exhaustive import training_near_exhaustive
from codeword_sampling import sample_codewords
from net import DNBTNet  # 确保类名一致

# ======================================================
# 参数设置
# ======================================================
Nt = 256
fc = 100e9
c = 3e8
lambda_c = c / fc
d = lambda_c / 2
B = 0
M = 1
Q = 16
mode = 0
overhead_max = 5000

# 近场参数范围
Rmin, Rmax = 10, 40
theta_min, theta_max = -np.pi / 3, np.pi / 3

# Monte Carlo 次数与 SNR 扫描范围
N_iter = 1000
SNR_dB_list = np.arange(-5, 25, 5)
SNR_list = 10 ** (SNR_dB_list / 10)
SNR_len = len(SNR_dB_list)

# ======================================================
# 加载训练好的 DNBT 模型
# ======================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dnbt_model = DNBTNet(N=256, S=6, num_blocks=10, dropout_p=0.2, use_bn=True)
dnbt_model.load_state_dict(torch.load("dnbt_trained.pth", map_location=device))
dnbt_model.to(device)
dnbt_model.eval()
print("✅ DNBT 模型加载成功！")

# ======================================================
# 生成极化域码本（仅一次）
# ======================================================
w_phase_near, sin_theta_nb, r_list = polar_codebook(
    Nt, s=1, d=d, lambda_c=lambda_c,
    delta=1.2, rho_min=3,
    theta_min=theta_min, theta_max=theta_max
)
D2 = len(sin_theta_nb)
S2 = len(r_list)
print(f"角度采样点数 D2 = {D2}, 距离采样点数 S2 = {S2}")

# 采样码字索引（与训练保持一致）
# sampled_w, sampled_indices, _, _ = sample_codewords(
#    w_phase_near, D_interval=16, delta_n=8, return_mask=True
# )
sampled_indices = np.load('sampled_indices.npy', allow_pickle=True)

# ======================================================
# 初始化结果矩阵
# ======================================================
rate_exhaustive = np.zeros((SNR_len, N_iter))
rate_opt = np.zeros((SNR_len, N_iter))
rate_dnbt = np.zeros((SNR_len, N_iter))

### 新增：保存每次仿真的阵列增益（用来做归一化）
array_gain_ex_all = np.zeros((SNR_len, N_iter))
array_gain_opt_all = np.zeros((SNR_len, N_iter))
array_gain_dnbt_all = np.zeros((SNR_len, N_iter))
array_gain_opt_ex = np.zeros((SNR_len, N_iter))
# ======================================================
# 主 Monte Carlo 循环
# ======================================================
t0 = time()
for idx, SNR_dB in enumerate(SNR_dB_list):
    SNR = SNR_list[idx]
    print(f"\n--- SNR = {SNR_dB:.1f} dB [{idx+1}/{SNR_len}] ---")

    for i_iter in range(N_iter):
        # 1️⃣ 随机用户位置
        r_user = np.random.uniform(Rmin, Rmax)
        theta_user = np.random.uniform(theta_min, theta_max)

        # 2️⃣ 生成信道
        H, hc = near_field_channel(Nt, d, fc, B, M, r_user, theta_user)

        # 3️⃣ 生成观测矩阵 Y（用于 DNBT 推理）
        Y = np.zeros((D2, S2))
        for s, idx_list in enumerate(sampled_indices):
            for n in idx_list:
                signal = np.vdot(hc, w_phase_near[:, n, s])
                signal_power = np.mean(np.abs(signal) ** 2)
                noise_power = signal_power / (10 ** (SNR_dB / 10.0))

                noise = np.sqrt(noise_power / 2) * (np.random.randn() + 1j * np.random.randn())
                Y[n, s] = np.abs(signal + noise)
        # 归一化
        Y = (Y - Y.min()) / (Y.max() - Y.min())

        # ======================================================
        # ① 穷举法（Exhaustive Training）
        # ======================================================
        rate_exhaustive_iter, array_gain_ex, _, _, idx_ex = training_near_exhaustive(
            D2, S2, hc, w_phase_near, SNR, SNR_dB,
            Q, mode, M, overhead_max, sin_theta_nb, r_list
        )
        rate_exhaustive[idx, i_iter] = np.mean(rate_exhaustive_iter)
        ### 新增：保存穷举阵列增益
        array_gain_ex_all[idx, i_iter] = array_gain_ex

        _, array_gain_opt,_, _, _ = training_near_exhaustive(
            D2, S2, hc, w_phase_near, SNR, 1000,
            Q, mode, M, overhead_max, sin_theta_nb, r_list
        )
        array_gain_opt_ex[idx, i_iter] = array_gain_opt

        # ======================================================
        # ② 理论最优上界（Perfect CSI / MRT）
        # ======================================================
        wc_opt = np.exp(1j * np.angle(hc)) / np.sqrt(Nt)
        array_gain = np.abs(np.vdot(hc, wc_opt))**2
        rate_opt[idx, i_iter] = np.log2(1 + SNR * array_gain)
        ### 新增：保存理论阵列增益
        array_gain_opt_all[idx, i_iter] = array_gain

        # ======================================================
        # ③ DNBT 推理（Deep Neural Beam Training）
        # ======================================================
        Y_tensor = torch.tensor(Y, dtype=torch.float32).unsqueeze(0).to(device)  # (1, N, S)
        with torch.no_grad():
            logits = dnbt_model(Y_tensor)
            idx_pred = torch.argmax(logits, dim=-1).item()

        # 将预测索引映射回 (a, s)
        a_pred = idx_pred % D2
        s_pred = idx_pred // D2

        wc_pred = w_phase_near[:, a_pred, s_pred]
        array_gain_dnbt = np.abs(np.vdot(hc, wc_pred))**2
        rate_dnbt[idx, i_iter] = np.log2(1 + SNR * array_gain_dnbt)
        ### 新增：保存 DNBT 阵列增益
        array_gain_dnbt_all[idx, i_iter] = array_gain_dnbt

    print(f"✅ Completed SNR = {SNR_dB:.1f} dB | Elapsed = {time() - t0:.1f}s")

# ======================================================
# Monte Carlo 平均（速率）
# ======================================================
rate_exhaustive_mean = np.nanmean(rate_exhaustive, axis=1)
rate_opt_mean = np.nanmean(rate_opt, axis=1)
rate_dnbt_mean = np.nanmean(rate_dnbt, axis=1)

# ======================================================
# 计算并输出：不同 SNR 下的平均归一化阵列增益
# ======================================================
# 防止除零（理论上 array_gain_opt 不会是 0，这里稳妥起见）
norm_gain_ex = array_gain_ex_all / array_gain_opt_ex
norm_gain_dnbt = array_gain_dnbt_all / array_gain_opt_ex

# 对每个 SNR 做 Monte Carlo 平均
norm_gain_ex_mean = np.nanmean(norm_gain_ex, axis=1)
norm_gain_dnbt_mean = np.nanmean(norm_gain_dnbt, axis=1)

print("\n========= 仿真结果（速率） =========")
for i, snr in enumerate(SNR_dB_list):
    print(f"SNR={snr:>3} dB | Exhaustive={rate_exhaustive_mean[i]:.4f} | "
          f"DNBT={rate_dnbt_mean[i]:.4f} | Opt={rate_opt_mean[i]:.4f}")

print("\n========= 不同 SNR 下的平均归一化阵列增益 =========")
for i, snr in enumerate(SNR_dB_list):
    print(f"SNR={snr:>3} dB | "
          f"E[array_gain_ex/array_gain]={norm_gain_ex_mean[i]:.4f} | "
          f"E[array_gain_dnbt/array_gain]={norm_gain_dnbt_mean[i]:.4f}")

# ======================================================
# 绘图展示（速率）
# ======================================================
plt.figure(figsize=(6,5))
plt.plot(SNR_dB_list, rate_exhaustive_mean, 'o-', label='Exhaustive Search')
plt.plot(SNR_dB_list, rate_dnbt_mean, 's-', label='DNBT (Deep Learning)')
plt.plot(SNR_dB_list, rate_opt_mean, '--', label='Perfect CSI (Upper Bound)')
plt.xlabel('SNR (dB)')
plt.ylabel('Achievable Rate (bit/s/Hz)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

### 如需画“归一化阵列增益 vs SNR”曲线，你也可以再加一幅：
plt.figure(figsize=(6,5))
plt.plot(SNR_dB_list, norm_gain_ex_mean, 'o-', label='Exhaustive / Opt')
plt.plot(SNR_dB_list, norm_gain_dnbt_mean, 's-', label='DNBT / Opt')
plt.xlabel('SNR (dB)')
plt.ylabel('Normalized Array Gain')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()
