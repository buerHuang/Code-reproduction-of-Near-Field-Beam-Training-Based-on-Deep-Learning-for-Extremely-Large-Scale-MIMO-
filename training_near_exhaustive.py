import numpy as np


def training_near_exhaustive(
    D2: int,
    S2: int,
    hc: np.ndarray,
    w_phase_near: np.ndarray,
    SNR_t: float,
    SNR_dB: float,
    Q: int,
    mode: int,
    M: int,
    overhead_max: int,
    sin_theta_near_nb: np.ndarray,
    r_near_nb: np.ndarray,
):
    """
    近场穷举波束训练 (Exhaustive Near-Field Beam Training)
    -------------------------------------------------------
    对所有角度-距离码字进行扫描，选择使得接收信号增益最大的波束。

    参数:
    ----------
    D2 : int
        角度方向码字数量 (angle samples)
    S2 : int
        距离方向码字数量 (distance rings)
    hc : ndarray (Nt,)
        中心频率处的近场信道向量
    w_phase_near : ndarray (Nt, D2, S2)
        近场极化域码本
    SNR_t : float
        线性信噪比（SNR）
    SNR_dB : float
        信噪比（dB）
    Q : int
        频域滤波长度 (用于低信噪比模式)
    mode : int
        1 → 高信噪比模式（用于overhead曲线）
        其他 → 低信噪比模式
    M : int
        频率采样点 (保留兼容性)
    overhead_max : int
        最大训练开销 (最大可扫描码字数量)
    sin_theta_near_nb : ndarray (D2,)
        角度采样的sin(theta)列表
    r_near_nb : ndarray (S2,)
        距离采样列表

    返回:
    ----------
    rate_near_nb : float or ndarray
        模式1 → 长度为 overhead 的速率序列
        模式2 → 单个标量速率 (最优波束对应的速率)
    theta_best : float
        最优角度 (弧度)
    r_best : float
        最优距离 (米)
    """

    overhead = min(D2 * S2, overhead_max)

    # 高信噪比模式：逐步扫描并记录累积最大速率
    if mode == 1:
        rate_near_nb = np.zeros(overhead)
        for idx in range(overhead):
            s = idx // D2  # 距离索引
            a = idx % D2   # 角度索引
            wc_near_nb = w_phase_near[:, a, s]
            array_gain = np.abs(np.vdot(hc, wc_near_nb))**2
            temp = np.log2(1 + SNR_t * array_gain)
            if idx == 0:
                rate_near_nb[idx] = temp
            else:
                rate_near_nb[idx] = max(rate_near_nb[idx - 1], temp)

        theta_best, r_best = None, None  # 高信噪比模式不输出具体角距
        return rate_near_nb, theta_best, r_best

    # -------------------------------------------------------------
    # 低信噪比模式：加噪仿真 + 穷举最大增益
    # -------------------------------------------------------------
    else:
        array_gain = np.zeros(overhead)

        for idx in range(overhead):
            s = idx // D2
            a = idx % D2
            wc_near_nb = w_phase_near[:, a, s]

            # 理论接收信号
            y = np.vdot(hc, wc_near_nb)
            signal = np.tile(y, Q)

            # 根据信号功率自适应噪声功率 (等效 MATLAB awgn)
            signal_power = np.mean(np.abs(signal) ** 2)
            noise_power = signal_power / (10 ** ((SNR_dB) / 10.0))

            noise = np.sqrt(noise_power / 2) * (np.random.randn(Q) + 1j * np.random.randn(Q))
            temp = signal + noise

            array_gain[idx] = np.abs(np.sum(temp)) ** 2

        # 选择最优波束
        idx_max = np.argmax(array_gain)
        s = idx_max // D2
        a = idx_max % D2
        theta_best = np.arcsin(sin_theta_near_nb[a])
        r_best = r_near_nb[s]

        wc_best = w_phase_near[:, a, s]
        array_gain_final = np.abs(np.vdot(hc, wc_best))**2
        rate_near_nb = np.log2(1 + SNR_t * np.abs(np.vdot(hc, wc_best)) ** 2)

        return rate_near_nb, array_gain_final, theta_best, r_best, idx_max
