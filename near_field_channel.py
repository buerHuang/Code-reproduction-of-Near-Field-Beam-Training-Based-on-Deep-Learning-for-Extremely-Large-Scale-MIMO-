import numpy as np
from near_field_manifold import near_field_manifold

c = 3e8  # 光速 (m/s)

def near_field_channel(Nt, d, fc, B, M, r, theta):
    """
    生成近场信道矩阵 H 及中心频率分量 hc (对应 MATLAB near_field_channel)

    参数:
        Nt : int       # 天线数
        d  : float     # 天线间距
        fc : float     # 载波频率
        B  : float     # 带宽（可设为0表示窄带）
        M  : int       # 频率采样点数
        r  : float     # 用户距离 (m)
        theta : float  # 入射角 (弧度)
    返回:
        H  : ndarray (M, Nt)
        hc : ndarray (Nt,)
    """
    H = np.zeros((M + 1, Nt), dtype=complex)
    for m in range(M + 1):
        if m == M:
            f = fc
        else:
            f = fc + B / M * (m - (M - 1) / 2)
        at = near_field_manifold(Nt, d, f, r, theta)
        H[m, :] = (f / fc) * np.exp(-1j * 2 * np.pi * f * r / c) * at

    hc = H[-1, :]  # 中心频率对应分量
    H = H[:-1, :]  # 前 M 个用于信道矩阵
    return H, hc
