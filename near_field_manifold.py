import numpy as np

c = 3e8  # 光速 (m/s)

def near_field_manifold(Nt, d, f, r0, theta0):
    """
    生成近场阵列流形向量 a_t (对应 MATLAB near_field_manifold)

    参数:
        Nt : int         # 天线数
        d  : float       # 天线间距 (m)
        f  : float       # 频率 (Hz)
        r0 : float       # 用户距离 (m)
        theta0 : float   # 入射角 (弧度)
    返回:
        at : ndarray (Nt,)
    """
    nn = np.arange(-(Nt - 1) / 2, (Nt - 1) / 2 + 1)
    r = np.sqrt(r0**2 + (nn * d)**2 - 2 * r0 * nn * d * np.sin(theta0))
    at = np.exp(-1j * 2 * np.pi * f * (r - r0) / c) / np.sqrt(Nt)
    return at
