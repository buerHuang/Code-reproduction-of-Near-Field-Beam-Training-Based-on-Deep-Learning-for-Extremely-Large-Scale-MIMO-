import numpy as np
from polar_domain_manifold import polar_domain_manifold

c = 3e8  # 光速 (m/s)

def polar_codebook(Nt, s, d, lambda_c, delta, rho_min, theta_min, theta_max):
    """
    生成近场极化域码本 (对应 MATLAB PolarCode)

    参数:
        Nt : int
        s  : float   # 角度采样比例
        d  : float   # 天线间距
        lambda_c : float  # 波长
        delta : float     # 控制码本密度的参数
        rho_min : float   # 最小径向采样因子
        theta_min, theta_max : float  # 角度范围 (弧度)
    返回:
        w : ndarray (Nt, D, S)
        sin_theta_list : ndarray (D,)
        r_list : ndarray (S,)
    """
    fc = c / lambda_c
    sin_theta_list = np.arange(np.sin(theta_min), np.sin(theta_max), 2 / (s * Nt))
    sin_theta_list = np.linspace(np.sin(theta_min), np.sin(theta_max), (s * Nt), endpoint=True)
    D = len(sin_theta_list)

    Z_delta = (Nt * d)**2 / (2 * lambda_c * delta**2)
    S = int(np.floor(Z_delta / rho_min)) + 1

    r_list = np.zeros(S)
    for si in range(S):
        if si == 0:
            r_list[si] = 200 * Nt**2 * d**2 / lambda_c
        else:
            r_list[si] = Z_delta / si

    w = np.zeros((Nt, D, S), dtype=complex)
    for si in range(S):
        for idx, sin_theta in enumerate(sin_theta_list):
            r = r_list[si] * (1 - sin_theta**2)
            theta = np.arcsin(sin_theta)
            w[:, idx, si] = polar_domain_manifold(Nt, d, fc, r, theta)

    return w, sin_theta_list, r_list
