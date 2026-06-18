import numpy as np

c = 3e8  # 光速 (m/s)

def polar_domain_manifold(Nt, d, f, r0, theta0):
    """
    极化域阵列流形 (对应 MATLAB polar_domain_manifold)
    参数:
        Nt : int
        d  : float
        f  : float
        r0 : float
        theta0 : float (弧度)
    返回:
        at : ndarray (Nt,)
    """
    nn = np.arange(-(Nt - 1) / 2, (Nt - 1) / 2 + 1)
    r = np.sqrt(r0**2 + (nn * d)**2 - 2 * r0 * nn * d * np.sin(theta0))
    at = np.exp(-1j * 2 * np.pi * f * (r - r0) / c) / np.sqrt(Nt)
    return at
