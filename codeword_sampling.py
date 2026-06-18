import numpy as np

def sample_codewords(w, D_interval=8, delta_n=1, return_mask=False, verbose=False):
    """
    基于论文 Section III-A，对极化域码本进行初始码字采样。
    论文思想：每个距离环角度等间隔采样，并在相邻环之间错位 Δn。

    参数:
        w : ndarray (Nt, D, S)
            完整极化域码本 (polar_codebook() 生成)
        D_interval : int
            角度采样间隔 D_interval
        delta_n : int
            相邻距离环起始角索引偏移 Δn
        return_mask : bool
            是否返回一个 (D, S) 的布尔掩码矩阵
        verbose : bool
            是否打印采样信息

    返回:
        sampled_w : ndarray (Nt, T, S)
            采样后的子码本
        sampled_indices : list[list[int]]
            每个距离环的角索引列表
        sampled_flat_indices : list[int]
            在全码本中的线性索引集合
        mask : ndarray (D, S), 可选
            若 return_mask=True，返回布尔掩码矩阵
    """
    Nt, D, S = w.shape
    T = D // D_interval
    sampled_w = np.zeros((Nt, T, S), dtype=complex)
    sampled_indices = []

    n0 = 0
    for s in range(S):
        indices = [(n0 + k * D_interval) % D for k in range(T)]
        sampled_indices.append(indices)
        sampled_w[:, :, s] = w[:, indices, s]
        n0 = (n0 + delta_n) % D

    # 全局线性索引
    sampled_flat_indices = [(s * D + n) for s in range(S) for n in sampled_indices[s]]

    if verbose:
        print(f"[sample_codewords] 每环采样 {T} 个角度, 间隔={D_interval}, 偏移={delta_n}, 总采样={len(sampled_flat_indices)}")

    if return_mask:
        mask = np.zeros((D, S), dtype=bool)
        for s, idx_list in enumerate(sampled_indices):
            mask[idx_list, s] = True
        return sampled_w, sampled_indices, sampled_flat_indices, mask

    return sampled_w, sampled_indices, sampled_flat_indices
