import torch
from typing import Optional, Tuple
from typing import List

@torch.no_grad()
def get_symmetric_states(
    obs: Optional[torch.Tensor] = None,
    actions: Optional[torch.Tensor] = None,
    env=None,
    obs_type: str = "policy"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    对称数据增强函数，返回左右镜像后的 obs 和 actions。

    Args:
        obs (torch.Tensor): 观测数据 [batch_size, obs_dim]
        actions (torch.Tensor): 动作数据 [batch_size, action_dim]
        env: 环境对象，需包含 symmetry.joint_permutation 和 symmetry.obs_permutation
        obs_type (str): 观测类型 "policy" or "critic"

    Returns:
        obs_augmented (torch.Tensor): 增强后的观测 [2*B, obs_dim]
        actions_augmented (torch.Tensor): 增强后的动作 [2*B, act_dim]
    """

    # 1. 获取对称映射索引
    obs_perm = make_critic_obs_permutation(policy_perm = env.cfg.symmetry.obs_permutation[obs_type])
    act_perm = env.cfg.symmetry.joint_permutation           # list[int], len = act_dim
    if isinstance(obs_perm, list):
        obs_perm = torch.tensor(obs_perm)
    if isinstance(act_perm, list):
        act_perm = torch.tensor(act_perm)
    
    # 2. 初始化输出
    n_scan=187
    n_priv_latent=36
    hist_len=10
    priv_start = 33 + n_scan
    hist_start = priv_start + n_priv_latent
    obs_augmented = []
    actions_augmented = []
    negate_indices = [0, 2, 7,
                      priv_start+1,
                      hist_start+0*33+0,hist_start+0*33+2,hist_start+0*33+7,
                      hist_start+1*33+0,hist_start+1*33+2,hist_start+1*33+7,
                      hist_start+2*33+0,hist_start+2*33+2,hist_start+2*33+7,
                      hist_start+3*33+0,hist_start+3*33+2,hist_start+3*33+7,
                      hist_start+4*33+0,hist_start+4*33+2,hist_start+4*33+7,
                      hist_start+5*33+0,hist_start+5*33+2,hist_start+5*33+7,
                      hist_start+6*33+0,hist_start+6*33+2,hist_start+6*33+7,
                      hist_start+7*33+0,hist_start+7*33+2,hist_start+7*33+7,
                      hist_start+8*33+0,hist_start+8*33+2,hist_start+8*33+7,
                      hist_start+9*33+0,hist_start+9*33+2,hist_start+9*33+7,]  # 想要取反的列索引
    # print("11111111111111",type(obs_perm))
    obs_perm[negate_indices] = -obs_perm[negate_indices]

    if obs is not None:
        obs_sym = obs[:, obs_perm]
        obs_augmented = torch.cat([obs, obs_sym], dim=0)

    if actions is not None:
        actions_sym = actions[:, act_perm]
        actions_augmented = torch.cat([actions, actions_sym], dim=0)

    return obs_augmented, actions_augmented

def make_critic_obs_permutation(policy_perm: List[int], n_scan=187, n_priv_latent=36, hist_len=10):
    # base: 33维 policy obs
    perm = policy_perm.copy()

    # scan: 直接反转 scan 序列（如果为 1D 或平铺）
    perm += list(range(33, 33 + n_scan))  # 如果 scan 没有左右结构可不变

    # priv_latent 部分
    priv_start = 33 + n_scan
    perm += [priv_start + 0, priv_start + 1, priv_start + 2]  # base_lin_vel: flip y
    perm += [priv_start + 4,  # contact_filt: R -> L
             priv_start + 3]  # contact_filt: L -> R
    perm += list(range(priv_start + 5, priv_start + n_priv_latent))  # 其余保持不变

    # 历史 obs：共 330 维，每 33 是 policy obs，一共 10 段
    hist_start = priv_start + n_priv_latent
    for i in range(hist_len):
        base = hist_start + i * 33
        perm += [base + idx for idx in policy_perm]
    return perm

