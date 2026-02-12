import torch


def adamw_step_torch(
    p: torch.Tensor,
    g: torch.Tensor,
    m: torch.Tensor,
    v: torch.Tensor,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    step: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    p_f64 = p.to(torch.float32)
    g_f64 = g.to(torch.float32)
    m_f64 = m.to(torch.float32)
    v_f64 = v.to(torch.float32)

    p_f64 = p_f64 * (1.0 - lr * weight_decay)

    m_next = beta1 * m_f64 + (1 - beta1) * g_f64
    v_next = beta2 * v_f64 + (1 - beta2) * (g_f64 * g_f64)

    bc1 = 1 - beta1**step
    bc2 = 1 - beta2**step

    m_hat = m_next / bc1
    v_hat = v_next / bc2

    update = lr * m_hat / (torch.sqrt(v_hat) + eps)
    p_final = p_f64 - update

    return p_final, m_next, v_next
