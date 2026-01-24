import torch


def copy_fp32_to_bf16_stochastic_torch_(target: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    val_i32 = source.view(torch.int32)
    noise = torch.randint(
        0, 65536, source.shape,
        device=source.device,
        dtype=torch.int32
    )
    val_noisy = val_i32 + noise
    res = (val_noisy >> 16).to(torch.int16).view(torch.bfloat16)
    target.copy_(res)
    return target
