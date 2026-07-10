from __future__ import annotations

import copy
import gc
import io
import math
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _valid_num_groups(num_channels: int, max_groups: int = 8) -> int:
    groups = min(max_groups, num_channels)
    while groups > 1 and num_channels % groups != 0:
        groups -= 1
    return groups


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim != 1:
            t = t.view(-1)

        half_dim = self.dim // 2
        emb_scale = math.log(10000) / max(half_dim - 1, 1)
        freqs = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb_scale)

        emb = t.float()[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.GroupNorm(_valid_num_groups(in_channels), in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_channels))
        self.norm2 = nn.GroupNorm(_valid_num_groups(out_channels), out_channels)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))
        return h + self.shortcut(x)


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class DiffusionUNet(nn.Module):
    """Lightweight conditional DDPM-style U-Net for compute comparison."""

    def __init__(
        self,
        in_channels: int = 23,
        out_channels: int = 3,
        base_channels: int = 32,
        channel_mults: Sequence[int] = (1, 2, 4),
        num_res_blocks: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        if num_res_blocks < 1:
            raise ValueError("num_res_blocks must be >= 1.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.channel_mults = tuple(channel_mults)
        self.num_res_blocks = num_res_blocks

        time_emb_dim = base_channels * 4
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        channels = [base_channels * mult for mult in self.channel_mults]
        self.init_conv = nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.skip_channels = []

        current_channels = channels[0]
        for level, out_ch in enumerate(channels):
            blocks = nn.ModuleList(
                [ResBlock(current_channels, out_ch, time_emb_dim=time_emb_dim, dropout=dropout)]
            )
            current_channels = out_ch
            for _ in range(num_res_blocks - 1):
                blocks.append(
                    ResBlock(current_channels, current_channels, time_emb_dim=time_emb_dim, dropout=dropout)
                )
            self.down_blocks.append(blocks)
            self.skip_channels.append(current_channels)
            self.downsamples.append(Downsample(current_channels) if level != len(channels) - 1 else nn.Identity())

        self.mid_block1 = ResBlock(current_channels, current_channels, time_emb_dim=time_emb_dim, dropout=dropout)
        self.mid_block2 = ResBlock(current_channels, current_channels, time_emb_dim=time_emb_dim, dropout=dropout)

        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for level in reversed(range(len(channels))):
            skip_ch = self.skip_channels[level]
            out_ch = channels[level]
            blocks = nn.ModuleList(
                [ResBlock(current_channels + skip_ch, out_ch, time_emb_dim=time_emb_dim, dropout=dropout)]
            )
            current_channels = out_ch
            for _ in range(num_res_blocks - 1):
                blocks.append(
                    ResBlock(current_channels, current_channels, time_emb_dim=time_emb_dim, dropout=dropout)
                )
            self.up_blocks.append(blocks)
            self.upsamples.append(Upsample(current_channels) if level != 0 else nn.Identity())

        self.out_norm = nn.GroupNorm(_valid_num_groups(current_channels), current_channels)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(current_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"x must be 4D [B, C, H, W], got shape {tuple(x.shape)}")
        if t.ndim == 0:
            t = t[None].repeat(x.shape[0])
        if t.shape[0] != x.shape[0]:
            raise ValueError(f"Batch size mismatch: x batch={x.shape[0]}, t batch={t.shape[0]}")

        t_emb = self.time_embedding(t)
        h = self.init_conv(x)
        skips = []

        for blocks, downsample in zip(self.down_blocks, self.downsamples):
            for block in blocks:
                h = block(h, t_emb)
            skips.append(h)
            h = downsample(h)

        h = self.mid_block1(h, t_emb)
        h = self.mid_block2(h, t_emb)

        for blocks, upsample in zip(self.up_blocks, self.upsamples):
            skip = skips.pop()
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = torch.cat([h, skip], dim=1)
            for block in blocks:
                h = block(h, t_emb)
            h = upsample(h)

        return self.out_conv(self.out_act(self.out_norm(h)))


class SimpleDDIMScheduler:
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        device: torch.device | str = "cpu",
    ):
        self.num_train_timesteps = num_train_timesteps
        self.device = torch.device(device)
        betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32, device=self.device)
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)

    def get_timesteps(self, sampling_steps: int) -> torch.Tensor:
        return torch.linspace(
            self.num_train_timesteps - 1,
            0,
            steps=sampling_steps,
            device=self.device,
        ).long()

    def step(self, epsilon_pred: torch.Tensor, y_t: torch.Tensor, t: int, prev_t: int) -> torch.Tensor:
        alpha_t = self.alphas_cumprod[t]
        alpha_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else torch.tensor(1.0, device=y_t.device)

        y0_pred = (y_t - torch.sqrt(1.0 - alpha_t) * epsilon_pred) / torch.sqrt(alpha_t)
        return torch.sqrt(alpha_prev) * y0_pred + torch.sqrt(1.0 - alpha_prev) * epsilon_pred


class DiffusionForwardWrapper(nn.Module):
    def __init__(self, model: nn.Module, timestep: int = 500):
        super().__init__()
        self.model = model
        self.timestep = timestep

    def forward(self, model_input: torch.Tensor) -> torch.Tensor:
        t = torch.full((model_input.shape[0],), self.timestep, device=model_input.device, dtype=torch.long)
        return self.model(model_input, t)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def state_dict_size_mb(model: nn.Module, output_dir: str | Path, name: str) -> float:
    del output_dir, name
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.tell() / 1024 / 1024


def cleanup_thop_state(model: nn.Module) -> None:
    for module in model.modules():
        for attr in ("total_ops", "total_params"):
            if hasattr(module, attr):
                try:
                    delattr(module, attr)
                except Exception:
                    pass
        for hooks in ("_forward_hooks", "_forward_pre_hooks", "_backward_hooks"):
            try:
                getattr(module, hooks).clear()
            except Exception:
                pass


def cuda_autocast(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def compute_flops_forward(
    model: nn.Module,
    inputs: tuple[torch.Tensor, ...],
    device: torch.device,
    macs_to_flops_factor: float = 2.0,
) -> float:
    from thop import profile

    model_for_profile = copy.deepcopy(model).to(device)
    profile_inputs = tuple(item.detach().clone().to(device) for item in inputs)
    try:
        model_for_profile.eval()
        with torch.no_grad():
            macs, _ = profile(model_for_profile, inputs=profile_inputs, verbose=False)
        return macs * macs_to_flops_factor / 1e9
    finally:
        cleanup_thop_state(model_for_profile)
        del model_for_profile
        del profile_inputs
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


@torch.no_grad()
def diffusion_sample(
    model: nn.Module,
    x_cond: torch.Tensor,
    scheduler: SimpleDDIMScheduler,
    sampling_steps: int,
    target_channels: int = 3,
    use_amp: bool = False,
) -> torch.Tensor:
    batch_size, _, height, width = x_cond.shape
    y_t = torch.randn(batch_size, target_channels, height, width, device=x_cond.device, dtype=x_cond.dtype)
    timesteps = scheduler.get_timesteps(sampling_steps)

    for index, timestep in enumerate(timesteps):
        t_int = int(timestep.item())
        prev_t = int(timesteps[index + 1].item()) if index < len(timesteps) - 1 else -1
        t_batch = torch.full((batch_size,), t_int, device=x_cond.device, dtype=torch.long)
        model_input = torch.cat([y_t, x_cond], dim=1)
        with cuda_autocast(enabled=use_amp and x_cond.device.type == "cuda"):
            epsilon_pred = model(model_input, t_batch)
        y_t = scheduler.step(epsilon_pred=epsilon_pred, y_t=y_t, t=t_int, prev_t=prev_t)

    return y_t


@torch.no_grad()
def peak_memory_single_forward(
    model: nn.Module,
    inputs_list: list[torch.Tensor],
    device: torch.device,
    use_amp: bool = False,
) -> float:
    if device.type != "cuda":
        return 0.0

    model.eval()
    for input_tensor in inputs_list[: min(3, len(inputs_list))]:
        with cuda_autocast(enabled=use_amp):
            _ = model(input_tensor)

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    for input_tensor in inputs_list:
        with cuda_autocast(enabled=use_amp):
            _ = model(input_tensor)

    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024 / 1024


@torch.no_grad()
def peak_memory_diffusion_sampling(
    model: nn.Module,
    x_cond_list: list[torch.Tensor],
    scheduler: SimpleDDIMScheduler,
    sampling_steps: int,
    device: torch.device,
    target_channels: int = 3,
    use_amp: bool = False,
) -> float:
    if device.type != "cuda":
        return 0.0

    model.eval()
    for x_cond in x_cond_list[: min(2, len(x_cond_list))]:
        _ = diffusion_sample(model, x_cond, scheduler, sampling_steps, target_channels, use_amp)

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    for x_cond in x_cond_list:
        _ = diffusion_sample(model, x_cond, scheduler, sampling_steps, target_channels, use_amp)

    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024 / 1024
