"""Stateless, prompt/epoch-keyed training randomness (never rollout randomness)."""
import hashlib
import json

import torch


DISCRETE_TIME_TYPES = {"discrete", "discrete_with_init", "discrete_wo_init"}


def prompt_training_seed(seed, epoch, prompt):
    """Use the same decoded prompt text as PerPromptStatTracker, not Python hash()."""
    if epoch < 0:
        raise ValueError("epoch must be nonnegative")
    payload = json.dumps(["prompt-training-v1", int(seed), int(epoch), prompt],
                         ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def shared_prompt_training_inputs(prompt_seeds, schedule, latent_template, *,
                                  num_timesteps, timestep_fraction, time_type):
    """Return time-major [T*B,...] tensors, independent of rank/batch/order.

    Each prompt gets its own private generator. Recreating it in another
    microbatch/rank or after resume yields the same T times and T noise tensors.
    The ambient CPU/CUDA RNGs are untouched. No per-epoch noise cache is needed.
    """
    if time_type not in DISCRETE_TIME_TYPES:
        raise ValueError("Shared prompt noise requires a discrete time_type")
    batch = latent_template.shape[0]
    if schedule.ndim != 2 or schedule.shape[0] != batch or batch == 0:
        raise ValueError("schedule must have shape [batch, rollout_steps]")
    if len(prompt_seeds) != batch or num_timesteps < 1 or not 0 < timestep_fraction <= 1:
        raise ValueError("Invalid prompt seeds, timestep count or fraction")
    device = latent_template.device
    if schedule.device != device:
        raise ValueError("Schedule and latents must be on the same device")
    with_init = time_type == "discrete_with_init"
    remaining = num_timesteps - int(with_init)
    start = 0 if time_type == "discrete" else 1
    end = int(schedule.shape[1] * timestep_fraction)
    if end - start < remaining:
        raise ValueError("Not enough discrete indices for distinct timestep strata")
    seeds = prompt_seeds.detach().cpu().tolist() if isinstance(prompt_seeds, torch.Tensor) else list(prompt_seeds)
    times, noises = [], []
    # Only cache duplicates inside this microbatch (bounded memory).
    cache = {}
    for index, seed in enumerate(seeds):
        seed = int(seed)
        if seed not in cache:
            generator = torch.Generator(device=device).manual_seed(seed)
            indices = torch.zeros(1 if with_init else 0, dtype=torch.long, device=device)
            if remaining:
                boundaries = torch.linspace(start, end, remaining + 1, device=device)
                lower, upper = boundaries[:-1].long(), boundaries[1:].long()
                uniform = torch.rand(remaining, generator=generator, device=device)
                indices = torch.cat([indices, lower + (uniform * (upper - lower)).long()])
            noise = torch.stack([
                torch.randn(latent_template.shape[1:], generator=generator,
                            device=device, dtype=latent_template.dtype)
                for _ in range(num_timesteps)
            ])
            cache[seed] = indices, noise
        indices, noise = cache[seed]
        times.append(schedule[index, indices] / 1000)
        noises.append(noise)
    return (torch.stack(times, dim=1).reshape(-1, 1, 1, 1),
            torch.stack(noises, dim=1).reshape(-1, *latent_template.shape[1:]))
