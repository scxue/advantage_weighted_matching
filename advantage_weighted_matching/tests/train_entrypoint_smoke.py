"""Exercise the actual AWM entrypoint with a tiny SD3 transformer and fake rewards.

Only external model loading, text embedding, rollout and reward services are
replaced. The real LoRA initialization, loss/KL, data layout, eval, AdamW,
GradScaler, accumulation, EMA and checkpoint paths execute unchanged.
"""
import argparse
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import torch
from diffusers import SD3Transformer2DModel
from flow_grpo.resume import cpu_tree
from scripts import train_sd3_awm as train
from config import dgx_awm


class Tokenizer:
    def __call__(self, prompts, **kwargs):
        ids = [int(hashlib.sha256(p.encode()).hexdigest()[:8], 16) % 997 for p in prompts]
        return SimpleNamespace(input_ids=torch.tensor(ids).reshape(-1, 1))

    def batch_decode(self, ids, **kwargs):
        return [f"prompt-{int(i[0])}" for i in ids]


class Pipeline:
    def __init__(self):
        # Mimic pretrained components: identical weights, no process RNG changes.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17)
            self.transformer = SD3Transformer2DModel(
                sample_size=4, patch_size=2, in_channels=4, out_channels=4,
                num_layers=1, attention_head_dim=8, num_attention_heads=2,
                joint_attention_dim=16, caption_projection_dim=16,
                pooled_projection_dim=16, pos_embed_max_size=4,
            )
            self.vae = torch.nn.Linear(1, 1)
            self.text_encoder = torch.nn.Linear(1, 1)
            self.text_encoder_2 = torch.nn.Linear(1, 1)
            self.text_encoder_3 = torch.nn.Linear(1, 1)
        self.tokenizer = self.tokenizer_2 = self.tokenizer_3 = Tokenizer()
        self.scheduler = SimpleNamespace(timesteps=None)

    def set_progress_bar_config(self, **kwargs):
        pass


def embeddings(prompts, encoders, tokenizers, device, **kwargs):
    ids = tokenizers[0](prompts).input_ids.to(device).float() / 997
    return ids.reshape(-1, 1, 1).expand(-1, 2, 16), ids.expand(-1, 16)


def rollout(pipeline, prompt_embeds, num_inference_steps, **kwargs):
    device = prompt_embeds.device
    count = len(prompt_embeds)
    pipeline.scheduler.timesteps = torch.linspace(1000, 10, num_inference_steps, device=device)
    noise = torch.randn(count, 4, 4, 4, device=device)
    latents = [noise]
    for i in range(num_inference_steps):
        # Include independent solver-like random draws in the real logical context.
        latents.append(latents[-1] * .9 + .01 * torch.randn_like(noise))
    images = latents[-1][:, :3].sigmoid()
    return images, latents


def reward(images, prompts, metadata, **kwargs):
    values = images.detach().float().mean(dim=(1, 2, 3)).cpu().numpy()
    return {"avg": values, "ori_avg": values}, {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=["continuous", "resume"], required=True)
    parser.add_argument("--task", choices=["geneval", "ocr", "pickscore"], default="geneval")
    args = parser.parse_args()
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    config = getattr(dgx_awm, f"{args.task}_sd3_no_cfg_4nodes")()
    train_batches = config.logical_rounds * config.sample.train_batch_size // config.train.batch_size
    config.sample.num_batches_per_epoch = config.logical_rounds * (32 // world)
    config.train.num_batches_per_epoch = config.train.gradient_accumulation_steps = train_batches * (32 // world)
    config.num_epochs = 4
    config.eval_freq = 3  # Eval immediately after the checkpoint/resume boundary.
    config.resolution = 32
    config.checkpoint_every_steps = 1
    config.save_dir = str(args.out)
    if args.mode == "resume":
        config.resume_from = str(args.out / "checkpoints/checkpoint-1")
    train.FLAGS = SimpleNamespace(config=config)
    train.StableDiffusion3Pipeline.from_pretrained = lambda *a, **kw: Pipeline()
    train.compute_text_embeddings = embeddings
    train.pipeline_with_logprob = rollout
    train.flow_grpo.rewards.multi_score = lambda *a, **kw: reward
    train.wandb = SimpleNamespace(init=lambda **kw: None, run=SimpleNamespace(id="entrypoint-smoke", step=0),
                                 define_metric=lambda *a, **kw: None, log=lambda *a, **kw: None,
                                 Image=lambda *a, **kw: None)
    input_trace = []
    real_compute = train.compute_log_prob_awm
    def compute(*a, **kw):
        if torch.is_grad_enabled() and kw.get("cfg_scale_override") is None:
            def digest(tensor):
                return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            # timestep, prompt embeddings, clean latents, training noise.
            input_trace.append(tuple(digest(a[i]) for i in (3, 4, 8, 9)))
        return real_compute(*a, **kw)
    train.compute_log_prob_awm = compute
    real_load = train.load_checkpoint
    def load(*a, **kw):
        metadata = real_load(*a, **kw)
        from peft import get_peft_model_state_dict
        from test_resume import same
        import unittest
        test = unittest.TestCase()
        payload = torch.load(Path(a[0]) / "training.pt", map_location="cpu", weights_only=False)
        for name, weights in payload["adapters"].items():
            same(test, cpu_tree(get_peft_model_state_dict(a[1], adapter_name=name)), weights)
        same(test, cpu_tree(a[2].state_dict()), payload["optimizer"])
        same(test, a[3].state_dict(), payload["scheduler"])
        same(test, a[4].scaler.state_dict(), payload["scaler"])
        for logical in a[5].layout.logical_ids:
            saved = torch.load(Path(a[0]) / f"logical-{logical}.pt", map_location="cpu", weights_only=False)
            same(test, a[5].state_for(logical), saved)
        return metadata
    train.load_checkpoint = load
    real_save = train.save_checkpoint
    def save(*a, **kw):
        metadata = a[-1]
        model, optimizer, scheduler, accelerator, logical = a[1:6]
        if metadata["global_step"] == 1:
            return real_save(*a, **kw)
        filename = args.out / f"{args.mode}-{world}-rank{rank}.pt"
        result = cpu_tree({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                           "scheduler": scheduler.state_dict(), "scaler": accelerator.scaler.state_dict(),
                           "logical": {r: logical.state_for(r) for r in logical.layout.logical_ids},
                           "train_batches_per_logical_rank": train_batches,
                           "input_trace": input_trace[-train_batches * (32 // world):]})
        torch.save(result, filename)
        return filename
    train.save_checkpoint = save
    train.main(None)
    print(f"ENTRYPOINT PASS mode={args.mode} rank={rank} world={world}", flush=True)
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
