"""Actual CUDA/DDP/FP16/AdamW integration test, including changed world size.

Run --mode continuous on 8 GPUs, then --mode resume on 8 and 4 GPUs.
This is a tiny LoRA model, not a Geneval quality/production training run.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.distributed as dist
from accelerate import Accelerator
from flow_grpo.resume import cpu_tree, save_checkpoint, load_checkpoint
from test_resume import tiny_model, logical_state


def digest(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--mode", choices=["continuous", "resume"], required=True)
    args.add_argument("--out", type=Path, required=True)
    opts = args.parse_args()
    world = dist.get_world_size() if dist.is_initialized() else int(__import__("os").environ["WORLD_SIZE"])
    accumulation = 18 * (32 // world)
    accelerator = Accelerator(mixed_precision="fp16", gradient_accumulation_steps=accumulation)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = tiny_model(accelerator.device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    refs = [p for n, p in model.named_parameters() if ".ema." in n]
    for p, r in zip(parameters, refs, strict=True):
        r.data.copy_(p.data)
    state = logical_state(world, accelerator.process_index, parameters, accelerator.device)
    optimizer = torch.optim.AdamW(parameters, lr=.001, betas=(.9, .999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, .97)
    model, optimizer = accelerator.prepare(model, optimizer)
    raw = model.module  # DDP-only, like the training entrypoint; do not import DeepSpeed.
    sig = {"test": "cuda-ddp-fp16-v1", "config": {"train": {"ema": True}}}
    first = 0
    if opts.mode == "resume":
        meta = load_checkpoint(opts.out / "checkpoints/checkpoint-1", raw, optimizer,
                               scheduler, accelerator, state, sig)
        first = meta["next_epoch"]
    trace = {}
    for epoch in range(first, 3):
        inputs = []
        for local_batch in range(9 * len(state.layout.logical_ids)):
            logical, prompts = state.layout.rollout(epoch, local_batch)
            with state.use(logical):
                noise = torch.randn(6, 4, device=accelerator.device)
            trace[str((epoch, logical, local_batch % 9, "rollout"))] = (prompts, digest(noise))
            inputs.append(noise)
        inputs = torch.cat(inputs)[state.permutation()]
        batches = []
        for i, clean in enumerate(inputs.split(3)):
            logical = state.layout.logical_ids[i // 18]
            with state.use(logical):
                times = torch.rand(5, device=accelerator.device)
                noise = torch.randn(3, 4, device=accelerator.device)
            trace[str((epoch, logical, i % 18, "train"))] = (digest(clean), digest(times), digest(noise))
            batches.append((logical, clean + noise * times[0]))
        for logical, value in batches:
            with state.use(logical), accelerator.accumulate(model):
                with accelerator.autocast():
                    prediction = model(value)
                    loss = prediction.float().square().mean()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(parameters, 1.)
                optimizer.step()
                optimizer.zero_grad()
            if accelerator.sync_gradients:
                assert not accelerator.optimizer_step_was_skipped, "Unexpected overflow in tiny smoke test"
                scheduler.step()
        state.update_ema(parameters, epoch + 1)
        for p, r in zip(parameters, refs, strict=True):
            r.data.mul_(.001 * (epoch + 1)).add_(p.data, alpha=1 - .001 * (epoch + 1))
        if epoch == 0 and opts.mode == "continuous":
            metadata = {"global_step": 1, "next_epoch": 1, "next_rollout_cursor": 9,
                        "world_size": world, "signature": sig}
            save_checkpoint(opts.out, raw, optimizer, scheduler, accelerator, state, metadata)
    gathered = [None] * world
    dist.all_gather_object(gathered, trace)
    if accelerator.is_main_process:
        trace = {k: v for shard in gathered for k, v in shard.items()}
        result = cpu_tree({"model": raw.state_dict(), "optimizer": optimizer.state_dict(),
                           "scheduler": scheduler.state_dict(), "scaler": accelerator.scaler.state_dict(),
                           "trace": trace})
        if opts.mode == "continuous":
            torch.save(result, opts.out / "continuous.pt")
        else:
            expected = torch.load(opts.out / "continuous.pt", weights_only=False)
            for key, value in trace.items():
                assert value == expected["trace"][key], f"Random input mismatch: {key}"
            max_diff = max((tensor - expected["model"][k]).abs().max().item()
                           for k, tensor in result["model"].items())
            if world == 8:
                assert max_diff == 0., max_diff
                from test_resume import same
                import unittest
                same(unittest.TestCase(), result["optimizer"], expected["optimizer"])
                same(unittest.TestCase(), result["scheduler"], expected["scheduler"])
                same(unittest.TestCase(), result["scaler"], expected["scaler"])
            else:
                assert max_diff < .0001, max_diff
            print(json.dumps({"status": "PASS", "world": world, "input_records": len(trace),
                              "max_weight_difference": max_diff}), flush=True)
    accelerator.wait_for_everyone()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
