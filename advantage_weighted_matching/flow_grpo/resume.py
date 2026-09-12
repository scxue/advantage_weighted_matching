"""Epoch-boundary checkpoints and legacy 32-rank task input schedules.

Logical rank IDs, never physical CUDA IDs, identify persistent training state.
The input schedule deliberately preserves the old loader's three-batch lag.
"""
import contextlib
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import uuid
from importlib.metadata import version

import numpy as np
import torch
import torch.distributed as dist


SCHEDULE_VERSION = "geneval-legacy32-prefetch3-v1"


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return copy.deepcopy(value)


def capture_rng(device):
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "cpu": torch.get_rng_state().clone(),
        "cuda": torch.cuda.get_rng_state(device).cpu() if device.type == "cuda" else None,
    }


def restore_rng(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["cpu"].cpu())
    if device.type == "cuda":
        if state["cuda"] is None:
            raise ValueError("CUDA RNG is missing from the checkpoint")
        torch.cuda.set_rng_state(state["cuda"].cpu(), device)


@contextlib.contextmanager
def preserve_rng(device):
    state = capture_rng(device)
    try:
        yield
    finally:
        restore_rng(state, device)


class Legacy32Layout:
    logical_world = 32
    rounds = 9
    rollout_batch = 6
    train_batch = 3
    group_size = 24
    images_per_rank = rounds * rollout_batch

    def __init__(self, dataset_size, world_size, rank, *, rounds=9, train_batch=3):
        if world_size not in (1, 2, 4, 8, 16, 32):
            raise ValueError("Legacy32 requires a process count dividing 32 (smaller counts support smoke tests)")
        if dataset_size < 8 or not 0 <= rank < world_size:
            raise ValueError("Invalid dataset size or rank")
        if not isinstance(rounds, int) or rounds < 1 or train_batch not in (2, 3):
            raise ValueError("Expected positive logical rounds and train batch 2 or 3")
        self.rounds, self.train_batch = rounds, train_batch
        self.images_per_rank = self.rounds * self.rollout_batch
        self.train_batches = self.images_per_rank // self.train_batch
        self.dataset_size, self.world_size, self.rank = dataset_size, world_size, rank
        self.logical_ids = list(range(rank, self.logical_world, world_size))

    @classmethod
    def from_config(cls, config, dataset_size, world_size, rank):
        if config.logical_ranks != 32 or config.prompt_fn not in ("geneval", "general_ocr") or not config.use_lora:
            raise ValueError("Logical inputs require 32 logical ranks and Geneval/text-prompt LoRA")
        layout = cls(dataset_size, world_size, rank,
                     rounds=config.logical_rounds, train_batch=config.train.batch_size)
        slots = len(layout.logical_ids)
        actual = (config.sample.train_batch_size, config.sample.num_image_per_prompt,
                  config.sample.num_batches_per_epoch, config.train.num_batches_per_epoch,
                  config.train.gradient_accumulation_steps, config.sample.test_batch_size)
        expected = (6, 24, layout.rounds * slots, layout.train_batches * slots,
                    layout.train_batches * slots, 6)
        if actual != expected:
            raise ValueError(f"Config/GPU count does not match logical layout: {actual} != {expected}")
        if config.train.off_policy or config.train.num_inner_epochs != 1:
            raise ValueError("Logical inputs require on-policy training with one inner epoch")
        return layout

    def indices(self, epoch, round_index, logical_rank):
        if epoch < 0 or not 0 <= round_index < self.rounds or not 0 <= logical_rank < 32:
            raise ValueError("Invalid logical input cursor")
        q = epoch * self.rounds + round_index
        # The old sampler's seed is hard-coded to 42, independent of config.seed.
        generator = torch.Generator().manual_seed(42 + max(0, q - 3))
        selected = torch.randperm(self.dataset_size, generator=generator)[:8]
        repeated = selected.repeat_interleave(self.group_size)
        shuffled = repeated[torch.randperm(192, generator=generator)]
        return shuffled[logical_rank * 6:(logical_rank + 1) * 6].tolist()

    def rollout(self, epoch, local_batch):
        slot, round_index = divmod(local_batch, self.rounds)
        logical_rank = self.logical_ids[slot]
        return logical_rank, self.indices(epoch, round_index, logical_rank)

    def sample_ids(self, epoch, local_batch):
        slot, round_index = divmod(local_batch, self.rounds)
        start = (epoch * self.rounds + round_index) * 192 + self.logical_ids[slot] * 6
        return list(range(start, start + 6))

    def canonical(self, gathered):
        """Physical-rank/slot order -> original logical-rank order."""
        shape = gathered.shape
        return gathered.reshape(self.world_size, len(self.logical_ids), -1, *shape[1:]).transpose(0, 1).reshape(shape)

    def local(self, canonical):
        shape = canonical.shape
        return canonical.reshape(32, -1, *shape[1:])[self.logical_ids].reshape(-1, *shape[1:])


class LogicalState:
    def __init__(self, layout, device, rngs, emas):
        self.layout, self.device = layout, torch.device(device)
        self.rngs, self.emas = rngs, emas

    @contextlib.contextmanager
    def use(self, logical_rank):
        with preserve_rng(self.device):
            restore_rng(self.rngs[logical_rank], self.device)
            try:
                yield
            finally:
                self.rngs[logical_rank] = capture_rng(self.device)

    def permutation(self):
        pieces = []
        for slot, logical_rank in enumerate(self.layout.logical_ids):
            with self.use(logical_rank):
                size = self.layout.images_per_rank
                pieces.append(torch.randperm(size, device=self.device) + slot * size)
        return torch.cat(pieces)

    def update_ema(self, parameters, step):
        for ema in self.emas.values():
            ema.step(parameters, step)

    def state_for(self, logical_rank):
        return {"rng": cpu_tree(self.rngs[logical_rank]),
                "ema": cpu_tree(self.emas[logical_rank].state_dict())}

    def load_for(self, logical_rank, state):
        # Validate RNG eagerly: a missing RNG must not silently produce a warm start.
        with preserve_rng(self.device):
            restore_rng(state["rng"], self.device)
        self.rngs[logical_rank] = state["rng"]
        self.emas[logical_rank].load_state_dict(state["ema"])


class LogicalEvalLoader:
    """Same batches, padding, RNG streams and per-rank EMA as the old 32-GPU eval."""
    def __init__(self, dataset, state, parameters):
        self.dataset, self.state, self.parameters = dataset, state, parameters
        self.ema_mode = False

    def __len__(self):
        return math.ceil(len(self.dataset) / 192) * len(self.state.layout.logical_ids)

    def __iter__(self):
        for logical_rank in self.state.layout.logical_ids:
            with self.state.use(logical_rank):
                # Original DataLoader iterator draws one CPU worker base seed per pass.
                torch.empty((), dtype=torch.int64).random_()
        for round_index in range(math.ceil(len(self.dataset) / 192)):
            for logical_rank in self.state.layout.logical_ids:
                start = (round_index * 32 + logical_rank) * 6
                records = [self.dataset[i % len(self.dataset)] for i in range(start, start + 6)]
                with self.state.use(logical_rank):
                    ema = self.state.emas[logical_rank]
                    if self.ema_mode:
                        ema.copy_ema_to(self.parameters, store_temp=True)
                    try:
                        yield self.dataset.collate_fn(records)
                    finally:
                        if self.ema_mode:
                            ema.copy_temp_to(self.parameters)


def file_hash(path):
    hasher = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def signature(config, dataset_files):
    values = copy.deepcopy(config.to_dict())
    for key in ("run_name", "save_dir", "resume_from", "logdir", "num_epochs", "save_freq",
                "checkpoint_every_steps", "num_checkpoint_limit", "wandb_run_id", "dataset"):
        values.pop(key, None)
    # These are physical execution parameters; validated against the logical layout separately.
    values["sample"].pop("num_batches_per_epoch", None)
    values["train"].pop("gradient_accumulation_steps", None)
    values["train"].pop("num_batches_per_epoch", None)
    values["train"].pop("lora_path", None)
    package = Path(__file__).resolve().parent
    sources = [package / "resume.py", package / "ema.py", package / "stat_tracking.py",
               package / "prompt_noise.py",
               package / "wandb_logging.py",
               package.parent / "scripts/train_sd3_awm.py"]
    sources += sorted((package / "diffusers_patch").glob("*.py"))
    return {"schedule": SCHEDULE_VERSION, "config": values,
            "data": {Path(p).name: file_hash(p) for p in dataset_files},
            "software": {name: version(name) for name in ("torch", "accelerate", "diffusers", "peft", "transformers")},
            "source": {str(p.relative_to(package.parent)): file_hash(p) for p in sources}}


def broadcast(value):
    if dist.is_initialized():
        box = [value]
        dist.broadcast_object_list(box, src=0)
        return box[0]
    return value


def resolve_checkpoint(path):
    path = Path(path).expanduser().resolve()
    if (path / "complete.json").is_file():
        return path
    if path.name.startswith("checkpoint-"):
        raise ValueError(f"{path} is not a complete resumable checkpoint (old LoRA exports are warm-start only)")
    root = path / "checkpoints" if (path / "checkpoints").is_dir() else path
    candidates = [p for p in root.glob("checkpoint-*")
                  if p.name.removeprefix("checkpoint-").isdigit() and (p / "complete.json").is_file()]
    if not candidates:
        raise ValueError(f"No complete training checkpoint in {path}; LoRA-only checkpoints cannot restore Adam/RNG")
    return max(candidates, key=lambda p: int(p.name.split("-")[-1]))


def read_metadata(path):
    path = resolve_checkpoint(path)
    manifest = json.loads((path / "complete.json").read_text())
    if manifest["version"] != 1:
        raise ValueError("Unsupported checkpoint format")
    for name, size in manifest["files"].items():
        if not (path / name).is_file() or (path / name).stat().st_size != size:
            raise ValueError(f"Incomplete or truncated checkpoint file: {path / name}")
    if file_hash(path / "metadata.json") != manifest["metadata_sha256"]:
        raise ValueError("Checkpoint metadata checksum mismatch")
    metadata = json.loads((path / "metadata.json").read_text())
    required = {"training.pt", "metadata.json"} | {f"logical-{i}.pt" for i in range(32)}
    required |= {f"physical-{i}.pt" for i in range(metadata["world_size"])}
    if not required.issubset(manifest["files"]):
        raise ValueError("Checkpoint manifest is missing required state files")
    return path, metadata


def check_collective_error(error):
    """Propagate caught filesystem failures instead of stranding peers at a barrier."""
    errors = [error]
    if dist.is_initialized():
        errors = [None] * dist.get_world_size()
        dist.all_gather_object(errors, error)
    if any(errors):
        raise RuntimeError("Checkpoint failed: " + "; ".join(e for e in errors if e))


def save_checkpoint(save_dir, model, optimizer, scheduler, accelerator, logical, metadata):
    from peft.utils import get_peft_model_state_dict
    step = metadata["global_step"]
    root = Path(save_dir) / "checkpoints"
    destination = root / f"checkpoint-{step}"
    # All ranks use one path; a killed writer leaves an ignored .incomplete directory.
    token = broadcast(uuid.uuid4().hex if accelerator.is_main_process else None)
    temporary = root / f".incomplete-{step}-{token}"
    error = None
    try:
        if accelerator.is_main_process:
            if destination.exists():
                raise FileExistsError(f"Refusing to overwrite checkpoint {destination}")
            temporary.mkdir(parents=True)
    except Exception as exc:
        error = repr(exc)
    check_collective_error(error)
    ambient = capture_rng(accelerator.device)
    error = None
    try:
        for logical_rank in logical.layout.logical_ids:
            torch.save(logical.state_for(logical_rank), temporary / f"logical-{logical_rank}.pt")
        torch.save(ambient, temporary / f"physical-{accelerator.process_index}.pt")
        if accelerator.is_main_process:
            payload = {
                "adapters": {name: cpu_tree(get_peft_model_state_dict(model, adapter_name=name))
                             for name in ("default", "ema")},
                "optimizer": cpu_tree(optimizer.state_dict()),
                "scheduler": scheduler.state_dict(),
                "scaler": accelerator.scaler.state_dict() if accelerator.scaler is not None else None,
                "accelerator_step": accelerator.step,
            }
            torch.save(payload, temporary / "training.pt")
            # Keep the existing inference/warm-start export paths.
            model.save_pretrained(temporary / "lora_non_ema")
            if metadata["signature"].get("config", {}).get("train", {}).get("ema", False):
                parameters = [p for p in model.parameters() if p.requires_grad]
                shadow = logical.emas[0]
                shadow.copy_ema_to(parameters, store_temp=True)
                try:
                    model.save_pretrained(temporary / "lora_ema")
                finally:
                    shadow.copy_temp_to(parameters)
            (temporary / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    except Exception as exc:
        error = repr(exc)
    check_collective_error(error)
    error = None
    try:
        if accelerator.is_main_process:
            sizes = {str(p.relative_to(temporary)): p.stat().st_size
                     for p in temporary.rglob("*") if p.is_file()}
            marker = {"version": 1, "files": sizes, "metadata_sha256": file_hash(temporary / "metadata.json")}
            (temporary / "complete.json").write_text(json.dumps(marker, indent=2))
            os.replace(temporary, destination)
    except Exception as exc:
        error = repr(exc)
    check_collective_error(error)
    restore_rng(ambient, accelerator.device)
    return destination


def _load_checkpoint_local(path, model, optimizer, scheduler, accelerator, logical, expected_signature):
    from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
    path, metadata = read_metadata(path)
    if metadata["signature"] != expected_signature:
        raise ValueError("Resume config/data/software signature differs; use an explicit warm start for a new experiment")
    if metadata["next_epoch"] * logical.layout.rounds != metadata["next_rollout_cursor"]:
        raise ValueError("Checkpoint is not at a complete epoch boundary")
    payload = torch.load(path / "training.pt", map_location="cpu", weights_only=False)
    if set(payload["adapters"]) != {"default", "ema"}:
        raise ValueError("Checkpoint must contain both policy and KL-reference adapters")
    for name, weights in payload["adapters"].items():
        current = get_peft_model_state_dict(model, adapter_name=name)
        if set(current) != set(weights) or any(current[k].shape != weights[k].shape for k in current):
            raise ValueError(f"Adapter schema mismatch: {name}")
        set_peft_model_state_dict(model, weights, adapter_name=name)
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    if (accelerator.scaler is None) != (payload["scaler"] is None):
        raise ValueError("GradScaler/mixed-precision mismatch")
    if accelerator.scaler is not None:
        accelerator.scaler.load_state_dict(payload["scaler"])
    # At an epoch boundary there are no outstanding accumulated gradients.
    # Rebase the microstep counter when the physical accumulation factor changes.
    accelerator.step = metadata["global_step"] * accelerator.gradient_accumulation_steps
    optimizer.zero_grad()
    for logical_rank in logical.layout.logical_ids:
        logical.load_for(logical_rank, torch.load(path / f"logical-{logical_rank}.pt", map_location="cpu", weights_only=False))
    model.set_adapter("default")
    if metadata["world_size"] == accelerator.num_processes:
        restore_rng(torch.load(path / f"physical-{accelerator.process_index}.pt", map_location="cpu", weights_only=False), accelerator.device)
    return metadata


def load_checkpoint(path, model, optimizer, scheduler, accelerator, logical, expected_signature):
    error, metadata = None, None
    try:
        metadata = _load_checkpoint_local(path, model, optimizer, scheduler, accelerator, logical, expected_signature)
    except Exception as exc:
        error = repr(exc)
    check_collective_error(error)
    return metadata
