"""CPU regression tests; run from advantage_weighted_matching/ with unittest discover."""
import ast
import copy
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Sampler
from accelerate.data_loader import prepare_data_loader

from flow_grpo.ema import EMAModuleWrapper
from flow_grpo.resume import (
    Legacy32Layout, LogicalState, LogicalEvalLoader, capture_rng, preserve_rng,
    save_checkpoint, load_checkpoint, read_metadata, resolve_checkpoint, cpu_tree,
)


def same(test, a, b):
    if isinstance(a, torch.Tensor):
        test.assertTrue(torch.equal(a, b))
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        test.assertEqual(a.keys(), b.keys())
        for k in a:
            same(test, a[k], b[k])
    elif isinstance(a, (list, tuple)):
        test.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            same(test, x, y)
    else:
        test.assertEqual(a, b)


def logical_state(world, rank, parameters, device=torch.device("cpu"), *, rounds=9, train_batch=3):
    layout = Legacy32Layout(997, world, rank, rounds=rounds, train_batch=train_batch)
    rngs, emas = {}, {}
    with preserve_rng(device):
        for logical in layout.logical_ids:
            random.seed(42 + logical)
            np.random.seed(42 + logical)
            torch.manual_seed(42 + logical)
            rngs[logical] = capture_rng(device)
            emas[logical] = EMAModuleWrapper(parameters, decay=.99, device=device)
    return LogicalState(layout, device, rngs, emas)


def tiny_model(device=torch.device("cpu")):
    torch.manual_seed(17)
    model = torch.nn.Sequential(torch.nn.Linear(4, 4, bias=False))
    cfg = LoraConfig(r=2, lora_alpha=2, target_modules=["0"])
    model = get_peft_model(model, cfg).to(device)
    model.add_adapter("ema", cfg)
    model.set_adapter("default")
    return model


class Data:
    def __init__(self, size=997):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return index

    @staticmethod
    def collate_fn(records):
        return records


class ResumeTests(unittest.TestCase):
    def test_legacy_prepared_loader_exact(self):
        path = Path(__file__).resolve().parents[1] / "scripts/train_sd3_awm.py"
        tree = ast.parse(path.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DistributedKRepeatSampler")
        scope = {"torch": torch, "Sampler": Sampler}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
        for rank in (0, 15, 16, 31):
            sampler = scope[node.name](Data(), 6, 24, 32, rank, seed=42)
            loader = DataLoader(Data(), batch_sampler=sampler, num_workers=1, collate_fn=Data.collate_fn)
            prepared = prepare_data_loader(loader, num_processes=32, process_index=rank)
            iterator = iter(prepared)
            layout = Legacy32Layout(997, 32, rank)
            try:
                for q in range(460):  # Includes initial skips and the epoch-50 eval boundary.
                    sampler.set_epoch(q)
                    actual = next(iterator)
                    self.assertEqual(actual, layout.indices(q // 9, q % 9, rank), (rank, q))
                    pickscore = Legacy32Layout(997, 32, rank, rounds=3, train_batch=2)
                    self.assertEqual(actual, pickscore.indices(q // 3, q % 3, rank), (rank, q))
            finally:
                iterator.close()

    def test_canonical_and_sample_ids(self):
        expected = torch.arange(32 * 54 * 2).reshape(32 * 54, 2)
        for world in (1, 4, 8, 16, 32):
            pieces = [Legacy32Layout(997, world, r).local(expected) for r in range(world)]
            same(self, Legacy32Layout(997, world, 0).canonical(torch.cat(pieces)), expected)
            records = {}
            for rank in range(world):
                layout = Legacy32Layout(997, world, rank)
                for b in range(9 * len(layout.logical_ids)):
                    _, indices = layout.rollout(21, b)
                    records.update(zip(layout.sample_ids(21, b), indices))
            if world == 1:
                reference = records
            self.assertEqual(records, reference)

    def test_random_streams_world_size_and_resume(self):
        def draw(state):
            result = {}
            for logical in state.layout.logical_ids:
                with state.use(logical):
                    result[logical] = [random.random(), np.random.rand(), torch.randn(6, 16, 4, 4)]
                    # SA-solver's internal noise and timestep/noise generation shapes.
                    result[logical] += [torch.randn(6, 16, 4, 4) for _ in range(14)]
                    result[logical] += [torch.rand(5), torch.randn(3, 16, 4, 4), torch.randperm(54)]
            return result
        reference = {}
        for rank in range(32):
            reference.update(draw(logical_state(32, rank, [torch.ones(2)])))
        for rank in range(16):
            state = logical_state(16, rank, [torch.ones(2)])
            ambient = capture_rng(torch.device("cpu"))
            actual = draw(state)
            for logical, result in actual.items():
                same(self, result, reference[logical])
            same(self, ambient, capture_rng(torch.device("cpu")))
            saved = {r: state.state_for(r) for r in state.layout.logical_ids}
            uninterrupted = draw(state)
            for logical, data in saved.items():
                state.load_for(logical, data)
            same(self, uninterrupted, draw(state))

    def test_eval_padding_rng_ema_and_policy_restoration(self):
        references = {}
        for world in (32, 16):
            actual = {}
            for rank in range(world):
                parameter = torch.nn.Parameter(torch.ones(2))
                state = logical_state(world, rank, [parameter])
                for logical, ema in state.emas.items():
                    ema.ema_parameters[0].fill_(logical)
                loader = LogicalEvalLoader(Data(2212), state, [parameter])
                for ema_mode in (False, True):
                    loader.ema_mode = ema_mode
                    for i, records in enumerate(loader):
                        logical = state.layout.logical_ids[i % len(state.layout.logical_ids)]
                        key = (ema_mode, logical, i // len(state.layout.logical_ids))
                        self.assertEqual(parameter[0].item(), logical if ema_mode else 1)
                        actual[key] = (records, torch.randn(6, 2))
                    self.assertEqual(parameter[0].item(), 1)
            if world == 32:
                references = actual
            same(self, references, actual)

    def test_eval_matches_original_accelerate_loader(self):
        for rank in range(32):
            parameter = torch.nn.Parameter(torch.ones(2))
            state = logical_state(32, rank, [parameter])
            loader = DataLoader(Data(2212), batch_size=6, shuffle=False, num_workers=0,
                                collate_fn=Data.collate_fn)
            old = prepare_data_loader(loader, num_processes=32, process_index=rank)
            before = state.state_for(rank)
            with state.use(rank):
                expected = [(records, torch.randn(6, 2)) for records in old]
            state.load_for(rank, before)
            actual = [(records, torch.randn(6, 2)) for records in LogicalEvalLoader(Data(2212), state, [parameter])]
            same(self, expected, actual)

    def test_full_checkpoint_and_next_update(self):
        for rounds, train_batch in ((9, 3), (3, 2)):
            with self.subTest(rounds=rounds, train_batch=train_batch):
                self.check_full_checkpoint_and_next_update(rounds, train_batch)

    def check_full_checkpoint_and_next_update(self, rounds, train_batch):
        model = tiny_model()
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, .9)
        state = logical_state(1, 0, parameters, rounds=rounds, train_batch=train_batch)
        accumulation = state.layout.train_batches * 32
        acc = SimpleNamespace(device=torch.device("cpu"), process_index=0, num_processes=1,
                              is_main_process=True, scaler=None, step=accumulation,
                              gradient_accumulation_steps=accumulation, wait_for_everyone=lambda: None)
        signature = {"test": 1, "config": {"train": {"ema": True}}}
        def update():
            with state.use(0):
                loss = model(torch.randn(train_batch, 4)).square().mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            state.update_ema(parameters, 1)
            return loss.detach()
        update()
        metadata = {"global_step": 1, "next_epoch": 3, "next_rollout_cursor": 3 * rounds,
                    "world_size": 1, "signature": signature}
        with tempfile.TemporaryDirectory() as root:
            path = save_checkpoint(root, model, optimizer, scheduler, acc, state, metadata)
            self.assertEqual(read_metadata(root)[1], metadata)
            self.assertTrue((path / "lora_non_ema/adapter_model.safetensors").is_file())
            self.assertTrue((path / "lora_ema/adapter_model.safetensors").is_file())
            expected_loss = update()
            expected = cpu_tree((model.state_dict(), optimizer.state_dict(), scheduler.state_dict(), state.state_for(0)))
            torch.manual_seed(923)
            load_checkpoint(root, model, optimizer, scheduler, acc, state, signature)
            same(self, update(), expected_loss)
            same(self, (model.state_dict(), optimizer.state_dict(), scheduler.state_dict(), state.state_for(0)), expected)
            self.assertEqual(acc.step, accumulation)
            wrong_layout = logical_state(1, 0, parameters, rounds=rounds + 1, train_batch=train_batch)
            with self.assertRaisesRegex(RuntimeError, "epoch boundary"):
                load_checkpoint(root, model, optimizer, scheduler, acc, wrong_layout, signature)
            with self.assertRaises(RuntimeError):
                load_checkpoint(root, model, optimizer, scheduler, acc, state, {"different": True})
            with self.assertRaises(RuntimeError):
                save_checkpoint(root, model, optimizer, scheduler, acc, state, metadata)
            incomplete = Path(root) / "checkpoints/checkpoint-999"
            incomplete.mkdir()
            self.assertEqual(resolve_checkpoint(root), path)
            with self.assertRaises(ValueError):
                resolve_checkpoint(incomplete)
            (path / "logical-31.pt").unlink()
            with self.assertRaises(ValueError):
                read_metadata(root)


if __name__ == "__main__":
    unittest.main()
