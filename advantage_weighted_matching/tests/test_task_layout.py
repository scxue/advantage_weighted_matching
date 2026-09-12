"""CPU coverage of task configs, text data, and topology-independent training inputs."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from config import dgx_awm
from flow_grpo.prompt_noise import prompt_training_seed, shared_prompt_training_inputs
from flow_grpo.resume import Legacy32Layout, LogicalEvalLoader, signature
from test_resume import logical_state, same


class TaskLayoutTests(unittest.TestCase):
    def test_task_configs_and_signatures(self):
        for task, nodes_list, beta, rounds, batch, noise, frequency in (
            ("geneval", (1, 2, 4), .003, 9, 3, .4, 50),
            ("ocr", (1, 4), .003, 9, 3, .4, 20),
            ("pickscore", (1, 4), .0001, 3, 2, .0, 50),
        ):
            reference = None
            for nodes in nodes_list:
                suffix = "1node" if nodes == 1 else f"{nodes}nodes"
                config = getattr(dgx_awm, f"{task}_sd3_no_cfg_{suffix}")()
                self.assertEqual(config.train.beta, beta)
                self.assertEqual(config.sample.noise_level, noise)
                self.assertEqual(config.eval_freq, frequency)
                self.assertEqual(config.train.learning_rate, .0003)
                self.assertEqual(config.train.ema_beta, 1)
                self.assertEqual(config.train.kl_ema_decay_rate, .001)
                self.assertEqual(config.train.ghuber_power, .5)
                self.assertEqual(config.train.base_kl_cfg_scale, 4.5)
                self.assertEqual(config.time_type, "discrete_with_init")
                self.assertTrue(config.train.share_prompt_training_noise)
                self.assertEqual(config.train.train_timesteps, 6)
                layout = Legacy32Layout.from_config(config, 997, nodes * 8, 0)
                self.assertEqual((layout.rounds, layout.train_batch), (rounds, batch))
                self.assertEqual(layout.images_per_rank * 32, 1728 if rounds == 9 else 576)
                self.assertEqual(config.train.gradient_accumulation_steps,
                                 layout.train_batches * len(layout.logical_ids))
                current = signature(config, [])
                if reference is None:
                    reference = current
                else:
                    self.assertEqual(current, reference)
                config.sample.num_batches_per_epoch += 1
                with self.assertRaises(ValueError):
                    Legacy32Layout.from_config(config, 997, nodes * 8, 0)
            config.logical_rounds += 1
            self.assertNotEqual(signature(config, []), reference)

    def test_shuffled_shared_inputs_world8_world16_world32(self):
        for rounds, batch in ((9, 3), (3, 2)):
            reference = None
            for world in (32, 16, 8):
                records, canonical_parts = {}, []
                for rank in range(world):
                    state = logical_state(world, rank, [torch.ones(2)],
                                          rounds=rounds, train_batch=batch)
                    layout = state.layout
                    pairs = []
                    for local_batch in range(rounds * len(layout.logical_ids)):
                        _, indices = layout.rollout(7, local_batch)
                        pairs.extend(zip(layout.sample_ids(7, local_batch), indices))
                    canonical_parts.append(torch.tensor(pairs))
                    saved = {r: state.state_for(r) for r in layout.logical_ids}
                    order = state.permutation()
                    for r, value in saved.items():
                        state.load_for(r, value)
                    self.assertTrue(torch.equal(order, state.permutation()))
                    pairs = [pairs[i] for i in order.tolist()]
                    for start in range(0, len(pairs), batch):
                        group = pairs[start:start + batch]
                        seeds = [prompt_training_seed(42, 7, f"p-{p}") for _, p in group]
                        times, noise = shared_prompt_training_inputs(
                            seeds, torch.linspace(1000, 10, 14).repeat(batch, 1),
                            torch.zeros(batch, 2, 4, 4), num_timesteps=6,
                            timestep_fraction=.9, time_type="discrete_with_init")
                        times, noise = times.reshape(6, batch), noise.reshape(6, batch, 2, 4, 4)
                        for j, (sid, prompt) in enumerate(group):
                            records[sid] = (prompt, times[:, j], noise[:, j])
                self.assertEqual(len(records), 32 * rounds * 6)
                canonical = Legacy32Layout(997, world, 0, rounds=rounds,
                                           train_batch=batch).canonical(torch.cat(canonical_parts))
                by_prompt = {}
                for prompt, times, noise in records.values():
                    if prompt in by_prompt:
                        same(self, (times, noise), by_prompt[prompt])
                    by_prompt[prompt] = (times, noise)
                if reference is None:
                    reference = records, canonical
                else:
                    same(self, (records, canonical), reference)

    def test_actual_trainer_diagnostic_and_rebatch(self):
        path = Path(__file__).resolve().parents[1] / "scripts/train_sd3_awm.py"
        main = next(n for n in ast.parse(path.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == "main")
        epoch = next(n for n in main.body if isinstance(n, ast.For)
                     and ast.unparse(n.target) == "epoch")
        body = epoch.body
        diagnostic = next(n for n in body if isinstance(n, ast.If)
                          and ast.unparse(n.test) == "logical" and "actual_sizes" in ast.unparse(n))
        start = next(i for i, n in enumerate(body) if isinstance(n, ast.Assign)
                     and ast.unparse(n.targets[0]) == "perm")
        end = next(i for i, n in enumerate(body) if isinstance(n, ast.Expr)
                   and ast.unparse(n.value) == "pipeline.transformer.train()")
        code = compile(ast.Module(body=[diagnostic, *body[start:end]], type_ignores=[]),
                       str(path), "exec")
        for task in ("geneval", "ocr", "pickscore"):
            config = getattr(dgx_awm, f"{task}_sd3_no_cfg_1node")()
            state = logical_state(8, 0, [torch.ones(2)], rounds=config.logical_rounds,
                                  train_batch=config.train.batch_size)
            layout = state.layout
            size = layout.images_per_rank * len(layout.logical_ids)
            samples = {"sample_id": torch.arange(size),
                       "advantages": (torch.arange(size) % 4 == 0).float().reshape(size, 1),
                       "timesteps": torch.linspace(1000, 10, 14).repeat(size, 1),
                       "latents": torch.zeros(size, 14, 1),
                       "next_latents": torch.zeros(size, 14, 1)}
            env = dict(torch=torch, np=np, logical=state, layout=layout, samples=samples,
                       total_batch_size=size, num_timesteps=14, config=config,
                       global_step=0, log_metrics=lambda *a, **kw: None,
                       accelerator=SimpleNamespace(device=torch.device("cpu")))
            exec(code, env)
            batches = env["samples_batched"]
            self.assertEqual(len(batches), config.train.num_batches_per_epoch)
            ids = []
            for i, sample in enumerate(batches):
                slot = i // layout.train_batches
                self.assertEqual(sample["_logical_rank"], layout.logical_ids[slot])
                self.assertEqual(len(sample["sample_id"]), config.train.batch_size)
                self.assertTrue(torch.all(sample["sample_id"] // layout.images_per_rank == slot))
                ids.extend(sample["sample_id"].tolist())
            self.assertEqual(sorted(ids), list(range(size)))

    def test_text_dataset_eval_and_fingerprint(self):
        # Use the real dataset/collate class without importing GPU training dependencies.
        import os
        import tempfile
        from torch.utils.data import Dataset
        path = Path(__file__).resolve().parents[1] / "scripts/train_sd3_awm.py"
        node = next(n for n in ast.parse(path.read_text()).body
                    if isinstance(n, ast.ClassDef) and n.name == "TextPromptDataset")
        scope = {"os": os, "Dataset": Dataset}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
        with tempfile.TemporaryDirectory() as directory:
            for split in ("train", "test"):
                (Path(directory) / f"{split}.txt").write_text("\n".join(f"prompt {i}" for i in range(197)))
            dataset = scope["TextPromptDataset"](directory, "test")
            references = None
            for world in (32, 8):
                records = {}
                for rank in range(world):
                    parameter = torch.nn.Parameter(torch.ones(2))
                    state = logical_state(world, rank, [parameter], rounds=3, train_batch=2)
                    for i, (prompts, metadata) in enumerate(LogicalEvalLoader(dataset, state, [parameter])):
                        self.assertEqual(metadata, [{}] * 6)
                        key = (i // len(state.layout.logical_ids),
                               state.layout.logical_ids[i % len(state.layout.logical_ids)])
                        records[key] = prompts, torch.randn(6)
                if references is None:
                    references = records
                else:
                    same(self, references, records)
            config = dgx_awm.pickscore_sd3_no_cfg_1node()
            files = [Path(directory) / f"{split}.txt" for split in ("train", "test")]
            before = signature(config, files)
            files[0].write_text("changed training data")
            self.assertNotEqual(signature(config, files), before)


if __name__ == "__main__":
    unittest.main()
