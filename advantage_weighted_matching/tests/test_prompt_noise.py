"""CPU tests of shared randomness and the actual trainer's input-preparation block."""
import ast
import contextlib
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from diffusers.utils.torch_utils import randn_tensor
from ml_collections import ConfigDict

from flow_grpo.prompt_noise import prompt_training_seed, shared_prompt_training_inputs
from flow_grpo.resume import Legacy32Layout, signature


def seeds(prompts, epoch=2, seed=42):
    return torch.tensor([prompt_training_seed(seed, epoch, p) for p in prompts])


def inputs(prompts, epoch=2, time_type="discrete_with_init", device="cpu"):
    return shared_prompt_training_inputs(
        seeds(prompts, epoch).to(device),
        torch.linspace(1000, 10, 14, device=device).repeat(len(prompts), 1),
        torch.empty(len(prompts), 2, 4, 4, dtype=torch.float16, device=device),
        num_timesteps=6, timestep_fraction=.9, time_type=time_type)


def actual_preparation():
    # Execute the production loop itself, without loading SD3 or reward servers.
    path = Path(__file__).resolve().parents[1] / "scripts/train_sd3_awm.py"
    tree = ast.parse(path.read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    loop = next(n for n in ast.walk(main) if isinstance(n, ast.For)
                and "Calculating old log_probs; off-policy sample only" in ast.unparse(n.iter))
    scope = {"torch": torch, "contextlib": contextlib, "randn_tensor": randn_tensor,
             "shared_prompt_training_inputs": shared_prompt_training_inputs,
             "tqdm": lambda items, **kw: items, "logical": None, "epoch": 2,
             "accelerator": SimpleNamespace(device=torch.device("cpu"), is_local_main_process=False)}
    return compile(ast.Module(body=[loop], type_ignores=[]), str(path), "exec"), scope


class PromptNoiseTests(unittest.TestCase):
    def test_same_prompt_across_batches_and_order(self):
        t, noise = inputs(["a", "b", "a"])
        t, noise = t.reshape(6, 3), noise.reshape(6, 3, 2, 4, 4)
        self.assertTrue(torch.equal(t[:, 0], t[:, 2]))
        self.assertTrue(torch.equal(noise[:, 0], noise[:, 2]))
        other_t, other_noise = inputs(["b", "a"])
        self.assertTrue(torch.equal(t[:, :2].flip(1), other_t.reshape(6, 2)))
        self.assertTrue(torch.equal(noise[:, :2].flip(1), other_noise.reshape(6, 2, 2, 4, 4)))
        self.assertEqual(noise.dtype, torch.float16)

    def test_prompt_epoch_seed_and_time_noise_independence(self):
        t, noise = inputs(["a", "b"])
        noise = noise.reshape(6, 2, 2, 4, 4)
        for slot in range(6):
            self.assertFalse(torch.equal(noise[slot, 0], noise[slot, 1]))
        for slot in range(1, 6):
            self.assertFalse(torch.equal(noise[0, 0], noise[slot, 0]))
        other_t, other_noise = inputs(["a", "b"], epoch=3)
        self.assertFalse(torch.equal(noise.flatten(), other_noise.flatten()))
        self.assertFalse(torch.equal(t, other_t))
        self.assertNotEqual(prompt_training_seed(42, 2, "a"), prompt_training_seed(43, 2, "a"))
        self.assertEqual(prompt_training_seed(42, 2, "a"), prompt_training_seed(42, 2, "a"))

    def test_time_strata_and_init(self):
        for mode in ["discrete_with_init", "discrete_wo_init", "discrete"]:
            t, _ = inputs(["a", "b"], time_type=mode)
            schedule = torch.linspace(1000, 10, 14) / 1000
            indices = (t.reshape(6, 2, 1) - schedule).abs().argmin(-1)
            count = 5 if mode == "discrete_with_init" else 6
            offset = 1 if mode == "discrete_with_init" else 0
            boundaries = torch.linspace(0 if mode == "discrete" else 1, 12, count + 1).long()
            self.assertTrue(torch.all(indices[offset:] >= boundaries[:-1, None]))
            self.assertTrue(torch.all(indices[offset:] < boundaries[1:, None]))
            if offset:
                self.assertTrue(torch.all(indices[0] == 0))

    def test_global_rng_and_stateless_resume(self):
        torch.manual_seed(123)
        before = torch.get_rng_state().clone()
        expected = inputs(["a", "b"])
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        torch.randn(71)
        # No saved cache/counter is needed; a fresh helper call resumes exactly.
        resumed = inputs(["a", "b"])
        for a, b in zip(expected, resumed):
            self.assertTrue(torch.equal(a, b))

    def test_actual_trainer_sharing_and_unchanged_rollout_rng(self):
        code, scope = actual_preparation()
        config = ConfigDict({"time_type": "discrete_with_init", "sample": {"num_steps": 14},
                             "train": {"cfg": False, "off_policy": False, "batch_size": 3,
                                       "train_timesteps": 6, "timestep_fraction": .9,
                                       "gradient_accumulation_steps": 2}})
        results = []
        for enabled in [False, True]:
            batches = []
            for prompts in [["a", "b", "a"], ["b", "a", "c"]]:
                batches.append({"prompt_noise_seed": seeds(prompts),
                                "prompt_embeds": torch.zeros(3, 2, 4),
                                "pooled_prompt_embeds": torch.zeros(3, 4),
                                "timesteps": torch.linspace(1000, 10, 14).repeat(3, 1),
                                "latents": torch.zeros(3, 14, 2, 4, 4),
                                "next_latents": torch.arange(3.).reshape(3, 1, 1, 1, 1).expand(3, 14, 2, 4, 4)})
            env = dict(scope, config=config, samples_batched=batches, share_prompt_noise=enabled)
            torch.manual_seed(1234)
            exec(code, env)
            results.append((batches, torch.get_rng_state().clone(), torch.randn(6, 2, 4, 4)))
        self.assertTrue(torch.equal(results[0][1], results[1][1]))
        self.assertTrue(torch.equal(results[0][2], results[1][2]))  # Next rollout noise unchanged.
        batches = results[1][0]
        a = batches[0]["train_random_noise"].reshape(6, 3, 2, 4, 4)
        b = batches[1]["train_random_noise"].reshape(6, 3, 2, 4, 4)
        self.assertTrue(torch.equal(a[:, 0], a[:, 2]))
        self.assertTrue(torch.equal(a[:, 0], b[:, 1]))
        self.assertTrue(torch.equal(a[:, 1], b[:, 0]))
        self.assertFalse(torch.equal(results[0][0][0]["train_random_noise"], batches[0]["train_random_noise"]))
        for batch in batches:
            self.assertTrue(torch.equal(batch["clean_latents"][:, 0, 0, 0], torch.arange(3.).repeat(6)))

    def test_actual_prompt_keys_follow_shuffle_and_rebatch(self):
        path = Path(__file__).resolve().parents[1] / "scripts/train_sd3_awm.py"
        main = next(n for n in ast.parse(path.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == "main")
        epoch_loop = next(n for n in main.body if isinstance(n, ast.For)
                          and isinstance(n.target, ast.Name) and n.target.id == "epoch")
        body = epoch_loop.body
        attach = next(n for n in body if isinstance(n, ast.If)
                      and ast.unparse(n.test) == "share_prompt_noise")
        start = next(i for i, n in enumerate(body) if isinstance(n, ast.Assign)
                     and ast.unparse(n.targets[0]) == "perm")
        end = next(i for i, n in enumerate(body) if isinstance(n, ast.Expr)
                   and ast.unparse(n.value) == "pipeline.transformer.train()")
        tokenizer = SimpleNamespace(batch_decode=lambda ids, **kw: [f"p-{int(row[0])}" for row in ids])
        samples = {"prompt_ids": torch.tensor([[7], [8], [7], [9], [7], [8]]),
                   "timesteps": torch.linspace(1000, 10, 14).repeat(6, 1),
                   "latents": torch.zeros(6, 14, 2, 4, 4),
                   "next_latents": torch.zeros(6, 14, 2, 4, 4)}
        env = {"torch": torch, "share_prompt_noise": True, "epoch": 2,
               "samples": samples, "pipeline": SimpleNamespace(tokenizer=tokenizer),
               "prompt_training_seed": prompt_training_seed, "logical": None,
               "total_batch_size": 6, "num_timesteps": 14,
               "config": ConfigDict({"seed": 42, "train": {"num_batches_per_epoch": 2}}),
               "accelerator": SimpleNamespace(device=torch.device("cpu"))}
        code = compile(ast.Module(body=[attach, *body[start:end]], type_ignores=[]), str(path), "exec")
        exec(code, env)
        for batch in env["samples_batched"]:
            prompts = tokenizer.batch_decode(batch["prompt_ids"])
            self.assertTrue(torch.equal(batch["prompt_noise_seed"], seeds(prompts)))

    def test_world16_world32_full_epoch_prompt_groups(self):
        reference = None
        for world in [32, 16]:
            records = {}
            for rank in range(world):
                layout = Legacy32Layout(997, world, rank)
                for logical in layout.logical_ids:
                    pairs = []
                    for rnd in range(9):
                        ids = layout.indices(7, rnd, logical)
                        pairs += [((7 * 9 + rnd) * 192 + logical * 6 + j, f"p-{p}")
                                  for j, p in enumerate(ids)]
                    order = torch.randperm(54, generator=torch.Generator().manual_seed(42 + logical)).tolist()
                    pairs = [pairs[j] for j in order]
                    for start in range(0, 54, 3):
                        group = pairs[start:start + 3]
                        t, noise = inputs([p for _, p in group], epoch=7)
                        t, noise = t.reshape(6, 3), noise.reshape(6, 3, 2, 4, 4)
                        for j, (sid, prompt) in enumerate(group):
                            records[sid] = prompt, t[:, j].clone(), noise[:, j].clone()
            self.assertEqual(len(records), 1728)
            if reference is None:
                reference = records
                by_prompt = {}
                for prompt, t, noise in records.values():
                    if prompt in by_prompt:
                        self.assertTrue(torch.equal(t, by_prompt[prompt][0]))
                        self.assertTrue(torch.equal(noise, by_prompt[prompt][1]))
                    by_prompt[prompt] = t, noise
            else:
                self.assertEqual(records.keys(), reference.keys())
                for sid, (prompt, t, noise) in records.items():
                    self.assertEqual(prompt, reference[sid][0])
                    self.assertTrue(torch.equal(t, reference[sid][1]))
                    self.assertTrue(torch.equal(noise, reference[sid][2]))

    def test_configs_and_checkpoint_signature(self):
        from config import dgx_awm
        factories = {name for name in vars(dgx_awm) if name.startswith("geneval_")}
        self.assertEqual(factories, {"geneval_sd3_no_cfg_1node", "geneval_sd3_no_cfg_2nodes",
                                    "geneval_sd3_no_cfg_4nodes"})
        reference = None
        for nodes, factory in [(1, dgx_awm.geneval_sd3_no_cfg_1node),
                               (2, dgx_awm.geneval_sd3_no_cfg_2nodes),
                               (4, dgx_awm.geneval_sd3_no_cfg_4nodes)]:
            config = factory()
            self.assertTrue(config.train.share_prompt_training_noise)
            self.assertEqual(config.train.train_timesteps, 6)
            self.assertEqual(config.logical_ranks, 32)
            self.assertEqual(config.train.beta, 0.003)
            self.assertEqual(config.train.ghuber_power, 0.5)
            self.assertEqual(config.time_type, "discrete_with_init")
            self.assertEqual(config.train.base_kl_cfg_scale, 4.5)
            self.assertEqual(config.train.kl_ema_decay_rate, 0.001)
            self.assertEqual(config.train.ema_beta, 1)
            self.assertEqual((config.train.adam_beta1, config.train.adam_beta2), (0.9, 0.999))
            self.assertEqual(config.sample.num_batches_per_epoch, 36 // nodes)
            self.assertEqual(config.train.num_batches_per_epoch, 72 // nodes)
            self.assertEqual(config.train.gradient_accumulation_steps, 72 // nodes)
            self.assertEqual(config.sample.train_batch_size * config.sample.num_batches_per_epoch
                             * nodes * 8, 1728)
            expected = signature(config, [])
            if reference is None:
                reference = expected
            else:
                self.assertEqual(expected, reference)
            self.assertIn("flow_grpo/prompt_noise.py", expected["source"])
            config.train.share_prompt_training_noise = False
            self.assertNotEqual(signature(config, []), expected)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable on login node")
    def test_cuda_generators_do_not_advance_ambient_rng(self):
        torch.cuda.init()
        before = torch.cuda.get_rng_state().clone()
        t, noise = inputs(["a", "b", "a"], device="cuda")
        self.assertTrue(torch.equal(before, torch.cuda.get_rng_state()))
        self.assertTrue(torch.equal(t.reshape(6, 3)[:, 0], t.reshape(6, 3)[:, 2]))
        noise = noise.reshape(6, 3, 2, 4, 4)
        self.assertTrue(torch.equal(noise[:, 0], noise[:, 2]))


if __name__ == "__main__":
    unittest.main()
