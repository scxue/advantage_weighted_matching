"""Compare final states emitted by train_entrypoint_smoke.py."""
import argparse
import json
import unittest
import torch
from test_resume import same


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    args = parser.parse_args()
    def read(mode, world, rank):
        return torch.load(f"{args.directory}/{mode}-{world}-rank{rank}.pt", map_location="cpu", weights_only=False)
    expected = read("continuous", 8, 0)
    train_batches = expected.get("train_batches_per_logical_rank", 18)
    expected_logical = {}
    expected_inputs = {}
    for rank in range(8):
        shard = read("continuous", 8, rank)
        expected_logical.update(shard["logical"])
        for slot, logical in enumerate(range(rank, 32, 8)):
            expected_inputs[logical] = shard["input_trace"][slot*train_batches:(slot+1)*train_batches]
    test = unittest.TestCase()
    for world in (8, 4):
        max_model_difference = 0.
        max_ema_difference = 0.
        for rank in range(world):
            actual = read("resume", world, rank)
            same(test, actual["scheduler"], expected["scheduler"])
            same(test, actual["scaler"], expected["scaler"])
            if world == 8:
                # Fresh DDP rebuilds its reduction buckets; allow FP32 roundoff.
                for key, tensor in actual["model"].items():
                    torch.testing.assert_close(tensor, expected["model"][key], rtol=0, atol=1e-9)
            for slot, logical in enumerate(range(rank, 32, world)):
                same(test, actual["input_trace"][slot*train_batches:(slot+1)*train_batches], expected_inputs[logical])
            for key, tensor in actual["model"].items():
                max_model_difference = max(max_model_difference, (tensor - expected["model"][key]).abs().max().item())
            for logical, data in actual["logical"].items():
                same(test, data["rng"], expected_logical[logical]["rng"])
                for a, b in zip(data["ema"]["ema_parameters"], expected_logical[logical]["ema"]["ema_parameters"], strict=True):
                    max_ema_difference = max(max_ema_difference, (a - b).abs().max().item())
        assert max_model_difference < 1e-4 and max_ema_difference < 1e-4
        print(json.dumps({"status": "ENTRYPOINT_COMPARE_PASS", "world": world,
                          "all_32_rng_streams_exact": True,
                          "all_training_inputs_exact": True,
                          "max_model_difference": max_model_difference,
                          "max_long_ema_difference": max_ema_difference}))


if __name__ == "__main__":
    main()
