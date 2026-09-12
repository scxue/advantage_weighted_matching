"""No cloud writes: logger unit tests plus actual W&B offline history/media checks."""
import ast
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from flow_grpo.wandb_logging import TrainingStepLogger


class FakeRun:
    def __init__(self, next_step=0):
        self.step = next_step
        self.calls = []
        self.rows = []
        self.pending = {}

    def log(self, data, *, step, commit):
        if step < self.step:
            raise AssertionError("Logger attempted a backwards W&B write")
        if self.pending and step != self.step:
            raise AssertionError("Previous step was not explicitly committed")
        self.step = step
        self.calls.append((dict(data), step, commit))
        self.pending.update(data)
        if commit:
            self.rows.append({**self.pending, "_step": step})
            self.pending = {}
            self.step += 1


class LoggingTests(unittest.TestCase):
    def test_one_history_row_including_eval_and_media(self):
        run = FakeRun()
        logger = TrainingStepLogger(run.log, next_step=run.step)
        logger.flush()  # No phantom row for skipped epochs 0/1.
        for key in ["eval_reward", "eval_images", "reward_avg", "images", "grad_norm", "loss", "epoch_time"]:
            logger.log({key: key}, step=0)
        self.assertEqual(run.rows, [])
        self.assertEqual(run.step, 0)
        logger.flush()
        logger.flush()
        self.assertEqual(len(run.rows), 1)
        self.assertEqual(run.rows[0]["_step"], 0)
        self.assertEqual(run.rows[0]["train/global_step"], 0)
        self.assertEqual(run.rows[0]["images"], "images")
        self.assertEqual(run.rows[0]["epoch_time"], "epoch_time")
        self.assertFalse(logger.log({"duplicate": True}, step=0))

    def test_resume_replays_training_without_reupload(self):
        run = FakeRun()
        first = TrainingStepLogger(run.log, next_step=run.step)
        for step in range(100, 108):
            first.log({"loss": "original"}, step=step)
            first.flush()
        self.assertEqual(run.step, 108)
        original_rows = list(run.rows)
        resumed = TrainingStepLogger(run.log, next_step=run.step)
        calls_before = len(run.calls)
        updates = []
        for step in range(100, 111):
            updates.append(step)  # Training is never conditional on logger's return.
            accepted = resumed.log({"loss": "replayed", "images": object()}, step=step)
            self.assertEqual(accepted, step >= 108)
            resumed.flush()
            if step < 108:
                self.assertEqual(len(run.calls), calls_before)
        self.assertEqual(updates, list(range(100, 111)))
        self.assertEqual(run.rows[:8], original_rows)
        self.assertEqual([r["_step"] for r in run.rows], list(range(100, 111)))

    def test_cloud_behind_checkpoint_can_skip_forward(self):
        run = FakeRun(next_step=95)
        logger = TrainingStepLogger(run.log, next_step=run.step)
        logger.log({"loss": 1}, step=100)
        logger.flush()
        self.assertEqual(run.rows[0]["_step"], 100)
        self.assertEqual(run.step, 101)

    def test_multiple_updates_commit_previous_only_once(self):
        run = FakeRun()
        logger = TrainingStepLogger(run.log)
        logger.log({"reward": .5}, step=0)
        logger.log({"loss": 2}, step=0)
        logger.log({"loss": 1}, step=1)
        logger.log({"epoch_time": 10}, step=1)
        logger.flush()
        self.assertEqual([r["_step"] for r in run.rows], [0, 1])
        self.assertNotIn("epoch_time", run.rows[0])
        self.assertEqual(run.rows[1]["epoch_time"], 10)

    def test_step_validation(self):
        for step in [-1, None, 1.5, True]:
            with self.assertRaises(ValueError):
                TrainingStepLogger(lambda **kw: None, next_step=step)
        run = FakeRun()
        logger = TrainingStepLogger(run.log)
        logger.log({"loss": 1}, step=5)
        with self.assertRaises(ValueError):
            logger.log({"loss": 2}, step=4)

    def test_trainer_routes_eval_and_preserves_local_replay_logs(self):
        path = Path(__file__).resolve().parents[1] / "scripts/train_sd3_awm.py"
        tree = ast.parse(path.read_text())
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        callback = next(n for n in main.body if isinstance(n, ast.FunctionDef) and n.name == "log_metrics")
        run = FakeRun(next_step=108)
        logger = TrainingStepLogger(run.log, next_step=run.step)
        local = []
        acc = SimpleNamespace(is_main_process=True, log=lambda data, step: local.append((data, step)))
        scope = {"accelerator": acc, "wandb_logger": logger}
        exec(compile(ast.Module(body=[callback], type_ignores=[]), str(path), "exec"), scope)
        scope["log_metrics"]({"loss": 2}, step=100)
        self.assertEqual(local, [({"loss": 2}, 100)])
        self.assertEqual(run.calls, [])
        acc.is_main_process = False
        scope["wandb_logger"] = None
        scope["log_metrics"]({"loss": 1}, step=108)
        self.assertEqual(len(local), 2)
        calls = [n for n in ast.walk(main) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
        eval_calls = [n for n in calls if n.func.id == "eval"]
        self.assertEqual(len(eval_calls), 2)
        self.assertTrue(all(ast.unparse(n.args[-1]) == "log_metrics" for n in eval_calls))
        timing = next(n for n in calls if n.func.id == "log_metrics" and n.args
                      and isinstance(n.args[0], ast.Dict)
                      and any(isinstance(k, ast.Constant) and k.value == "epoch_time" for k in n.args[0].keys))
        self.assertEqual(ast.unparse(next(k.value for k in timing.keywords if k.arg == "step")), "global_step - 1")

    def test_real_offline_sdk_history_and_temporary_media(self):
        import wandb
        from PIL import Image
        from wandb.sdk.internal.datastore import DataStore
        from wandb.proto import wandb_internal_pb2

        with tempfile.TemporaryDirectory(prefix="awm-wandb-test-") as directory:
            run = wandb.init(project="awm-logging-offline-test", mode="offline", dir=directory,
                             settings=wandb.Settings(console="off", quiet=True, disable_code=True,
                                                     disable_git=True, x_disable_stats=True))
            run_dir = Path(run.dir)
            try:
                logger = TrainingStepLogger(run.log, next_step=run.step)
                run.define_metric("train/global_step")
                run.define_metric("*", step_metric="train/global_step")
                logger.log({"reward_avg": .5}, step=0)
                with tempfile.TemporaryDirectory(prefix="awm-media-test-") as media:
                    path = Path(media) / "image.jpg"
                    Image.new("RGB", (8, 8), "red").save(path)
                    logger.log({"eval_images_non_ema": [wandb.Image(str(path))]}, step=0)
                self.assertFalse(path.exists())
                logger.log({"loss": 1.25, "epoch_time": 3.0}, step=0)
                logger.flush()
                self.assertEqual(run.step, 1)
                # Reconstruct the logger against the SDK's actual next writable
                # step. Offline mode itself cannot exercise cloud reconnect.
                resumed = TrainingStepLogger(run.log, next_step=run.step)
                self.assertFalse(resumed.log({"loss": -999, "images": object()}, step=0))
                resumed.log({"loss": .75}, step=1)
                resumed.flush()
                self.assertEqual(run.step, 2)
            finally:
                run.finish()
                # Close the offline service before unittest exits with a bool
                # status, which this SDK's protobuf exit_code does not accept.
                wandb.teardown(exit_code=0)

            history = next(Path(directory).glob("wandb/offline-run-*/*.wandb"))
            rows = []
            ds = DataStore()
            ds._fp = history.open("rb")
            ds._index = 0
            ds._size_bytes = history.stat().st_size
            ds._opened_for_scan = True
            try:
                ds._read_header()
                while True:
                    data = ds.scan_data()
                    if data is None:
                        break
                    record = wandb_internal_pb2.Record()
                    record.ParseFromString(data)
                    if record.WhichOneof("record_type") == "history":
                        row = {}
                        for item in record.history.item:
                            keys = [item.key] if item.key else list(item.nested_key)
                            target = row
                            for key in keys[:-1]:
                                target = target.setdefault(key, {})
                            target[keys[-1]] = json.loads(item.value_json)
                        rows.append(row)
            finally:
                ds._fp.close()
            self.assertEqual([r["_step"] for r in rows], [0, 1])
            self.assertEqual([r["train/global_step"] for r in rows], [0, 1])
            self.assertEqual(rows[0]["loss"], 1.25)
            self.assertEqual(rows[0]["reward_avg"], .5)
            self.assertEqual(rows[0]["epoch_time"], 3.0)
            self.assertEqual(rows[0]["eval_images_non_ema"]["count"], 1)
            self.assertEqual(len(list((run_dir / "media/images").glob("*"))), 1)
            self.assertEqual(rows[1]["loss"], .75)


if __name__ == "__main__":
    unittest.main()
