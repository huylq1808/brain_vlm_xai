"""Worker visibility and cancellation only; no trained model or accuracy claim."""
import contextlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from brain_vlm import runtime


class LiveLogTests(unittest.TestCase):
    def setUp(self):
        parent = Path(__file__).resolve().parents[2] / "outputs/.tmp"
        parent.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=parent);self.root = Path(self.temp.name)
    def tearDown(self):
        assert self.root.resolve().is_relative_to(Path(__file__).resolve().parents[2] / "outputs/.tmp")
        self.temp.cleanup()

    def test_output_and_heartbeat_are_visible_before_worker_finishes(self):
        gate, log = self.root / "continue", self.root / "live.log"
        script = "import sys,time;from pathlib import Path;print('FIRST');print('sample_test_token_123');p=Path(sys.argv[1]);\nwhile not p.exists():time.sleep(.02)\nprint('LAST')"
        result, errors = [], []
        def run():
            try:result.append(runtime._run_logged([sys.executable, "-c", script, str(gate)], log, "cache", stream=True, heartbeat_seconds=.1))
            except BaseException as error:errors.append(error)
        with patch.dict(runtime.os.environ, {"HF_TOKEN": "sample_test_token_123"}), contextlib.redirect_stdout(io.StringIO()):
            worker = threading.Thread(target=run);worker.start()
            try:
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline:
                    text = log.read_text(encoding="utf-8") if log.exists() else ""
                    if "FIRST" in text and "has not produced a new line" in text:break
                    time.sleep(.02)
                self.assertIn("FIRST", text);self.assertIn("has not produced a new line", text)
                self.assertTrue(worker.is_alive());self.assertFalse(result)
                self.assertNotIn("sample_test_token_123", text)
            finally:
                gate.write_text("continue");worker.join(timeout=10)
        self.assertFalse(worker.is_alive());self.assertFalse(errors)
        self.assertIn("LAST", result[0]);self.assertIn("exit 0", log.read_text())

    def test_nonzero_exit_retains_child_error_in_live_log(self):
        log = self.root / "failure.log"
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "exit 7"):
                runtime._run_logged([sys.executable, "-c", "print('specific failure');raise SystemExit(7)"], log, "cache", stream=True)
        self.assertIn("specific failure", log.read_text())

    def test_interrupt_stops_the_direct_worker(self):
        original = subprocess.Popen;children = []
        def create(*args, **kwargs):
            process = original(*args, **kwargs);children.append(process);return process
        events = Mock();events.get.side_effect = KeyboardInterrupt
        with patch("subprocess.Popen", side_effect=create), patch("queue.Queue", return_value=events), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(KeyboardInterrupt):
                runtime._run_logged([sys.executable, "-c", "import time;time.sleep(30)"], self.root / "stop.log", "cache", stream=True)
        self.assertEqual(len(children), 1);self.assertIsNotNone(children[0].poll())


    def test_vq_progress_hooks_leave_features_and_weights_unchanged(self):
        import torch
        from brain_vlm.prima.progress import attach_vq_progress
        model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2)).eval()
        x = torch.arange(12).reshape(3, 4).float()
        expected = model(x).detach().clone();weights = {k: v.detach().clone() for k, v in model.state_dict().items()}
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            attach_vq_progress(model);attach_vq_progress(model)
            actual = model(x)
        self.assertTrue(torch.equal(expected, actual))
        self.assertTrue(all(torch.equal(weights[k], v) for k, v in model.state_dict().items()))
        self.assertEqual(len(model._brain_vlm_progress_handles), 2)
        self.assertIn("first input", output.getvalue());self.assertIn("1 batches / 3 patches", output.getvalue())



if __name__ == "__main__":unittest.main()
