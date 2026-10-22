"""Offline reader checks on the received real model run; no model execution."""
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

PROJECT = Path(__file__).resolve().parents[2]
RUN = PROJECT / "outputs/runs/pretrained10_20261022T023259_599965Z"
TEMP_ROOT = PROJECT / "outputs/validation/pretrained10_received_20261022"
from brain_vlm import viewer as reader


@unittest.skipUnless((RUN / "summary.json").is_file(), "Received actual pretrained run required")
class PretrainedReaderTests(unittest.TestCase):
    def temporary(self):
        TEMP_ROOT.mkdir(parents=True, exist_ok=True)
        context = tempfile.TemporaryDirectory(prefix="reader_test_", dir=TEMP_ROOT)
        assert Path(context.name).resolve().is_relative_to(TEMP_ROOT.resolve())
        return context

    def test_patient_drafts_calls_and_cited_plane_stay_distinct(self):
        report = reader.read_pretrained_run(RUN)
        self.assertEqual(len(report["results"]), 2)
        self.assertEqual(len(report["calls"]), 1)
        self.assertEqual(len(report["evidence"]), 12)
        self.assertEqual(report["calls"][0]["citations"][0]["views"][0]["native_canonical_z"], 116)
        self.assertEqual(report["metrics"]["totals"]["correct_repair"], 1)
        self.assertEqual(report["clinical_correctness"], "not_evaluated")
        self.assertEqual(report["new_model_calls"], 0)

    def test_reader_keeps_raw_files_unchanged_and_rejects_inside_output(self):
        with self.temporary() as temp:
            before = {p.relative_to(RUN).as_posix():reader._sha(p) for p in RUN.rglob("*") if p.is_file()}
            page = reader.render_pretrained_run(RUN, Path(temp) / "review")
            self.assertIn("data:image/png;base64,", page.read_text(encoding="utf-8"))
            after = {p.relative_to(RUN).as_posix():reader._sha(p) for p in RUN.rglob("*") if p.is_file()}
            self.assertEqual(before, after)
            with self.assertRaisesRegex(ValueError, "outside raw"):
                reader.render_pretrained_run(RUN, RUN / "invalid_review")

    def test_absent_evaluation_does_not_invent_accuracy(self):
        with self.temporary() as temp:
            report = reader.read_pretrained_run(RUN, Path(temp))
            self.assertIsNone(report["metrics"])

    def test_changed_image_is_not_silently_displayed_as_original(self):
        with self.temporary() as temp:
            clone = Path(temp) / "copy"
            shutil.copytree(RUN, clone)
            evidence = json.loads((clone / "evidence.json").read_text(encoding="utf-8"))
            image = clone / evidence[0]["path"]
            image.write_bytes(image.read_bytes() + b"changed")
            with self.assertRaisesRegex(ValueError, "Changed/duplicate evidence"):
                reader.read_pretrained_run(clone)


if __name__ == "__main__": unittest.main()
