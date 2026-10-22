"""Software-only integration probes; fake model answers never count as efficacy."""
import json
from pathlib import Path
import tempfile
import unittest

from brain_vlm.common import read, write
from brain_vlm.baseline import pretrained

PROJECT = Path(__file__).resolve().parents[2]
SAMPLE = PROJECT / "outputs/samples/baseline_demo_v1/sample_001"


class FakeGenerator:
    def __init__(self, signal="high", stopped=True, citation="image_07"):
        self.calls = 0
        self.signal, self.citation = signal, citation
        self.metadata = {"software_test_double":True, "last_stop_token_id":106 if stopped else None,
                         "output_hit_token_cap":not stopped}

    def configure_response(self, **options): pass
    def set_response_schema(self, schema): self.schema = schema
    def __call__(self, prompt, images):
        self.calls += 1
        self.images = images
        return json.dumps({"targets":{"target_01":{"entity_status":"identified", "assessability":"sufficient",
            "observed_signal":self.signal, "evidence_ids":[self.citation],
            "reason":"Image " + self.citation + " shows target contrast relative to visible gray matter."}}})


@unittest.skipUnless(SAMPLE.is_dir(), "Private sample ZIP required for MRI integration probes")
class PretrainedSoftware(unittest.TestCase):
    def setUp(self):
        base = PROJECT / "outputs/validation/pretrained10"
        base.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="software_", dir=base)
        self.root = Path(self.temp.name).resolve()
        assert self.root.is_relative_to(base.resolve())
        self.config = self.root / "config.json"
        self.out = self.root / "run"
        write(self.config, {"version":pretrained.VERSION, "sample_root":str(SAMPLE), "output_root":str(self.out),
            "variants":["source", "controlled_low", "controlled_intermediate"], "slices_per_sequence":3,
            "models":{}, "device":"cuda", "max_images":16, "max_new_tokens":768, "download_missing":False})

    def tearDown(self): self.temp.cleanup()

    def infer(self, generator):
        def loader(cfg, tokens):
            return generator, {"model_loaded":False, "software_test_double":True}, {"model_id":"software_test_double_not_MedGemma"}
        pretrained.infer(self.config, loader=loader)

    def test_real_views_joint_call_and_exact_edits(self):
        p = pretrained.prepare(self.config)
        self.assertEqual((p["image_count"], p["observation_jobs"]), (12, 1))
        evidence = read(self.out / "evidence.json")
        self.assertEqual({e["sequence"] for e in evidence}, {"t1", "t1c", "t2", "flair"})
        job = next(iter(read(self.out / "jobs.json").values()))
        self.assertNotIn("draft_text", job["task"])
        self.assertNotIn("asserted_direction", job["prompt"])
        self.assertFalse(any(e["mask_overlay"] for e in evidence))
        generator = FakeGenerator()
        self.infer(generator)
        self.assertEqual(generator.calls, 1)
        self.assertEqual(len(generator.images), 12)
        pretrained.finalize(self.config)
        summary = read(self.out / "summary.json")
        self.assertEqual([r["edits"] for r in summary["outputs"]], [0, 1, 1])
        self.assertEqual(summary["unique_model_calls"], 1)
        self.assertTrue((self.out / "presentation.html").exists())
        # Reconnect must reuse the exact saved call and not silently retry.
        pretrained.infer(self.config, loader=lambda *a: self.fail("Loaded model for cached call"))
        self.assertEqual(generator.calls, 1)
        from brain_vlm.baseline.evaluation import evaluate
        evaluate(self.out, self.root / "evaluation")
        self.assertIsNone(read(self.root / "evaluation/metrics.json")["totals"]["correct_repair"])

    def test_incomplete_response_preserves_text(self):
        self.infer(FakeGenerator(stopped=False))
        pretrained.finalize(self.config)
        self.assertTrue(all(r["edits"] == 0 for r in read(self.out / "summary.json")["outputs"]))

    def test_wrong_sequence_citation_defers(self):
        self.infer(FakeGenerator(citation="image_01"))
        pretrained.finalize(self.config)
        summary = read(self.out / "summary.json")
        self.assertEqual(summary["schema_valid_calls"], 0)
        self.assertTrue(all(not r["accepted"] and r["edits"] == 0 for r in summary["outputs"]))

    def test_no_model_result_no_synthetic_fallback(self):
        pretrained.prepare(self.config)
        with self.assertRaisesRegex(ValueError, "Run infer before finalize"):
            pretrained.finalize(self.config)

    def test_interrupted_call_not_automatically_retried(self):
        pretrained.prepare(self.config)
        key = next(iter(read(self.out / "jobs.json")))
        write(self.out / "attempts" / (key + ".json"), {"automatic_retries":0})
        with self.assertRaisesRegex(ValueError, "Interrupted call"):
            self.infer(FakeGenerator())


if __name__ == "__main__": unittest.main()
