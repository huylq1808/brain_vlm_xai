"""Run the bounded baseline, read a saved run, or extract PRIMA encoders."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for action in ("prepare", "infer", "finalize"):
        command = commands.add_parser(action)
        command.add_argument("config", type=Path)
    viewer = commands.add_parser("present", help="Read a saved pretrained run; no model load")
    viewer.add_argument("run", type=Path)
    viewer.add_argument("--output", type=Path, required=True, help="New review folder outside the raw run")
    evaluator = commands.add_parser("evaluate", help="Offline reference-relative T2 metrics")
    evaluator.add_argument("run", type=Path)
    evaluator.add_argument("--output", type=Path, required=True)
    evaluator.add_argument("--labels", type=Path)
    extractor = commands.add_parser("extract-prima", help="Export existing encoders; no distillation training")
    extractor.add_argument("--model-root", type=Path, required=True)
    extractor.add_argument("--compact-root", type=Path)
    extractor.add_argument("--prima-root", type=Path, default=Path("vendor/Prima"))
    extractor.add_argument("--no-download", action="store_true")
    extractor.add_argument("--record", type=Path)
    args = parser.parse_args()
    if args.command in {"prepare", "infer", "finalize"}:
        from brain_vlm.baseline import pretrained
        getattr(pretrained, args.command)(args.config)
    elif args.command == "present":
        from brain_vlm.viewer import render_pretrained_run
        print(render_pretrained_run(args.run, args.output))
    elif args.command == "evaluate":
        from brain_vlm.baseline.evaluation import evaluate
        print(evaluate(args.run, args.output, args.labels))
    else:
        from brain_vlm.baseline.vlm.models import ensure_prima
        from brain_vlm.common import write
        compact = args.compact_root or args.model_root / "prima_encoders_v1"
        if args.record and args.record.resolve().is_relative_to(compact.resolve()):
            parser.error("Registry record must be outside the compact weights")
        record = ensure_prima(args.model_root, compact, args.prima_root, not args.no_download)
        if args.record:
            write(args.record, record)
        print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
