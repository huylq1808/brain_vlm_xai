"""Write one exact-edit result and its review artifacts."""
import csv
import json
from brain_vlm.common import write


def _save_study(folder, request, result):
    write(folder / "request.json", request)
    write(folder / "result.json", result)
    write(folder / "assessment.json", result["review"])
    write(folder / "decisions.json", result["report_decisions"])
    (folder / "revised_findings.txt").write_text(result["revised_findings"] + "\n", encoding="utf-8")
    (folder / "unresolved.json").write_text(json.dumps(result["unresolved_items"], ensure_ascii=False, indent=2), encoding="utf-8")
    with (folder / "decisions.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        fields = ["id", "action", "edit_type", "input", "output", "reason", "evidence_ids"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(result["report_decisions"])
