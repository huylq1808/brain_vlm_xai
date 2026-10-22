"""Read a saved pretrained baseline without model imports or raw-run mutation.

This standard-library reader is used by the single baseline notebook.
An optional offline visibility diagnostic never changes the saved decisions.
"""
import base64
import difflib
import hashlib
import html
import json
from pathlib import Path


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _inside(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Artifact path outside run: " + relative)
    return path


def _details(title, value):
    return "<details><summary>" + html.escape(title) + "</summary><pre>" + html.escape(json.dumps(value, ensure_ascii=False, indent=2)) + "</pre></details>"


def _diff(before, after):
    left, right = [], []
    for tag, a, b, c, d in difflib.SequenceMatcher(a=before, b=after, autojunk=False).get_opcodes():
        x, y = html.escape(before[a:b]), html.escape(after[c:d])
        left.append(x if tag == "equal" else "<del>" + x + "</del>")
        right.append(y if tag == "equal" else "<ins>" + y + "</ins>")
    return "<div class='columns'><div><h3>Draft input</h3><pre>" + "".join(left) + "</pre></div><div><h3>Revised Findings</h3><pre>" + "".join(right) + "</pre></div></div>"


def read_pretrained_run(run, evaluation=None):
    root = Path(run).resolve()
    summary = _read(root / "summary.json")
    if summary["baseline"] != "pretrained_medgemma_signal_v1":
        raise ValueError("Expected a saved pretrained B0 run")
    evidence = _read(root / "evidence.json")
    by_id = {}
    for item in evidence:
        path = _inside(root, item["path"])
        if _sha(path) != item["sha256"] or item["evidence_id"] in by_id:
            raise ValueError("Changed/duplicate evidence: " + item["evidence_id"])
        by_id[item["evidence_id"]] = item
    calls, results, source_hashes = [], [], {}
    for path in sorted((root / "observations").glob("*.json")):
        record = _read(path)
        meta = record.get("generator_metadata") or {}
        citations = []
        observed = (record.get("observations") or {}).get("targets", {})
        for target, row in observed.items():
            ids = row.get("evidence_ids", [])
            if not set(ids) <= set(by_id):
                raise ValueError("Observation cites unknown image")
            citations.append({"target":target, "observation":row,
                              "views":[by_id[key] for key in ids]})
        calls.append({"job_id":record["job_id"], "execution":record.get("execution"),
            "schema_status":record.get("response_status"), "raw_response":record.get("raw_response"),
            "task":record["task"], "prompt":record["prompt"], "citations":citations,
            "eos_observed":meta.get("last_stop_token_id") is not None,
            "token_cap_reached":meta.get("output_hit_token_cap"),
            "input_tokens":meta.get("last_input_tokens"), "output_tokens":meta.get("last_output_tokens"),
            "pixel_shape":meta.get("last_pixel_values_shape"), "generation_seconds":meta.get("last_generation_seconds"),
            "resources":meta.get("last_resources"), "actual_model_inputs":meta.get("actual_model_inputs"),
            "metadata":meta})
        source_hashes[path.relative_to(root).as_posix()] = _sha(path)
    for entry in summary["outputs"]:
        path = _inside(root, entry["result_path"])
        result = _read(path); request = _read(path.with_name("request.json"))
        policy = _read(path.with_name("policy.json"))
        if result["original_findings"] != request["draft_text"]:
            raise ValueError("Changed original draft/result handoff")
        text = result["original_findings"]
        for edit in sorted(result["applied_edits"], key=lambda e:e["start"], reverse=True):
            if text[edit["start"]:edit["end"]] != edit["text"]:
                raise ValueError("Changed edit span")
            text = text[:edit["start"]] + edit["replacement"] + text[edit["end"]:]
        if text != result["revised_findings"]:
            raise ValueError("Undeclared rewrite in final Findings")
        results.append({"entry":entry, "request":request, "result":result, "policy":policy})
        for p in [path, path.with_name("request.json"), path.with_name("policy.json")]:
            source_hashes[p.relative_to(root).as_posix()] = _sha(p)
    evaluation = Path(evaluation).resolve() if evaluation else root.with_name(root.name + "_evaluation")
    metrics = _read(evaluation / "metrics.json") if (evaluation / "metrics.json").is_file() else None
    load = _read(root / "model_load.json") if (root / "model_load.json").is_file() else None
    for name in ["summary.json", "model_load.json", "evidence.json", "jobs.json", "prepared.json"]:
        if (root / name).is_file(): source_hashes[name] = _sha(root / name)
    return {"reader_version":"pretrained_reader_v1", "run_name":root.name, "summary":summary,
        "evidence":evidence, "calls":calls, "results":results, "model_load":load,
        "metrics":metrics, "source_sha256":source_hashes,
        "new_model_calls":0, "clinical_correctness":"not_evaluated",
        "interpretation":"Reads recorded pretrained execution; no model loading, clinical adjudication, label-driven edits or raw-result mutation."}


def render_pretrained_run(run, output, evaluation=None, visibility_diagnostic=None):
    root, out = Path(run).resolve(), Path(output).resolve()
    if out.is_relative_to(root) or root.is_relative_to(out):
        raise ValueError("Keep derived review outside raw run")
    report = read_pretrained_run(root, evaluation)
    if visibility_diagnostic:
        report["offline_visibility_diagnostic"] = _read(visibility_diagnostic)
    out.mkdir(parents=True, exist_ok=True)
    esc = lambda value: html.escape(str(value))

    def picture(item):
        uri = "data:image/png;base64," + base64.b64encode(_inside(root,item["path"]).read_bytes()).decode()
        title = f"{item['evidence_id']} · {item['sequence'].upper()} · z={item['native_canonical_z']}"
        return "<figure><img alt='" + esc(title) + "' src='" + uri + "'><figcaption>" + esc(title) + "</figcaption></figure>"

    style = "body{font:16px/1.6 system-ui;max-width:1250px;margin:auto;padding:24px;color:#15324a;background:#eef3f8}section{background:white;border-radius:12px;padding:22px;margin:18px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f6fa;padding:12px}.columns{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:14px}.gallery{display:flex;flex-wrap:wrap}figure{width:230px;margin:8px}img{width:100%}figcaption{font-size:13px}td,th{border:1px solid #ccd;padding:8px;text-align:left}table{border-collapse:collapse;width:100%}.notice{border-left:5px solid #b96808;padding:14px;background:#fff4dd}del{background:#fecaca}ins{background:#bbf7d0}details{margin:12px 0}"
    text = "<!doctype html><html lang='vi'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Pretrained baseline run reader</title><style>" + style + "</style>"
    text += "<h1>Pretrained Brain MRI baseline</h1><p>" + esc(report["run_name"]) + "</p>"
    text += "<p class='notice'><b>Ba mức cần phân biệt:</b> model đã chạy; output hợp schema; kết quả đúng về y khoa. Presenter xác minh luồng artifact và hiển thị judgment đã lưu. Khớp weak report label không xác nhận citation/finding/reference tissue đúng.</p>"
    flow = [("M0/M6", "4 NIfTI3D → min–max → sampled axial PNGs", "Đã thực hiện; sparse coverage"),
            ("M1/M5 PRIMA", "Frozen MRI representation / retrieval", "Không chạy trong B0; paired B1 đã có kết quả riêng"),
            ("M2", "Optional draft generation", "Supplied/cached draft; không generation mới trong B0"),
            ("M3", "Draft → bounded T2 claims / exact spans", "Không map toàn bộ report"),
            ("M7", "PNG + neutral target → pretrained MedGemma observations", str(len(report["calls"])) + " unique saved call(s)"),
            ("M8/M9", "Model-local gate → exact editor → Findings / decisions / unresolved", "Không independent clinical binding"),
            ("M4 / M10", "Omission discovery / actual-assessor attribution", "Chưa thực hiện trong B0")]
    text += "<section><h2>1 · Luồng và phạm vi</h2><table><tr><th>Module</th><th>Input → output</th><th>Trạng thái</th></tr>" + "".join("<tr>"+"".join("<td>"+esc(x)+"</td>" for x in row)+"</tr>" for row in flow) + "</table></section>"
    load = report["model_load"]
    if load:
        meta = load.get("model_metadata", {}); snapshot = load.get("snapshot", {})
        text += "<section><h2>2 · Model load thật và tài nguyên</h2>" + _details("Snapshot, parameters, RAM/VRAM interval", {"snapshot":snapshot, "parameters":meta.get("parameter_count"), "dtype":meta.get("dtype"), "quantization":meta.get("quantization"), "model_load_resources":load.get("resources")})
        text += "<p>RAM = host DRAM; VRAM = GPU memory. Peak từng interval không cộng lại. Call count không nhân theo số draft variants.</p></section>"
    text += "<section><h2>3 · Actual model-input views</h2><p>Các PNG này thực sự được đưa vào vision encoder; không phải ảnh review sau một numeric prediction.</p><div class='gallery'>" + "".join(picture(e) for e in report["evidence"]) + "</div>" + _details("Sequence, native z, normalization, hashes",report["evidence"]) + "</section>"
    for call in report["calls"]:
        state = {k:call[k] for k in ["execution", "schema_status", "eos_observed", "token_cap_reached", "input_tokens", "output_tokens", "pixel_shape", "generation_seconds", "resources"]}
        text += "<section><h2>4 · M7 observation — " + esc(call["job_id"]) + "</h2>" + _details("Execution/schema/resources", state)
        for target in call["citations"]:
            text += "<h3>Model judgment + ảnh được trích dẫn</h3><pre>" + esc(json.dumps(target["observation"],ensure_ascii=False,indent=2)) + "</pre><div class='gallery'>" + "".join(picture(e) for e in target["views"]) + "</div>"
        text += "<p class='notice'>Reason là lời của model, gồm cả anatomy/diagnosis suy đoán nếu có; chúng không trở thành verified facts hoặc được chèn vào Findings. ID hợp lệ và đúng sequence chưa chứng minh target hiện diện hoặc reference tissue đúng.</p>"
        text += _details("Neutral task và reference yêu cầu",call["task"]) + _details("Exact prompt",call["prompt"]) + _details("Raw response",call["raw_response"]) + _details("Actual tensor inputs",call["actual_model_inputs"]) + "</section>"
    for row in report["results"]:
        result = row["result"]; entry = row["entry"]
        text += "<section><h2>5 · " + esc(entry["study_id"] + " / " + entry["variant"]) + "</h2>" + _diff(result["original_findings"],result["revised_findings"])
        text += _details("Exact approved edits",result["applied_edits"]) + _details("Decisions",result["report_decisions"]) + _details("Policy/model-local identity",row["policy"]) + _details("Unresolved / unassessed",{"unresolved":result["unresolved_items"], "unassessed":result.get("unassessed_spans",[])}) + "</section>"
    metrics = report["metrics"]
    text += "<section><h2>6 · Weak-reference outcomes</h2><p>Không gọi precision/recall của một controlled example là clinical accuracy. Source reference lấy từ authored report; patient, draft và model-call denominators khác nhau.</p>"
    if metrics:
        text += _details("Tổng saved metrics",metrics["totals"]) + _details("Tách cohort/track",metrics["by_cohort_track"])
    else:
        text += "<p>Không tìm thấy saved evaluation. Không tự tạo nhãn hoặc correctness score.</p>"
    text += "</section>"
    if "offline_visibility_diagnostic" in report:
        text += "<section><h2>7 · Offline visibility diagnostic — không vào inference</h2><p>Source annotation kiểm target visibility trong scope của nó, không xác nhận clinical identity, signal hoặc reference tissue. Diagnostic này không đổi citation, assessment, edit hoặc saved metrics.</p>" + _details("Source-mask plane counts and provenance",report["offline_visibility_diagnostic"]) + "</section>"
    text += _details("Reader provenance / no model calls",{"new_model_calls":0,"source_sha256":report["source_sha256"],"clinical_correctness":"not_evaluated"}) + "</html>"
    (out / "presentation.html").write_text(text, encoding="utf-8")
    (out / "report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return out / "presentation.html"


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path)
    parser.add_argument("--visibility-diagnostic", type=Path)
    args = parser.parse_args()
    print(render_pretrained_run(args.run,args.output,args.evaluation,args.visibility_diagnostic))


if __name__ == "__main__": main()
