"""Pretrained MedGemma baseline: real image observation, bounded exact T2 edits.

No native component/signal checkpoint, masks, report labels or PRIMA embeddings.
The frozen adapter, observation contracts and editor are shared with the VLM
route. Prepare/infer/finalize are separate so reconnect does not reload weights
just to inspect results. One image observation is shared by assertion variants.
"""
from pathlib import Path
import argparse

from brain_vlm.common import read, write, sha, require, fingerprint, inventory, verify_inventory, package_root

VERSION = "pretrained_medgemma_signal_v1"
CITATION_POLICY_VERSION = "citation_id_sets_v1"


def context(config_path):
    path = Path(config_path).resolve(); cfg = read(path)
    required = {"version", "sample_root", "output_root", "variants", "slices_per_sequence",
                "models", "device", "max_images", "max_new_tokens", "download_missing"}
    require(required <= set(cfg) <= required | {"citation_policy"}, "Use the baseline notebook config")
    require(cfg.get("citation_policy", "strict") in {"strict", "stable_unique"}, "Unknown citation policy")
    require(cfg["version"] == VERSION and cfg["device"] == "cuda", "Use pretrained GPU baseline")
    require(type(cfg["slices_per_sequence"]) is int and 1 <= cfg["slices_per_sequence"] <= 4, "Use 1–4 slices per sequence")
    require(cfg["max_images"] == 16 and cfg["max_new_tokens"] == 768, "Use the fixed initial profile")
    sample_root, out = Path(cfg["sample_root"]).resolve(), Path(cfg["output_root"]).resolve()
    require(not out.is_relative_to(sample_root) and not sample_root.is_relative_to(out), "Separate inputs/results")
    sample = read(sample_root / "sample.json")
    require(sample["version"] == "single_men_t2_sample_v1", "Use the existing sample ZIPs")
    variants = cfg["variants"] or list(sample["drafts"])
    require(variants and len(set(variants)) == len(variants) and set(variants) <= set(sample["drafts"]), "Unknown/duplicate draft variant")
    require(set(sample["images"]) == {"t1", "t1c", "t2", "flair"}, "Four aligned MRI sequences required")
    used = list(sample["images"].values()) + [sample["drafts"][v]["path"] for v in variants]
    for name in used:
        target = (sample_root / name).resolve()
        require(target.is_relative_to(sample_root) and sha(target) == sample["files"][name], "Changed MRI/draft input")
    source = package_root()
    sources = {p.relative_to(source).as_posix(): sha(p) for p in sorted(source.rglob("*.py"))}
    identity = {"config":cfg, "sample_sha256":sha(sample_root / "sample.json"), "source_sha256":sources}
    if (out / "paired_plan.json").exists():
        identity["paired_plan_sha256"] = sha(out / "paired_plan.json")
    if (out / "identity.json").exists():
        require(read(out / "identity.json") == identity, "Changed source/config/sample: choose a new run")
    else:
        require(not out.exists() or {p.name for p in out.iterdir()} <= {"config.json", "logs", "paired_plan.json"}, "Preserve unrelated output")
        write(out / "identity.json", identity)
        import zipfile
        with zipfile.ZipFile(out / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as z:
            for name in sources: z.write(source / name, "src/brain_vlm/" + name)
    write(out / "config.json", cfg)
    return cfg, sample_root, out, sample, variants


def prepare(config_path):
    import nibabel as nib
    import numpy as np
    from brain_vlm.refinement.images import render_native_slices
    from brain_vlm.baseline.claims import request, NATURAL_VERSION
    from brain_vlm.refinement.claim_extraction import finding_anchor
    from brain_vlm.refinement.observations import observation_prompt, sequence_schema
    cfg, data, out, sample, variants = context(config_path)
    if (out / "prepared.json").exists():
        prepared = read(out / "prepared.json"); verify_inventory(out, prepared["artifacts"])
        return prepared
    require(not (out / "images").exists(), "Preserve partial rendering; use a new run")
    plan = read(out / "paired_plan.json") if (out / "paired_plan.json").exists() else None
    evidence, geometry = [], None
    for sequence in ["t1", "t1c", "t2", "flair"]:
        path = data / sample["images"][sequence]
        image = nib.as_closest_canonical(nib.load(path)); values = np.asarray(image.dataobj, dtype=np.float32)
        require(len(image.shape) == 3 and np.isfinite(values).all(), "Finite 3D MRI required")
        if geometry is None: geometry = (image.shape, image.affine)
        require(image.shape == geometry[0] and np.allclose(image.affine, geometry[1], atol=1e-4, rtol=0), "Register sequence grids explicitly before inference")
        indices = sorted(set(int(round(z)) for z in np.linspace(0, image.shape[2]-1, cfg["slices_per_sequence"]+2)[1:-1]))
        if plan is not None:
            indices = plan["planes"][sequence]
            require(1 <= len(indices) <= 4 and indices == sorted(set(indices)) and
                    all(type(z) is int and 0 <= z < image.shape[2] for z in indices), "Invalid paired plane plan")
        bounds = [float(values.min()), float(values.max())]
        rows = render_native_slices({sequence:str(path)}, out / "images" / sequence / "images", indices,
                                   max_slices=4, intensity_bounds=bounds, include_labels=False)
        for row in rows:
            row.update(evidence_id=f"image_{len(evidence)+1:02d}", path="images/"+sequence+"/"+row["path"],
                       normalization={"method":"per-volume min-max, RGB channels equal", "low":bounds[0], "high":bounds[1]},
                       model_input=True, role="study_context" if plan else "flat_uniform_context", native_canonical_z=row["canonical_ras_z"])
            row["header_world_z"], row["header_world_units"] = row.pop("world_z_mm"), image.header.get_xyzt_units()[0]
            evidence.append(row)
    require(len(evidence) <= cfg["max_images"], "Image budget exceeded")
    write(out / "evidence.json", evidence)
    jobs, requests = {}, {}
    for variant in variants:
        draft = (data / sample["drafts"][variant]["path"]).read_text(encoding="utf-8")
        req = request(sample["study_id"], draft, evidence, available_sequences=list(sample["images"]), claim_protocol=NATURAL_VERSION)
        req["preparation"].update(observer_scope="Frozen MedGemma on sampled multi-sequence images; no numeric observer",
            reference_scope="Model must identify normal-appearing gray matter in supplied T2 views, otherwise defer")
        anchor = finding_anchor(draft, "primary_finding_hypothesis")
        allowed = req["preparation"]["mapped"] and req["preparation"].get("observer_scope_alignment", {}).get("status") != "unresolved"
        job_id = None
        if allowed:
            target = {"description":"Inspect the candidate primary intracranial mass-like finding, if identifiable. The search cue does not prove a finding exists. If multiple findings compete or identity is unclear, defer.",
                      "signal_reference":"normal-appearing cortical gray matter visible in the same T2 images; if unavailable, use insufficient/unresolved",
                      "sequence":"t2", "finding_ref":"finding_01", "location_hypothesis":anchor["location_hypothesis"],
                      "finding_mapping":anchor["mapping_status"]}
            task = {"protocol":"image_first_signal_v1", "targets":{"target_01":target},
                    "evidence":[{k:row[k] for k in ["evidence_id", "sequence", "scope", "role", "native_canonical_z"]} for row in evidence]}
            prompt = observation_prompt(task) + "\nAll views belong to ONE aligned MRI study. Use other sequences as context, but report signal only on T2 with T2 citations. Sparse slices are not full-volume coverage. Do not infer study-wide absence."
            job_id = "joint_" + fingerprint(task)[:16]
            jobs[job_id] = {"job_id":job_id, "case":sample["study_id"], "group":"pretrained_joint", "sequence":"t2",
                "task":task, "sources":[{"sha256":e["sha256"], "sequence":e["sequence"]} for e in evidence],
                "images":[{**e, "resolved_path":str(out / e["path"])} for e in evidence],
                "prompt":prompt, "response_schema":sequence_schema(task),
                "output_policy":{"constrained_decoding":True, "system_instruction":"Return the requested final JSON only. No thinking block, prose or Markdown."}}
            if plan and plan.get("image_order") == "reverse":
                jobs[job_id]["images"].reverse()
                jobs[job_id]["sources"].reverse()
        write(out / "requests" / (variant + ".json"), req)
        requests[variant] = {"path":"requests/"+variant+".json", "job_id":job_id, "origin":sample["drafts"][variant]["origin"]}
    write(out / "jobs.json", jobs)
    files = [out / "evidence.json", out / "jobs.json", *list((out / "images").rglob("*.png")), *list((out / "requests").glob("*.json"))]
    all_files = inventory(out)
    prepared = {"version":VERSION, "study_id":sample["study_id"], "requests":requests, "observation_jobs":len(jobs),
        "image_count":len(evidence), "source_masks_used":False, "evaluation_labels_used":False,
        "prerequisite_small_checkpoint":False, "artifacts":{p.relative_to(out).as_posix():all_files[p.relative_to(out).as_posix()] for p in files}}
    write(out / "prepared.json", prepared)
    print("Prepared", len(evidence), "actual model-input PNGs;", len(jobs), "joint observation job(s). No model loaded.", flush=True)
    return prepared


def infer(config_path, loader=None):
    cfg, _, out, _, _ = context(config_path); prepare(config_path)
    from brain_vlm.refinement.observations import run_observation, job_fingerprint, job_config
    jobs = read(out / "jobs.json"); pending = []
    for key, job in jobs.items():
        if (out / "observations" / (key + ".json")).exists(): continue
        require(not (out / "attempts" / (key + ".json")).exists(), "Interrupted call preserved; choose a new run, no automatic retry")
        pending.append(job)
    if pending:
        if loader is None:
            import torch
            require(torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory/2**30 >= 12, "Use GPU >=12GiB VRAM; no CPU fallback")
            from brain_vlm.baseline.vlm.loader import load_medgemma
            loader = load_medgemma
        print("Loading pretrained MedGemma weights (no native small heads)...", flush=True)
        generator, load, model_config = loader(cfg, cfg["max_new_tokens"])
        write(out / "model_load.json", load); write(out / "model_config.json", model_config)
        for job in pending:
            write(out / "attempts" / (job["job_id"] + ".json"), {"automatic_retries":0,
                "fingerprint":job_fingerprint(job, job_config(job, model_config))})
            print("M7 actual pretrained forward/generate:", job["job_id"], len(job["images"]), "images", flush=True)
            record, _ = run_observation(job, generator, model_config, out)
            print("Raw call:", record["execution"], "observation contract:", record["response_status"], flush=True)
    if jobs:
        model_config = read(out / "model_config.json")
        for key, job in jobs.items():
            record = read(out / "observations" / (key + ".json"))
            require(record["fingerprint"] == job_fingerprint(job, job_config(job, model_config)), "Changed cached call")
    print("New model calls:", len(pending), "; saved records retained, including failures.", flush=True)


def compare(request, payload):
    from brain_vlm.refinement.contracts import review_template
    review = review_template(request, "model", "google/medgemma-1.5-4b-it")
    observed = payload["targets"]["target_01"] if payload else None
    semantics = request["preparation"].get("signal_semantics")
    for claim, row in zip(request["claims"], review["claims"]):
        if claim["claim_id"] != "t2_direction" or not observed or not semantics: continue
        row["entity_status"] = observed["entity_status"]
        for attr in row["attributes"]:
            if attr["slot_id"] != "t2_signal": continue
            attr["reason"] = observed["reason"]
            value = observed["observed_signal"]
            if observed["assessability"] != "sufficient" or value not in {"low", "intermediate", "high"}: continue
            verdict = "supported" if value == semantics["asserted_direction"] else "contradicted"
            replacement = semantics["renderings"].get(value)
            attr.update(assessability="sufficient", verdict=verdict, evidence_ids=observed["evidence_ids"])
            if verdict == "contradicted" and replacement:
                attr.update(replacement=replacement, replacement_supported=True)
    return review


def finalize(config_path):
    cfg, _, out, sample, _ = context(config_path); prepared = prepare(config_path)
    from brain_vlm.refinement.gates import gate_record
    from brain_vlm.refinement.editing import apply_assessment, describe_report
    from brain_vlm.baseline.product import _save_study
    outputs, records = [], {}
    citation_policy = cfg.get("citation_policy", "strict")
    for variant, entry in prepared["requests"].items():
        req = read(out / entry["path"]); key = entry["job_id"]; payload, readiness = None, []
        if key:
            path = out / "observations" / (key + ".json")
            require(path.exists(), "Run infer before finalize; no synthetic model fallback")
            record = read(path); records[key] = record
            gate = gate_record(record, "not_reviewed", "baseline_local", citation_policy=citation_policy)
            payload, readiness = gate["effective_observations"], gate["readiness"]
            write(out / "gates" / (key + ".json"), gate)
        review = compare(req, payload)
        result = apply_assessment(req, review, evidence_root=out)
        describe_report(req, result)
        result.update(backend=VERSION, clinical_correctness="not_evaluated", readiness=readiness,
                      citation_policy=citation_policy, citation_policy_version=CITATION_POLICY_VERSION)
        folder = out / "studies" / sample["study_id"] / variant
        _save_study(folder, req, result)
        accepted = bool(payload and all(r["correction_eligible"] for r in readiness))
        reasons = list(dict.fromkeys(c for r in readiness for c in r["causes"])) if key else ["no_mapped_whole_finding_T2_or_intra_core_scope_unresolved"]
        write(folder / "policy.json", {"accepted":accepted, "readiness":readiness, "reasons":reasons,
            "citation_policy":citation_policy, "citation_policy_version":CITATION_POLICY_VERSION,
            "citation_normalization":gate.get("citation_normalization", []) if key else [],
            "identity_source":"model_local_baseline_not_independent_review", "clinical_correctness":"not_evaluated"})
        outputs.append({"study_id":sample["study_id"], "variant":variant, "status":"completed",
            "mapped":req["preparation"]["mapped"], "accepted":accepted, "edits":len(result["applied_edits"]),
            "result_path":(folder / "result.json").relative_to(out).as_posix()})
    model_failed = any(r["execution"] != "generated" for r in records.values())
    format_counts = observation_format_counts(records.values())
    write(out / "summary.json", {"baseline":VERSION, "status":"completed_with_model_failures" if model_failed else "completed",
        "outputs":outputs, "unique_model_calls":len(records), "gpu_required":True, "clinical_correctness":"not_evaluated",
        "method_id":VERSION + ":" + CITATION_POLICY_VERSION + ":" + citation_policy,
        "citation_policy":citation_policy, "citation_policy_version":CITATION_POLICY_VERSION,
        **format_counts,
        "schema_valid_calls":sum(r["response_status"] == "valid" for r in records.values()),
        "legacy_schema_valid_calls_means":"raw_observation_contract_valid_calls; use json_schema_valid_calls for JSON schema",
        "note":"Shared pretrained image observations; no clinical label or small numeric observer. Completion/format is not image correctness."})
    present(config_path)
    print("Product:", out / "presentation.html", flush=True)


def observation_format_counts(records):
    """Measure syntax/schema separately from the stricter observation contract."""
    import jsonschema
    from brain_vlm.refinement.observations import parse_model_response, sequence_schema, validate_observations
    counts = {"json_schema_valid_calls": 0, "raw_observation_contract_valid_calls": 0}
    for record in records:
        try:
            parsed, _ = parse_model_response(record.get("raw_response") or "", "vlm_image_first")
            jsonschema.validate(parsed, sequence_schema(record["task"]))
            counts["json_schema_valid_calls"] += 1
            validate_observations(record["task"], parsed)
            counts["raw_observation_contract_valid_calls"] += 1
        except (ValueError, TypeError, KeyError, jsonschema.ValidationError):
            continue
    return counts


def present(config_path):
    from brain_vlm.result_view import esc, details, gallery, diff_html
    _, _, out, _, _ = context(config_path)
    evidence = read(out / "evidence.json")
    text = "<!doctype html><html lang='vi'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Pretrained MedGemma baseline</title><style>body{font:16px/1.5 system-ui;max-width:1200px;margin:auto;padding:24px;color:#17304b}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f1f5f9;padding:12px}section{border:1px solid #ccd;padding:20px;margin:18px 0}.gallery{display:flex;flex-wrap:wrap}figure{width:220px;margin:6px}img{max-width:100%}.columns{display:grid;grid-template-columns:1fr 1fr;gap:12px}del{background:#fecaca}ins{background:#bbf7d0}</style><h1>Pretrained MedGemma → observation → exact Findings edits</h1><p>Model chính: MedGemma1.5 4B pretrained, không native small heads. PNG bên dưới là ảnh thực vào vision encoder. Task đầu tiên vẫn primary-finding T2 direction; sparse views không bảo đảm coverage toàn volume. PRIMA chưa bật ở arm B0 này; không truyền PRIMA embeddings vào MedGemma.</p>"
    text += "<h2>M0/M6 — input MRI → actual PNG views</h2>" + gallery(out, evidence) + details("Evidence metadata", evidence)
    if (out / "model_load.json").exists():text += details("Actual pretrained model load, weights and RAM/VRAM", read(out / "model_load.json"))
    if (out / "summary.json").exists():
        summary = read(out / "summary.json"); text += details("Execution, schema status and output inventory", summary)
        for record in sorted((out / "observations").glob("*.json")):
            text += "<h2>M7 — raw pretrained call</h2>" + details(record.stem, read(record))
            gate_path = out / "gates" / record.name
            if gate_path.exists():
                text += details("Effective observation, citation policy and audit (raw unchanged)", read(gate_path))
        for row in summary["outputs"]:
            folder = (out / row["result_path"]).parent; result = read(folder / "result.json")
            text += "<section><h2>" + esc(row["variant"]) + " — M3 → M8/M9</h2>" + diff_html(result["original_findings"],result["revised_findings"])
            text += details("Claim/permissions",read(folder / "request.json")) + details("Policy and model-local identity",read(folder / "policy.json"))
            text += details("Decisions",result["report_decisions"]) + details("Unresolved",result["unresolved_items"]) + "</section>"
    text += "<p>Controlled drafts contain artificial errors; no clinical accuracy is established. If the same model generates a draft and verifies it, their agreement is correlated, not independent validation.</p></html>"
    (out / "presentation.html").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "infer", "finalize", "present"])
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    globals()[args.action](args.config)
