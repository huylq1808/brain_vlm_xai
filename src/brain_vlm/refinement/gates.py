"""Binding and per-target evidence gates; raw model judgments are never overwritten."""
from copy import deepcopy
import hashlib
import re
from pathlib import Path
from brain_vlm.refinement.contracts import read_json, require
from brain_vlm.refinement.observations import observation_schema, validate_observations, observation_task, parse_model_response, unresolved_observation

VERSION = "m7_readiness_v4"


def evidence_reference_audit(row, evidence):
    """Check explicit opaque image pointers, not the medical content of a reason.

    A reason need not repeat IDs. If it names an image_XX pointer, that pointer
    must exist and be declared in evidence_ids. Never add citations for a model.
    """
    reason = row.get("reason") if isinstance(row, dict) else None
    mentioned = set(re.findall(r"\bimage_\d+\b", reason.casefold())) if isinstance(reason, str) else set()
    known = {e["evidence_id"].casefold() for e in evidence}
    ids = row.get("evidence_ids", []) if isinstance(row, dict) else []
    cited = {i.casefold() for i in ids if isinstance(i, str)} if isinstance(ids, list) else set()
    return {"mentioned": sorted(mentioned), "unknown": sorted(mentioned-known),
            "uncited": sorted((mentioned & known)-cited)}


def query_echo(reason, target):
    """Catch copied queries, including a long substring cut by schema limits.

    This detects a textual failure, not medical correctness. Short common MRI
    phrases and reasons with additional image observations are not rejected just
    for sharing words. A word cut mid-token still matches the source substring.
    """
    normalize = lambda value: " ".join(re.findall(r"\w+", value.casefold())) if isinstance(value, str) else ""
    text = normalize(reason)
    if not text:return False
    for key in ["description", "signal_reference", "location_hypothesis"]:
        source = normalize(target.get(key))
        if source and (text == source or (len(text.split()) >= 8 and text in source)):
            return True
    return False

def binding_template(input_root):
    root = Path(input_root)
    return {"schema_version": 1, "input_manifest_sha256": hashlib.sha256(
        (root / "input_manifest.json").read_bytes()).hexdigest(), "cases": {
        case: {"status": "not_reviewed", "finding_id": "", "component_id": "", "evidence_ids": [],
               "reviewer": "", "reviewer_role": "", "notes": ""}
        for case in read_json(root / "input_manifest.json")["cases"]}}

def binding_state(input_root, case, review):
    """External linkage review, bound to exact input manifest; no model-derived approval."""
    require(review.get("schema_version") == 1 and review.get("input_manifest_sha256") ==
            binding_template(input_root)["input_manifest_sha256"], "Stale/invalid binding review")
    entry = review.get("cases", {}).get(case)
    require(isinstance(entry, dict), "Missing binding review case")
    status = entry.get("status")
    require(status in {"not_reviewed", "matched", "ambiguous", "unmatched"}, "Invalid binding status")
    if status == "matched":
        require(all(isinstance(entry.get(k), str) and entry[k].strip()
                    for k in ["finding_id", "component_id", "reviewer", "reviewer_role"]),
                "Matched linkage needs finding/component and independent reviewer provenance")
        root = Path(input_root)
        manifest_entry = read_json(root / "input_manifest.json")["cases"][case]
        folder = root / manifest_entry["folder"]
        variants = [v for v in ["source", "generated", "controlled"] if v + "_request_sha256" in manifest_entry]
        require(variants, "Binding needs a declared draft request")
        request = read_json(folder / (variants[0] + "_request.json"))
        from brain_vlm.refinement.contracts import request_hash
        require(request_hash(request) == manifest_entry[variants[0] + "_request_sha256"], "Binding request changed")
        require(len({c["entity_id"] for c in request["claims"] if c["entity_id"] != "global"}) == 1,
                "This small linkage form supports one curated entity/group per case")
        task, _, _, _ = observation_task(request, read_json(folder / "observation_spec.json"))
        inventory = {e["evidence_id"]: e["sequence"] for e in task["evidence"]}
        ids = entry.get("evidence_ids")
        require(isinstance(ids, list) and ids and len(ids) == len(set(ids)) and set(ids) <= set(inventory),
                "Linkage evidence IDs must be unique supplied images")
        require({inventory[i] for i in ids} >= {t["sequence"] for t in task["targets"].values()},
                "Matched linkage must cite the required sequences")
    return status

def unique_citation_sets(payload):
    """Stable set representation only: never add/filter IDs or change judgments."""
    normalized=deepcopy(payload);audit=[]
    for key,row in normalized.get("targets",{}).items():
        ids=row.get("evidence_ids") if isinstance(row,dict) else None
        if not isinstance(ids,list) or not all(isinstance(i,str) for i in ids):continue
        unique=list(dict.fromkeys(ids))
        if unique!=ids:
            audit.append({"target":key,"raw_ids":ids.copy(),"effective_ids":unique.copy(),
                          "removed_duplicate_occurrences":len(ids)-len(unique)})
            row["evidence_ids"]=unique
    return normalized,audit


def gate_record(record, linkage="not_reviewed", policy="strict", *, citation_policy="strict"):
    """Preserve raw judgment, produce a separate effective payload for M8/M9."""
    require(linkage in {"not_reviewed", "matched", "ambiguous", "unmatched"}, "Invalid linkage")
    require(policy in {"strict", "baseline_local"}, "Unknown decision policy")
    require(citation_policy in {"strict","stable_unique"},"Unknown citation representation policy")
    citation_audit=[]
    task, parsed = record["task"], None
    try:
        parsed, _ = parse_model_response(record.get("raw_response") or "", "vlm_image_first")
        require(isinstance(parsed, dict) and set(parsed) == {"targets"} and
                isinstance(parsed["targets"], dict) and set(parsed["targets"]) == set(task["targets"]),
                "Invalid target envelope")
        if citation_policy=="stable_unique":parsed,citation_audit=unique_citation_sets(parsed)
    except (ValueError, TypeError, KeyError, AttributeError):
        parsed = None
    effective, decisions = {}, []
    for key, target in task["targets"].items():
        row = parsed["targets"][key] if parsed else None
        references = evidence_reference_audit(row, task["evidence"])
        causes = []
        if target.get("finding_mapping")=="ambiguous":causes.append("ambiguous_finding_mapping")
        if record.get("execution") != "generated" or (record.get("generator_metadata") or {}).get("output_hit_token_cap"):
            causes.append("generation_incomplete")
        if (policy == "baseline_local" or target.get("observation_kind") == "regional_presence") and (record.get("generator_metadata") or {}).get("last_stop_token_id") is None:
            causes.append("no_observed_stop_token")
        if not any(e["sequence"] == target["sequence"] for e in task["evidence"]):
            causes.append("missing_sequence")
        try:
            import jsonschema
            single_task = {**task, "targets": {key: target}}
            jsonschema.validate({"targets": {key: row}}, observation_schema(single_task))
            validate_observations(single_task, {"targets": {key: row}})
        except (ValueError, TypeError, KeyError, AttributeError, jsonschema.ValidationError):
            causes.append("invalid_observation")
        if references["unknown"]:causes.append("reason_references_unknown_image")
        if references["uncited"]:causes.append("reason_references_uncited_image")
        if linkage != "matched" and not (policy == "baseline_local" and linkage == "not_reviewed"):
            causes.append("unreviewed_binding" if linkage == "not_reviewed" else "ambiguous_binding"
                          if linkage == "ambiguous" else "unmatched_binding")
        if policy == "baseline_local" and target.get("observation_kind") != "regional_presence" and isinstance(row, dict):
            if query_echo(row.get("reason", ""), target):
                causes.append("query_echo_without_image_description")
        model_abstained = None
        if isinstance(row, dict):
            if target.get("observation_kind") == "regional_presence":
                # A clear local not_seen answer is not a model abstention. Scope
                # rules below comparison still cannot turn it into regional absence.
                model_abstained = row.get("observed_presence") == "unresolved" or row.get("assessability") == "insufficient"
            elif row.get("entity_status") == "unresolved" or row.get("assessability") == "insufficient":
                model_abstained = True
            elif row.get("entity_status") == "identified" and row.get("assessability") == "sufficient":
                model_abstained = False
        mixed = isinstance(row, dict) and row.get("observed_signal") == "mixed"
        effective[key] = unresolved_observation(target, "Pipeline deferred: " + ", ".join(causes)) if causes else deepcopy(row)
        decisions.append({"target": key, "sequence": target["sequence"], "linkage": linkage,
                          "evidence_reference_audit": references,
                          "identity_source": "model_local_baseline" if policy == "baseline_local" and linkage == "not_reviewed" else "independent_review" if linkage == "matched" else "unresolved",
                          "pipeline_blocked": bool(causes), "model_abstained": model_abstained,
                          "correction_eligible": target.get("observation_kind") != "regional_presence" and not causes and not model_abstained and not mixed,
                          "causes": causes + (["model_insufficient"] if model_abstained else [])
                          + (["mixed_signal"] if mixed else [])})
    payload = validate_observations(task, {"targets": effective})
    result = {"version": VERSION, "job_id": record["job_id"], "raw_response_status": record["response_status"],
            "readiness": decisions, "effective_observations": payload,
            "clinical_correctness": "not_evaluated"}
    if citation_policy!="strict":result.update(citation_policy=citation_policy,
        citation_normalization_version="evidence_id_set_v1",citation_normalization=citation_audit)
    return result
