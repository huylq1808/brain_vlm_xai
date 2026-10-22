"""Neutral image observations, strict saved-call reuse and comparison to draft claims."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from brain_vlm.refinement.contracts import draft_hash, image_path, read_json, require, review_template, validate_request, write_json
from brain_vlm.refinement.claim_extraction import REGIONAL_ENTITY, REGIONAL_ANCHOR

VERSION = "m7_observation_v5"
OBSERVATION_VERSION = "image_first_signal_v1"
SIGNALS = ["low", "high", "intermediate", "mixed", "unresolved"]
SIGNAL_ALIASES = {"low":"low", "hypointense":"low", "high":"high", "hyperintense":"high",
                  "intermediate":"intermediate", "isointense":"intermediate", "mixed":"mixed"}

def parse_model_response(raw, baseline="vlm_image_first"):
    """Normalize known wrappers only; never infer or repair a clinical value."""
    require(isinstance(raw, str) and raw.strip(), "Empty model response")
    payload, steps = raw.strip(), []
    if payload.startswith("<unused94>"):
        require(payload.count("<unused95>") == 1, "Missing or ambiguous MedGemma final-answer marker")
        payload = payload.split("<unused95>", 1)[1].strip()
        steps.append("medgemma_final_answer")
    lines = payload.splitlines()
    if lines and lines[0].strip() in {"```json", "```"}:
        require(len(lines) >= 3 and lines[-1].strip() == "```", "Incomplete JSON fence")
        payload = "\n".join(lines[1:-1])
        steps.append("json_fence")
    parsed = json.loads(payload)
    require(isinstance(parsed, dict), "Expected a JSON object")
    return parsed, steps
def _hash(value):
    return draft_hash(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")))

def observation_task(request, spec):
    """Project only curated queries, sequence inventory and image provenance.

    Signal assertions and claim/slot/case/entity IDs never enter the visible task.
    A bounded draft location may enter as an explicitly uncertain search cue.
    Opaque IDs also prevent values embedded in upstream names entering prompts.
    """
    validate_request(request)
    require(isinstance(spec, dict) and spec.get("schema_version") == 1, "Supply a versioned observation spec")
    slots = {s["slot_id"]: s for s in request["slots"]}
    pairs = sorted({(c["entity_id"], c["sequence"]) for c in request["claims"]
                    if c["entity_id"] != "global" and c["sequence"] is not None
                    and (c.get("observation_kind") == "regional_presence"
                         or any(slots[s]["attribute"] == "signal" for s in c["required_slots"]))})
    require(pairs, "Image-first needs mapped signal or regional-presence claims")
    targets, bindings = {}, {}
    for index, (entity, sequence) in enumerate(pairs, 1):
        anchor = REGIONAL_ANCHOR if entity == REGIONAL_ENTITY else spec.get("entities", {}).get(entity)
        require(isinstance(anchor, dict), f"Missing independently curated entity query: {entity}")
        require(all(isinstance(anchor.get(k), str) and anchor[k].strip()
                    for k in ("description", "signal_reference")), "Incomplete entity query")
        key = f"target_{index:02d}"
        targets[key] = {"description": anchor["description"], "signal_reference": anchor["signal_reference"],
                        "sequence": sequence}
        if entity != REGIONAL_ENTITY:
            targets[key].update(finding_ref="finding_01",location_hypothesis=anchor.get("location_hypothesis"),
                               finding_mapping=anchor.get("finding_mapping","single_finding_hypothesis"))
        if entity == REGIONAL_ENTITY:
            targets[key].update(observation_kind="regional_presence", scope="sampled_region_context")
        bindings[(entity, sequence)] = key
    visible, sources, evidence_map = [], [], {}
    aliases = {item["evidence_id"]: f"image_{i:02d}" for i,item in enumerate(request["evidence"],1)}
    for index, item in enumerate(request["evidence"], 1):
        key = f"image_{index:02d}"
        visible.append({"evidence_id": key, "sequence": item["sequence"], "scope": item["scope"],
                        "view_type": item.get("view_type", "native_slice")})
        visible[-1].update({k:item[k] for k in ["role", "native_canonical_z", "canonical_volume_shape", "geometry_link_status", "roi_sampling", "display_convention"] if k in item})
        if item.get("finding_id"):visible[-1]["finding_ref"] = "finding_01"
        if item.get("paired_context_id"):
            visible[-1]["paired_context_id"] = aliases[item["paired_context_id"]]
        if item.get("observation_kind"):
            visible[-1]["observation_kind"] = item["observation_kind"]
            visible[-1].update({k: item[k] for k in ["canonical_ras_z", "canonical_volume_shape", "whole_region_absence_coverage"] if k in item})
        sources.append({"sha256": item["sha256"], "source_sha256": item.get("source_sha256"),
                        "sequence": item["sequence"], "scope": item["scope"],
                        "view_type": item.get("view_type", "native_slice")})
        if item.get("observation_kind"):
            sources[-1]["observation_kind"] = item["observation_kind"]
        evidence_map[key] = item["evidence_id"]
    task = {"protocol": OBSERVATION_VERSION, "targets": targets, "evidence": visible}
    return task, sources, bindings, evidence_map

def observation_schema(task):
    ids = [e["evidence_id"] for e in task["evidence"]]
    def obj(properties):
        return {"type": "object", "properties": properties, "required": list(properties),
                "additionalProperties": False}
    row = obj({"reason": {"type": "string", "minLength": 1, "maxLength": 160},
               "entity_status": {"enum": ["identified", "unresolved"]},
               "assessability": {"enum": ["sufficient", "insufficient"]},
               "observed_signal": {"enum": SIGNALS},
               "evidence_ids": {"type": "array", "items": {"type": "string", "enum": ids},
                                "maxItems": len(ids)}})
    presence = obj({"reason": {"type": "string", "minLength": 1, "maxLength": 160},
                    "entity_status": {"enum": ["identified", "unresolved"]},
                    "assessability": {"enum": ["sufficient", "insufficient"]},
                    "observed_presence": {"enum": ["visible", "not_seen", "unresolved"]},
                    "evidence_ids": deepcopy(row["properties"]["evidence_ids"])})
    return obj({"targets": obj({key: deepcopy(presence if t.get("observation_kind") == "regional_presence" else row)
                                for key, t in task["targets"].items()})})

def observation_prompt(task):
    if task.get("protocol") == "compact_image_first_v1":
        return compact_observation_prompt(task)
    if any(t.get("observation_kind") == "regional_presence" for t in task["targets"].values()):
        return (
            "Inspect only the supplied brain MRI views using the neutral regional search instruction. "
            "No draft assertion is supplied. The query does not establish that a finding exists. "
            "For regional_presence targets report observed_presence: visible if a signal abnormality can be "
            "identified in the requested region; not_seen if none is seen in these views; unresolved if unclear. "
            "Assessability refers only to this local observation, never to absence in the entire region or volume. "
            "Visible requires identified entity_status, sufficient assessability and same-sequence citations. "
            "Not_seen requires unresolved entity_status and citations to inspected views; it is NOT study absence. "
            "Not_seen may coexist with insufficient assessability when inspection is limited; this is not a regional absence verdict. "
            "Do not diagnose etiology, judge significance, "
            "or infer signal on another sequence. Return only JSON with targets and the exact schema fields: "
            "entity_status, assessability, observed_presence, evidence_ids, reason (1-160 characters).\nINPUT:\n"
            + json.dumps(task, ensure_ascii=False))
    return (
        "Observe the supplied brain MRI images independently. No draft report or asserted signal is supplied. "
        "For each target query, identify the same finding in the named sequence, if possible. "
        "The query is a search instruction, not evidence that a finding exists. "
        "Candidate crops are proposals, not confirmed findings. Use paired_full_slice images for the exact native plane and study_context images for wider anatomy. "
        "A location_hypothesis is a search cue from the draft, not established anatomy. Same-grid region proposals across sequences do not certify the same finding/component. "
        "Crop and paired full plane use the same intensity window. A selected crop may miss the target; do not identify normal tissue as the finding merely because it was cropped. "
        "Compare signal with the stated reference tissue: low, high, intermediate (similar), or mixed. "
        "If findings compete, cannot be linked, or are too small/unclear, use unresolved entity_status, "
        "insufficient assessability and unresolved observed_signal. Do not summarize heterogeneous signal "
        "as uniformly high or low. A finding seen only on another sequence is insufficient for this signal. "
        "Not seeing a finding in sampled views does not establish its absence in the study. "
        "Cite only images from the queried sequence for a sufficient observation. "
        "Return JSON only with exactly targets, keyed by the target IDs below. Every target has exactly: "
        "reason (actual image observation, 1–160 characters), entity_status (identified|unresolved), "
        "assessability (sufficient|insufficient), observed_signal (low|high|intermediate|mixed|unresolved), "
        "evidence_ids (array of supplied image IDs). Insufficient requires unresolved observed_signal.\n"
        "INPUT:\n" + json.dumps(task, ensure_ascii=False))


def compact_observation_task(task):
    """Keep the target/sequence/reference but remove repeated search instructions.

    This changes the visible task, not image selection or any asserted signal.
    Current scope is the bounded focal-signal task; regional tasks stay legacy.
    """
    require(all(t.get("observation_kind", "signal") == "signal" for t in task["targets"].values()),
            "Compact v1 supports focal signal observations only")
    targets = {}
    for key, value in task["targets"].items():
        target = deepcopy(value)
        target["description"] = target["description"].split(". ", 1)[0].rstrip(".")
        target["signal_reference"] = target["signal_reference"].split(";", 1)[0].strip()
        targets[key] = target
    keys = ["evidence_id", "sequence", "scope", "view_type", "role", "native_canonical_z", "paired_context_id"]
    evidence = [{k: e[k] for k in keys if k in e} for e in task["evidence"]]
    return {"protocol": "compact_image_first_v1", "targets": targets, "evidence": evidence}


def compact_observation_prompt(task):
    """An opt-in concise prompt; no answer exemplar, clinical value or repair."""
    return (
        "Inspect the supplied MRI images. Answer only the requested sequence's signal question. "
        "Find the focal abnormality if visible; the location is a draft-derived search hypothesis and may be wrong. "
        "A crop is a proposal, not proof of a lesion. Compare the finding with the specified reference tissue. "
        "Use full slices for anatomy and the paired crop for detail. Do not infer another sequence's signal. "
        "If you cannot identify the finding and reference, return unresolved entity_status, insufficient "
        "assessability and unresolved observed_signal. Sampled views cannot establish whole-study absence. "
        "Write reason as your own visible image observation (1-160 characters); do not repeat the query. "
        "Cite only the 1-3 most informative supplied image IDs for a sufficient observation; use [] if no view supports it. "
        "Return only a JSON object with targets keyed by the supplied target ID. Each target has exactly: "
        "entity_status (identified|unresolved), assessability (sufficient|insufficient), "
        "observed_signal (low|high|intermediate|mixed|unresolved), evidence_ids (array), reason (string).\n"
        "INPUT:\n" + json.dumps(task, ensure_ascii=False))

def validate_observations(task, payload):
    require(isinstance(payload, dict) and set(payload) == {"targets"}, "Unexpected observation fields")
    rows = payload["targets"]
    require(isinstance(rows, dict) and set(rows) == set(task["targets"]), "Missing/unexpected observation targets")
    evidence = {e["evidence_id"]: e for e in task["evidence"]}
    for key, row in rows.items():
        if task["targets"][key].get("observation_kind") == "regional_presence":
            require(isinstance(row, dict) and set(row) == {"reason", "entity_status", "assessability",
                    "observed_presence", "evidence_ids"}, "Unexpected regional presence fields")
            require(isinstance(row["reason"], str) and 0 < len(row["reason"].strip()) <= 160
                    and row["reason"].strip().lower().rstrip(".") != "review pending", "Invalid presence reason")
            require(row["entity_status"] in {"identified", "unresolved"}
                    and row["assessability"] in {"sufficient", "insufficient"}, "Invalid presence state")
            require(row["observed_presence"] in {"visible", "not_seen", "unresolved"}, "Invalid presence observation")
            ids = row["evidence_ids"]
            require(isinstance(ids, list) and all(isinstance(e, str) for e in ids)
                    and len(ids) == len(set(ids)) and set(ids) <= set(evidence), "Invalid presence citations")
            require(all(evidence[e]["sequence"] == task["targets"][key]["sequence"] for e in ids),
                    "Presence citations must use the requested sequence")
            if row["observed_presence"] == "visible":
                require(row["entity_status"] == "identified" and row["assessability"] == "sufficient" and ids,
                        "Visible finding requires sufficient identified same-sequence evidence")
            else:
                require(row["entity_status"] == "unresolved", "No identified finding without visible presence")
            if row["assessability"] == "insufficient":
                require(row["observed_presence"] in {"not_seen", "unresolved"}, "Insufficient presence cannot assert a visible finding")
            if row["assessability"] == "sufficient":
                require(ids and row["observed_presence"] != "unresolved", "Sufficient local presence needs cited views")
            continue
        require(isinstance(row, dict) and set(row) == {"reason", "entity_status", "assessability",
                                                     "observed_signal", "evidence_ids"}, "Unexpected target fields")
        require(isinstance(row["reason"], str) and 0 < len(row["reason"].strip()) <= 160,
                "Missing/overlong observation reason")
        require(row["reason"].strip().lower().rstrip(".") != "review pending", "Pending observation reason")
        require(row["entity_status"] in {"identified", "unresolved"}, "Invalid observed entity")
        require(row["assessability"] in {"sufficient", "insufficient"}, "Invalid observation assessability")
        require(row["observed_signal"] in SIGNALS, "Invalid observed signal")
        ids = row["evidence_ids"]
        require(isinstance(ids, list) and all(isinstance(e, str) for e in ids)
                and len(ids) == len(set(ids)) and set(ids) <= set(evidence), "Invalid observation evidence IDs")
        if row["entity_status"] == "identified":
            require(ids and all(evidence[e]["sequence"] == task["targets"][key]["sequence"] for e in ids),
                    "Identified target needs evidence from its sequence")
        if row["assessability"] == "sufficient":
            require(row["entity_status"] == "identified" and row["observed_signal"] != "unresolved",
                    "Sufficient signal needs identified entity and observed value")
            require(ids and all(evidence[e]["sequence"] == task["targets"][key]["sequence"] for e in ids),
                    "Sufficient signal needs evidence from its sequence")
        else:
            require(row["observed_signal"] == "unresolved", "Insufficient observation cannot assert signal")
    return payload

def compare_observations(request, payload, bindings, evidence_map, model_id, reviewer_kind="model"):
    """Compare categorical values only; unobserved attributes stay unresolved."""
    review = review_template(request, reviewer_kind, model_id)
    slots = {s["slot_id"]: s for s in request["slots"]}
    for claim, row in zip(request["claims"], review["claims"]):
        target = bindings.get((claim["entity_id"], claim["sequence"]))
        observed = payload["targets"].get(target) if payload is not None else None
        row["entity_status"] = observed["entity_status"] if observed else "unresolved"
        for item in row["attributes"]:
            item["reason"] = "Outside image-first v1 signal/presence scope; requires separate review."
            if not observed:
                continue
            slot = slots[item["slot_id"]]
            ids = [evidence_map[e] for e in observed["evidence_ids"]]
            if claim.get("observation_kind") == "regional_presence":
                item.update(evidence_ids=ids)
                if observed.get("observed_presence") == "visible" and observed["assessability"] == "sufficient":
                    if claim.get("qualifiers"):
                        item["reason"] = "Local abnormality observed; qualifier has no reviewed threshold: " + ", ".join(claim["qualifiers"]) + ". " + observed["reason"]
                    else:
                        item.update(assessability="sufficient", verdict="contradicted",
                                    reason="Local abnormality contradicts unqualified regional absence; replacement requires the bounded wording policy. " + observed["reason"])
                else:
                    item["reason"] = "Sampled views do not establish whole-region absence. " + observed["reason"]
                continue
            if slot["attribute"] == "presence":
                if slot["asserted_value"] == "present" and observed["entity_status"] == "identified":
                    item.update(assessability="sufficient", verdict="supported", evidence_ids=ids,
                                reason="Target identified in the cited sequence images: " + observed["reason"])
                else:
                    item["reason"] = "Target absence is not established by this observation. " + observed["reason"]
            elif slot["attribute"] == "signal":
                if observed["assessability"] != "sufficient":
                    item["reason"] = observed["reason"]
                    continue
                value = observed["observed_signal"]
                assertion = SIGNAL_ALIASES.get(slot["asserted_value"].strip().lower())
                if assertion is None or value == "mixed":
                    item.update(evidence_ids=ids, reason="Mixed signal or unmapped assertion; exact comparison deferred. "
                                + observed["reason"])
                    continue
                verdict = "supported" if value == assertion else "contradicted"
                replacement = next((v for v in slot["allowed_replacements"]
                                    if SIGNAL_ALIASES.get(v.strip().lower()) == value), None)
                item.update(assessability="sufficient", verdict=verdict, evidence_ids=ids,
                            reason=f"Observed {value}; asserted {slot['asserted_value']}. " + observed["reason"])
                if verdict == "contradicted" and replacement is not None:
                    item.update(replacement=replacement, replacement_supported=True)
    return review
def job_fingerprint(job, model_config):
    # _hash sorts object keys: explicit unsorted schema text preserves LMFE field order.
    return _hash({"version": VERSION, "task": job["task"], "sources": job["sources"],
                  "ordered_schema": json.dumps(job["response_schema"], ensure_ascii=False),
                  "prompt": job["prompt"], "model_config": model_config})

def run_job(job, generator, model_config, destination):
    """Save each call immediately. Invalid/raw false-sufficient rows remain inspectable."""
    for image, source in zip(job["images"], job["sources"]):
        require(hashlib.sha256(Path(image["resolved_path"]).read_bytes()).hexdigest() == source["sha256"],
                "Observation PNG changed")
    fingerprint = job_fingerprint(job, model_config)
    path = Path(destination) / "observations" / (job["job_id"] + ".json")
    if path.is_file():
        record = read_json(path)
        require(record["fingerprint"] == fingerprint, "Changed experiment: choose a new run")
        if record["execution"] == "generated":
            # Validate cache content, not just its stored fingerprint.
            require(record["task"] == job["task"] and record["sources"] == job["sources"]
                    and record["prompt"] == job["prompt"]
                    and json.dumps(record["response_schema"]) == json.dumps(job["response_schema"])
                    and record["model_config"] == model_config, "Changed cached observation")
            import jsonschema
            parsed, payload = None, None
            try:
                parsed, _ = parse_model_response(record["raw_response"], "vlm_image_first")
                jsonschema.validate(parsed, job["response_schema"])
                payload = validate_observations(job["task"], parsed)
            except (ValueError, TypeError, KeyError, AttributeError, jsonschema.ValidationError):
                pass
            require(record["parsed_response"] == parsed and record["observations"] == payload
                    and record["response_status"] == ("valid" if payload is not None else "invalid"),
                    "Cached parsed assessment differs from raw response")
            return record, "reused"
    record = {"version": VERSION, "job_id": job["job_id"], "case": job["case"],
              "group": job["group"], "sequence": job["sequence"], "fingerprint": fingerprint,
              "task": job["task"], "sources": job["sources"], "prompt": job["prompt"],
              "response_schema": job["response_schema"], "model_config": model_config,
              "raw_response": None, "parsed_response": None, "observations": None,
              "response_status": "not_generated", "warnings": [], "generator_metadata": None}
    try:
        generator.set_response_schema(job["response_schema"])
        record["raw_response"] = generator(job["prompt"], job["images"])
        record["execution"] = "generated"
        try:
            import jsonschema
            parsed, normalization = parse_model_response(record["raw_response"], "vlm_image_first")
            record["parsed_response"] = parsed
            record["response_normalization"] = normalization
            jsonschema.validate(parsed, job["response_schema"])
            record["observations"] = validate_observations(job["task"], parsed)
            record["response_status"] = "valid"
        except (ValueError, TypeError, KeyError, AttributeError, jsonschema.ValidationError) as error:
            record["response_status"] = "invalid"
            record["warnings"].append(str(error))
    except Exception as error:
        record["execution"] = "failed"
        record["warnings"].append(type(error).__name__ + ": " + str(error))
    finally:
        record["generator_metadata"] = deepcopy(getattr(generator, "metadata", {}))
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, record)
    return record, record["execution"]
def sequence_schema(task):
    """Limit citations using known inventory, never using draft signal values."""
    schema = observation_schema(task)
    for key, target in task["targets"].items():
        row = schema["properties"]["targets"]["properties"][key]
        ids = [e["evidence_id"] for e in task["evidence"] if e["sequence"] == target["sequence"]]
        row["properties"]["evidence_ids"] = {
            "type": "array", "items": {"type": "string", **({"enum": ids} if ids else {})},
            "maxItems": len(ids)}
        value = "observed_presence" if target.get("observation_kind") == "regional_presence" else "observed_signal"
        order = ["entity_status", "assessability", value, "evidence_ids", "reason"]
        row["properties"] = {k: row["properties"][k] for k in order}
        row["required"] = order
    return schema


def unresolved_observation(target, reason):
    field = "observed_presence" if target.get("observation_kind") == "regional_presence" else "observed_signal"
    return {"entity_status": "unresolved", "assessability": "insufficient", field: "unresolved",
            "evidence_ids": [], "reason": reason[:160]}

def job_config(job, model_config):
    return {**model_config, "readiness_version": VERSION, "output_policy": job["output_policy"]}

def fingerprint(job, model_config):
    return job_fingerprint(job, job_config(job, model_config))

def run_observation(job, generator, model_config, destination):
    if generator is not None:
        generator.configure_response(**job["output_policy"])
    return run_job(job, generator, job_config(job, model_config), destination)
