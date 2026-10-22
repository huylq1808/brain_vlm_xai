"""M8 policy and M9 exact-span editor. No baseline dispatcher or free-form rewrite."""
from copy import deepcopy
import time
from brain_vlm.refinement.contracts import check_span, image_path, require, request_hash, validate_request, validate_review
from brain_vlm.refinement.claim_extraction import unmapped_spans, assessment_coverage, sentence_spans

PROTOCOL_VERSION = "m7_json_v3"
BASELINE_VERSION = "brain_mri_refinement_baseline_v2"


def apply_assessment(request, review, evidence_root=None, *, decision_policy="strict",
                     observations=None, bindings=None, evidence_map=None):
    """Validate one canonical assessment and apply only authorized exact edits."""
    started = time.perf_counter()
    validate_request(request)
    validate_review(request, review)
    if evidence_root is not None:
        for item in request["evidence"]:
            if item["kind"] == "native_slice":
                image_path(item, evidence_root)
    elif request["purpose"] == "development_mri":
        require(not any(a["verdict"] != "unresolved" for c in review["claims"]
                        for a in c["attributes"]), "Real image verdicts require verified evidence files")
    require(decision_policy in {"strict", "baseline_local"}, "Unknown edit policy")
    plan = build_edit_plan(request, review)
    if decision_policy == "baseline_local":
        plan["policy"] = BASELINE_VERSION
        for patch in plan["patches"]:patch["category"] = "signal"
        plan["patches"].extend(regional_patches(request, observations or {"targets":{}}, bindings or {}, evidence_map or {}))
    result = edit_report(request, plan)
    links = deepcopy(request["evidence"])
    if evidence_root is not None:
        for item in links:
            if item["kind"] == "native_slice":
                item["resolved_path"] = str(image_path(item, evidence_root))
    result.update(review=deepcopy(review), edit_plan=plan, schema_version=1,
                  case_id=request["case_id"], baseline="structured_review",
                  purpose=request["purpose"], request_sha256=request_hash(request),
                  raw_response=None, prompt=None, warnings=[], response_status="not_applicable",
                  response_normalization=[], response_failure_kind=None,
                  protocol_version=PROTOCOL_VERSION, elapsed_seconds=time.perf_counter()-started,
                  generator_metadata=None, evidence_links=links,
                  assessment_coverage=assessment_coverage(request, review))
    if decision_policy == "baseline_local":
        result.update(baseline_version=BASELINE_VERSION, decision_policy="baseline_local", clinical_correctness="not_evaluated")
        describe_report(request, result)
    return result


def regional_wording(claim, category):
    """Finite wording policy; scope edits do not manufacture a factual verdict."""
    if category == "uncertainty":
        qualifier = "clinically significant " if "significant" in claim.get("qualifiers", []) else ""
        return ("The supplied FLAIR images are insufficient to determine whether " + qualifier +
                "periventricular white matter signal abnormalities are present.")
    if category == "finding_and_scope":
        return ("Periventricular white matter signal abnormality is visible in the supplied FLAIR images. "
                "Its clinical significance and full extent are not established by these sampled views.")
    require(category == "scope", "Unknown regional wording policy")
    return ("No definite periventricular white matter signal abnormality is identified in the sampled FLAIR views. "
            "Whole-region absence and clinical significance are not established by this limited inspection.")


def regional_patches(request, observations, bindings, evidence_map):
    patches = []
    sources = {e["evidence_id"]:e for e in request["evidence"]}
    for claim in request["claims"]:
        if claim.get("observation_kind") != "regional_presence":continue
        row = observations["targets"].get(bindings.get((claim["entity_id"],claim["sequence"])))
        if not row or not row["evidence_ids"]:continue
        ids = [evidence_map[i] for i in row["evidence_ids"]]
        require(all(i in sources and sources[i]["sequence"] == claim["sequence"] for i in ids), "Wrong regional evidence")
        if row["assessability"] == "insufficient" or row["observed_presence"] == "unresolved":category = "uncertainty"
        elif row["observed_presence"] == "visible":category = "finding_and_scope"
        else:category = "scope"
        patches.append({**claim["source_span"], "replacement":regional_wording(claim,category),
            "category":category, "claim_ids":[claim["claim_id"]], "slot_ids":claim["required_slots"], "evidence_ids":ids,
            "reason":"Regional wording policy from the cited local observation: " + row["reason"] +
                     " Whole-region extent/significance is not established; this wording is produced by code."})
    return patches


def describe_report(request, result):
    """Sentence explanation from the one applied plan, without another writer."""
    edits, decisions = result["applied_edits"], []
    for span in sentence_spans(request["draft_text"]):
        start,end = span["start"],span["end"]
        local = [p for p in edits if max(start,p["start"]) < min(end,p["end"])]
        claims = [c for c in request["claims"] if c["claim_id"] != "unmapped_draft"
                  and max(start,c["source_span"]["start"]) < min(end,c["source_span"]["end"])]
        attrs = [a for a in result["attribute_decisions"] if a["claim_id"] in {c["claim_id"] for c in claims}]
        action = "correct" if local else "keep" if attrs and all(a["action"] == "keep" for a in attrs) else "flag"
        after = span["text"]
        for p in sorted(local,key=lambda p:p["start"],reverse=True):
            after = after[:p["start"]-start]+p["replacement"]+after[p["end"]-start:]
        reason = " ".join(dict.fromkeys(p.get("reason","Signal replacement supported by the local model observation.") for p in local)) if local else (
            "Mapped attributes supported; other wording remains unassessed." if action == "keep" else
            " ".join(dict.fromkeys(a["reason"] for a in attrs)) if attrs else "Outside supported attributes; preserved and unassessed.")
        decisions.append({"id":f"sentence_{len(decisions)+1:02d}", "source_span":{"start":start,"end":end},
            "input":span["text"], "output":after, "action":action,
            "edit_type":", ".join(dict.fromkeys(p.get("category","signal") for p in local)) if local else "none",
            "reason":reason, "evidence_ids":sorted({i for p in local for i in p["evidence_ids"]}|{i for a in attrs for i in a["evidence_ids"]}),
            "medical_correctness":"not_evaluated"})
    unresolved = [{"sentence":d["id"],"text":d["input"],"reason":d["reason"]} for d in decisions if d["action"] == "flag"]
    unresolved.extend({"claim":a["claim_id"],"slot":a["slot_id"],"reason":a["reason"]}
                      for a in result["attribute_decisions"] if a["action"] == "flag")
    unresolved.extend({"claim":c["claim_id"],"reason":"Whole-region extent/clinical significance remains unresolved."}
                      for c in request["claims"] if c.get("observation_kind") == "regional_presence")
    result.update(report_decisions=decisions, unresolved_items=unresolved,
        edit_counts={k:sum(p.get("category","signal")==k for p in edits) for k in ["signal","finding_and_scope","scope","uncertainty"]},
        status="revised_with_unresolved" if edits and unresolved else "revised" if edits else "unchanged_with_unresolved" if unresolved else "unchanged")

def _unassessed_spans(request):
    return unmapped_spans(request)

def _attribute_action(request, review, claim, slot, item, entity_status):
    action = {"claim_id": claim["claim_id"], "slot_id": slot["slot_id"],
              "attribute": slot["attribute"], "action": "flag", "replacement": None,
              "evidence_ids": [], "reason": "Missing assessment."}
    if item is None:
        return action
    action.update(reason=item["reason"], evidence_ids=item["evidence_ids"])
    if claim["sequence"] is not None and claim["sequence"] not in request["available_sequences"]:
        action["reason"] = "Required sequence is missing; no disease-absence conclusion follows."
        return action
    if review["reviewer"]["kind"] == "text_only":
        action["reason"] = "Text-only judgment has no visual support; retained for ablation only."
        return action
    if item["assessability"] != "sufficient" or item["verdict"] == "unresolved":
        return action
    sources = {e["evidence_id"]: e for e in request["evidence"]}
    if not item["evidence_ids"]:
        action["reason"] = "Factual verdict has no supporting evidence ID."
        return action
    if claim["entity_id"] == "global" and not any(
        sources[e]["scope"] == "study_context" for e in item["evidence_ids"]
    ):
        action["reason"] = "Global assertion needs explicitly reviewed study context."
        return action
    if slot["attribute"] in {"signal", "sequence"} and not any(
        sources[e]["sequence"] == claim["sequence"] for e in item["evidence_ids"]
    ):
        action["reason"] = "No cited evidence from the asserted sequence."
        return action
    if slot["attribute"] not in {"presence", "sequence"} and entity_status != "identified":
        action["reason"] = "Finding identity is unresolved; dependent attribute action blocked."
        return action
    if item["verdict"] == "supported":
        action["action"] = "keep"
    elif (slot["editable"] and item["replacement_supported"]
          and item["replacement"] in slot["allowed_replacements"]
          and item["replacement"] != slot["asserted_value"]):
        action.update(action="correct", replacement=item["replacement"])
    else:
        action["reason"] = "Contradicted, but no supported replacement with an enabled exact edit."
    return action

def build_edit_plan(request, review):
    validate_request(request)
    validate_review(request, review)
    reviewed = {c["claim_id"]: c for c in review["claims"]}
    slots = {s["slot_id"]: s for s in request["slots"]}
    actions = []
    for claim in request["claims"]:
        row = reviewed.get(claim["claim_id"], {})
        items = {a["slot_id"]: a for a in row.get("attributes", [])}
        entity_status = row.get("entity_status", "unresolved")
        presence = [items.get(s) for s in claim["required_slots"] if slots[s]["attribute"] == "presence"]
        if any(p is None or _attribute_action(request, review, claim, slots[p["slot_id"]],
                                             p, entity_status)["action"] != "keep"
               for p in presence):
            entity_status = "unresolved"
        for slot_id in claim["required_slots"]:
            actions.append(_attribute_action(request, review, claim, slots[slot_id],
                                             items.get(slot_id), entity_status))
    # Resolve shared presence before authorizing dependent edits in either claim.
    for slot_id in slots:
        linked = [a for a in actions if a["slot_id"] == slot_id]
        if len({(a["action"], a["replacement"]) for a in linked}) > 1:
            for action in linked:
                action.update(action="flag", replacement=None,
                              reason="Linked claims disagree or lack sufficient shared-slot review.")
    for claim in request["claims"]:
        linked = [a for a in actions if a["claim_id"] == claim["claim_id"]]
        if any(a["attribute"] == "presence" and a["action"] != "keep" for a in linked):
            for action in linked:
                if action["attribute"] not in {"presence", "sequence"} and action["action"] != "flag":
                    action.update(action="flag", replacement=None,
                                  reason="Presence was not accepted after shared-claim checks; dependent action blocked.")
    patches = []
    for slot_id, slot in slots.items():
        linked = [a for a in actions if a["slot_id"] == slot_id]
        signatures = {(a["action"], a["replacement"]) for a in linked}
        if len(linked) > 1 and len(signatures) != 1:
            for action in linked:
                action.update(action="flag", replacement=None,
                              reason="Linked claims disagree or lack sufficient shared-slot review.")
        elif linked[0]["action"] == "correct":
            patches.append({"slot_ids": [slot_id], **slot["source_span"],
                            "replacement": linked[0]["replacement"],
                            "claim_ids": [a["claim_id"] for a in linked],
                            "evidence_ids": sorted({e for a in linked for e in a["evidence_ids"]})})
            if slot["attribute"] == "side":patches[-1]["category"] = "side"
    # Distinct slots may map to the same exact text. Merge identical edits and block conflicts.
    merged = []
    for patch in patches:
        same = next((p for p in merged if (p["start"], p["end"], p["replacement"]) ==
                     (patch["start"], patch["end"], patch["replacement"])), None)
        if same is None:
            merged.append(patch)
        else:
            for key in ("slot_ids", "claim_ids", "evidence_ids"):
                same[key] = sorted(set(same[key]) | set(patch[key]))
    blocked = set()
    for patch in merged:
        for slot_id, slot in slots.items():
            span = slot["source_span"]
            if span is None or slot_id in patch["slot_ids"]:
                continue
            if max(patch["start"], span["start"]) < min(patch["end"], span["end"]):
                blocked.update(patch["slot_ids"])
    for i, first in enumerate(merged):
        for second in merged[i + 1:]:
            if max(first["start"], second["start"]) < min(first["end"], second["end"]):
                blocked.update(first["slot_ids"] + second["slot_ids"])
    if blocked:
        for action in actions:
            if action["slot_id"] in blocked:
                action.update(action="flag", replacement=None, reason="Conflicting overlapping edits.")
    merged = [p for p in merged if not set(p["slot_ids"]) & blocked]
    return {"request_sha256": request_hash(request), "attribute_decisions": actions, "patches": merged}

def edit_report(request, plan):
    validate_request(request)
    require(plan["request_sha256"] == request_hash(request), "Stale edit plan")
    text = request["draft_text"]
    patches = sorted(plan["patches"], key=lambda p: p["start"])
    for i, patch in enumerate(patches):
        check_span(text, patch)
        require(i == 0 or patches[i - 1]["end"] <= patch["start"], "Overlapping edit plan")
        if patch.get("category", "signal") not in {"signal", "side"}:
            require(plan.get("policy") == BASELINE_VERSION and len(patch["claim_ids"]) == 1, "Unknown wording plan")
            claim = next(c for c in request["claims"] if c["claim_id"] == patch["claim_ids"][0])
            require(claim.get("observation_kind") == "regional_presence" and
                    all(patch[k] == claim["source_span"][k] for k in ["start","end","text"]) and
                    patch["replacement"] == regional_wording(claim, patch["category"]), "Wording patch outside bounded policy")
            evidence = {e["evidence_id"]:e for e in request["evidence"]}
            require(patch["evidence_ids"] and all(e in evidence and evidence[e]["sequence"] == claim["sequence"] for e in patch["evidence_ids"]), "Uncited wording patch")
            continue
        slots = {s["slot_id"]: s for s in request["slots"]}
        for slot_id in patch["slot_ids"]:
            require(slot_id in slots, "Unknown edited slot")
            slot = slots[slot_id]
            if patch.get("category") == "side":
                require(slot["attribute"] == "side" and request.get("preparation", {}).get("task") == "s4_mask_grid_side_v1",
                        "Side edit outside opted-in geometry task")
            require(slot["editable"] and slot["source_span"] ==
                    {k: patch[k] for k in ("start", "end", "text")}, "Patch/slot mismatch")
            require(patch["replacement"] in slot["allowed_replacements"], "Replacement outside contract")
            decisions = [a for a in plan["attribute_decisions"] if a["slot_id"] == slot_id]
            linked = [c for c in request["claims"] if slot_id in c["required_slots"]]
            require({a["claim_id"] for a in decisions} == {c["claim_id"] for c in linked}
                    and all(a["action"] == "correct" and a["replacement"] == patch["replacement"]
                            for a in decisions), "Patch lacks compatible linked decisions")
    revised = text
    for patch in reversed(patches):
        revised = revised[:patch["start"]] + patch["replacement"] + revised[patch["end"]:]
    decisions = []
    for claim in request["claims"]:
        linked = [a for a in plan["attribute_decisions"] if a["claim_id"] == claim["claim_id"]]
        flags = [a["slot_id"] for a in linked if a["action"] == "flag"]
        changes = [a["slot_id"] for a in linked if a["action"] == "correct"]
        decisions.append({"claim_id": claim["claim_id"],
                          "action": "flag" if flags else "correct" if changes else "keep",
                          "corrected_slots": changes, "unresolved_slots": flags})
    unassessed = _unassessed_spans(request)
    unresolved = [a for a in plan["attribute_decisions"] if a["action"] == "flag"]
    return {"original_findings": text, "revised_findings": revised,
            "status": "requires_review" if unresolved or unassessed else "completed_within_declared_scope",
            "review_scope": "Only the explicitly declared claim attributes were assessed.",
            "applied_edits": patches, "claim_decisions": decisions,
            "attribute_decisions": plan["attribute_decisions"],
            "unresolved_items": unresolved, "unassessed_spans": unassessed}
