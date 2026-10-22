"""Small, standard-library contracts independent of PRIMA and its dependencies."""

import hashlib
import json
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def draft_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def request_hash(request):
    content = json.dumps(request, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return draft_hash(content)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _unique(rows, key):
    values = [row[key] for row in rows]
    require(all(isinstance(value, str) and value for value in values), f"Invalid {key}")
    require(len(values) == len(set(values)), f"Duplicate {key}")
    return set(values)


def check_span(text, span):
    require(isinstance(span, dict), "Expected a source span")
    start, end = span["start"], span["end"]
    require(type(start) is int and type(end) is int, "Offsets must be integers")
    require(0 <= start < end <= len(text), "Source span out of bounds")
    require(text[start:end] == span["text"], "Source span does not match the draft")


def validate_request(request):
    require(request["schema_version"] == 1, "Unsupported request schema")
    require(isinstance(request["case_id"], str) and request["case_id"], "Missing case ID")
    require(request["purpose"] in {"contract_fixture", "development_mri"}, "Unknown purpose")
    text = request["draft_text"]
    require(isinstance(text, str) and text.strip(), "Empty draft")
    require(draft_hash(text) == request["draft_sha256"], "Draft hash mismatch")
    sequences = request["available_sequences"]
    require(isinstance(sequences, list) and all(isinstance(s, str) and s for s in sequences),
            "Invalid sequence inventory")
    require(len(sequences) == len(set(sequences)), "Duplicate sequence")
    claims, slots, evidence = request["claims"], request["slots"], request["evidence"]
    require(claims and slots, "Declare at least one claim and attribute slot")
    claim_ids = _unique(claims, "claim_id")
    slot_ids = _unique(slots, "slot_id")
    _unique(evidence, "evidence_id")
    for item in evidence:
        require(item["sequence"] in sequences, "Evidence sequence is absent from inventory")
        require(item["kind"] in {"contract_fixture", "native_slice"}, "Unknown evidence kind")
        if item["kind"] == "contract_fixture":
            require(request["purpose"] == "contract_fixture", "Fixture evidence in MRI request")
        else:
            require(isinstance(item.get("path"), str) and item["path"], "Missing image path")
            require(len(item.get("sha256", "")) == 64, "Missing image hash")
            require(item.get("orientation_verified") is True, "Unverified image orientation")
    for claim in claims:
        check_span(text, claim["source_span"])
        require(isinstance(claim["entity_id"], str) and claim["entity_id"], "Missing entity ID")
        require(claim["sequence"] is None or isinstance(claim["sequence"], str), "Invalid sequence")
        required = claim["required_slots"]
        require(required and len(required) == len(set(required)), "Invalid required slots")
        require(set(required) <= slot_ids, "Unknown required slot")
    for slot in slots:
        require(slot["attribute"] in {"signal", "side", "anatomy", "presence", "sequence",
                                      "distribution", "uncertainty"}, "Unknown attribute")
        require(isinstance(slot["asserted_value"], str) and slot["asserted_value"], "Empty assertion")
        require(type(slot["editable"]) is bool, "editable must be boolean")
        if slot["source_span"] is not None:
            check_span(text, slot["source_span"])
        if slot["editable"]:
            geometric_side = (slot["attribute"] == "side" and
                request.get("preparation", {}).get("task") == "s4_mask_grid_side_v1" and
                set(slot["allowed_replacements"]) == {"left", "right", "bilateral"})
            require(slot["attribute"] == "signal" or geometric_side,
                    "Only signal or explicitly opted-in source-mask grid-side edits are enabled")
            require(slot["source_span"] is not None, "Editable slot has no source span")
            require(slot["source_span"]["text"] == slot["asserted_value"], "Edit value/span mismatch")
        replacements = slot["allowed_replacements"]
        require(isinstance(replacements, list)
                and all(isinstance(value, str) and value.strip() for value in replacements),
                "Invalid allowed replacements")
        linked = {c["claim_id"] for c in claims if slot["slot_id"] in c["required_slots"]}
        require(linked, "Unused attribute slot")
        for claim in claims:
            if claim["claim_id"] in linked and slot["source_span"] is not None:
                parent, child = claim["source_span"], slot["source_span"]
                require(parent["start"] <= child["start"] < child["end"] <= parent["end"],
                        "Attribute span lies outside a linked claim")
    return request


def review_template(request, reviewer_kind="human", reviewer_id="pending"):
    validate_request(request)
    require(reviewer_kind in {"human", "fixture", "model", "text_only", "automatic"}, "Unknown reviewer kind")
    return {
        "schema_version": 1,
        "case_id": request["case_id"],
        "request_sha256": request_hash(request),
        "reviewer": {"kind": reviewer_kind, "id": reviewer_id},
        "claims": [
            {
                "claim_id": claim["claim_id"],
                "entity_status": "unresolved",
                "attributes": [
                    {"slot_id": slot_id, "assessability": "insufficient",
                     "verdict": "unresolved", "replacement": None,
                     "replacement_supported": False, "evidence_ids": [],
                     "reason": "Review pending."}
                    for slot_id in claim["required_slots"]
                ],
            }
            for claim in request["claims"]
        ],
    }


def validate_review(request, review):
    require(review["schema_version"] == 1, "Unsupported review schema")
    require(review["case_id"] == request["case_id"], "Review case mismatch")
    require(review["request_sha256"] == request_hash(request), "Stale review: input changed")
    kind = review["reviewer"]["kind"]
    require(kind in {"human", "fixture", "model", "text_only", "automatic"}, "Unknown reviewer kind")
    if kind == "fixture":
        require(request["purpose"] == "contract_fixture", "Fixture review cannot label real MRI")
    claims = {c["claim_id"]: c for c in request["claims"]}
    evidence = {e["evidence_id"] for e in request["evidence"]}
    require(_unique(review["claims"], "claim_id") <= set(claims), "Unknown reviewed claim")
    for row in review["claims"]:
        require(row["entity_status"] in {"identified", "absent", "unresolved"}, "Invalid entity status")
        require(_unique(row["attributes"], "slot_id") <= set(claims[row["claim_id"]]["required_slots"]),
                "Unknown reviewed slot")
        for item in row["attributes"]:
            if kind == "automatic":
                require(item["verdict"] == "unresolved" and not item["replacement_supported"],
                        "Automatic fallback cannot assert a factual verdict")
            require(item["assessability"] in {"sufficient", "insufficient", "not_applicable"},
                    "Invalid assessability")
            require(item["verdict"] in {"supported", "contradicted", "unresolved"}, "Invalid verdict")
            require(item["assessability"] == "sufficient" or item["verdict"] == "unresolved",
                    "Insufficient evidence cannot produce a factual verdict")
            require(isinstance(item["reason"], str) and item["reason"].strip(), "Missing reason")
            ids = item["evidence_ids"]
            require(isinstance(ids, list) and len(ids) == len(set(ids)) and set(ids) <= evidence,
                    "Unknown or duplicate evidence ID")
            require(type(item["replacement_supported"]) is bool, "Invalid replacement support")
            value = item["replacement"]
            require(value is None or (isinstance(value, str) and value.strip()), "Invalid replacement")
            require(not item["replacement_supported"] or
                    (item["verdict"] == "contradicted" and value is not None),
                    "Replacement requires a contradiction and a value")
            require(item["verdict"] == "contradicted" or value is None, "Unexpected replacement")
    return review


def image_path(item, evidence_root):
    root = Path(evidence_root).resolve()
    path = (root / item["path"]).resolve()
    require(path.is_relative_to(root), "Evidence path leaves its case folder")
    require(path.is_file(), f"Evidence file missing: {item['path']}")
    require(hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"],
            f"Evidence image changed: {item['evidence_id']}")
    return path
