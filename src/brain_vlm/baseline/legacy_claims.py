"""Explicit T2 signal text, weak label provenance and finite exact edit slots.

Training labels use only authored global Findings. Qualified descriptions can
label direction weakly; changing their degree/heterogeneity is never authorized.
The existing MedGemma parser remains unchanged.
"""
import re
from copy import deepcopy

from brain_vlm.refinement.claim_extraction import PATTERN, SEQ, sequence_name
from brain_vlm.refinement.contracts import draft_hash, validate_request
from brain_vlm.baseline.schema import LEGACY_TASK as VERSION, SIGNALS

ALIASES = {"low": "low", "hypointense": "low", "high": "high", "hyperintense": "high",
           "intermediate": "intermediate", "isointense": "intermediate"}
AMBIGUOUS = re.compile(r"\b(?:mixed|heterogene\w*|uneven|possible|possibly|suspected|may|might|without|not|no|or)\b", re.I)
MULTIPLE = re.compile(r"\b(?:multiple|several|two|three)\s+(?:\w+\s+){0,3}(?:lesions|masses|foci|nodules)\b", re.I)
DEGREE = re.compile(r"\b(?:slightly|mildly|predominantly|mainly|partially)\b", re.I)


def candidates(text):
    if MULTIPLE.search(text):
        return []
    result = []
    for match in PATTERN.finditer(text):
        names = list(dict.fromkeys(sequence_name(s.group()) for s in re.finditer(SEQ, match["before"] or match["after"], re.I)))
        if "t2" not in names or match["value"].lower() not in ALIASES:
            continue
        start = max(text.rfind(",", 0, match.start()), text.rfind(".", 0, match.start()), text.rfind(";", 0, match.start())) + 1
        ends = [text.find(mark, match.end()) for mark in [",", ".", ";"]]
        end = min([p for p in ends if p >= 0] or [len(text)])
        local = text[start:end]
        if AMBIGUOUS.search(local) or re.search(r"\b(?:iso|low|high)\s*(?:to|and)\s*$", text[start:match.start()], re.I):
            continue
        result.append({"label": ALIASES[match["value"].lower()], "sequence": "t2", "coordinated_sequences": names,
            "phrase_span": {"start": match.start(), "end": match.end(), "text": match.group()},
            "value_span": {"start": match.start("value"), "end": match.end("value"), "text": match["value"]},
            "qualified": bool(DEGREE.search(local)), "local_clause": local})
    return result


def entity_issues(text):
    issues = []
    if MULTIPLE.search(text):
        issues.append("Explicit multiple-finding wording")
    if re.search(r"\b(?:additionally|additional|another|second|separate)\b[^.;]{0,100}\b(?:lesion|mass|nodule|focus|foci)\b", text, re.I):
        issues.append("An additional finding is described; one study-level T2label cannot be assigned to the union")
    return issues


def weak_label(text, *, strict_entity=False):
    if strict_entity and entity_issues(text):
        return None
    found = candidates(text)
    return found[0] if len(found) == 1 else None


def request(study_id, text, evidence, *, available_sequences=None, strict_entity=False):
    candidate = weak_label(text, strict_entity=strict_entity)
    if candidate:
        value = candidate["value_span"]["text"]
        vocabulary = ["hypointense", "isointense", "hyperintense"] if value.lower().endswith("intense") else list(SIGNALS)
        if value.isupper():
            vocabulary = [v.upper() for v in vocabulary]
        elif value[0].isupper():
            vocabulary = [v.capitalize() for v in vocabulary]
        editable = not candidate["qualified"] and candidate["coordinated_sequences"] == ["t2"]
        claims = [{"claim_id": "t2_direction", "entity_id": "primary_finding_hypothesis", "sequence": "t2",
                   "source_span": candidate["phrase_span"], "required_slots": ["t2_signal"]}]
        slots = [{"slot_id": "t2_signal", "attribute": "signal", "asserted_value": value,
                  "source_span": candidate["value_span"], "editable": editable, "allowed_replacements": vocabulary}]
    else:
        claims = [{"claim_id": "unmapped_draft", "entity_id": "global", "sequence": None,
            "source_span": {"start": 0, "end": len(text), "text": text}, "required_slots": ["scope"]}]
        slots = [{"slot_id": "scope", "attribute": "uncertainty", "asserted_value": "outside_supported_parser_scope",
                  "source_span": None, "editable": False, "allowed_replacements": []}]
    return validate_request({"schema_version": 1, "case_id": study_id, "purpose": "development_mri",
        "draft_text": text, "draft_sha256": draft_hash(text), "available_sequences": available_sequences or ["t1", "t2", "flair"],
        "claims": claims, "slots": slots, "evidence": deepcopy(evidence),
        "preparation": {"task": VERSION, "mapped": candidate is not None, "candidate": candidate,
            "entity_ambiguity": entity_issues(text) if strict_entity else [],
            "identity": "Single primary-finding hypothesis; clinical component identity not independently established",
            "reference_scope": "Weak authored-report T2 direction; not pixel-adjudicated clinical truth",
            "clinical_correctness": "not_evaluated"}})
