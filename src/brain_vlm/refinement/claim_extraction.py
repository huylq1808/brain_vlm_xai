"""Small English signal parser with exact offsets; unsupported text stays unassessed.

This extracts assertions, not image truth or finding identity. A location from
the draft can be a marked search hypothesis. Multiple findings remain ambiguous.
"""
import re
from copy import deepcopy

from brain_vlm.refinement.contracts import draft_hash, require, validate_request

VERSION = "bounded_signal_parser_v2"
CLAIM_VERSION = "bounded_signal_and_finding_anchor_v4"
REGIONAL_ENTITY = "region_periventricular_white_matter"
REGIONAL_ANCHOR = {
    "description": "Inspect the periventricular white matter for focal or confluent signal abnormalities in the supplied FLAIR views. The search instruction does not imply an abnormality exists.",
    "signal_reference": "surrounding normal-appearing white matter; do not infer whole-volume absence from sampled images"}
SEQ = r"(?:T1C|T1(?:WI?|[- ]weighted)?|T2(?:WI?|[- ]weighted)?|FLAIR)"
VALUE = r"(?:hypointense|hyperintense|isointense|intermediate|low|high|mixed)"
PATTERN = re.compile(
    rf"\b(?P<value>{VALUE})\s+(?:"
    rf"(?P<before>{SEQ})\s+signal|"
    rf"(?:signal\s+)?on\s+(?P<after>{SEQ}(?:\s*(?:and|,|/)\s*{SEQ})*)"
    rf"(?:\s+(?:sequences?|images?))?)\b", re.IGNORECASE)
QUALIFIER = re.compile(r"\b(no|not|without|possible|possibly|suspected|may|might|"
                       r"if|uncertain|equivocal|slightly|partial|partially|mixed|heterogeneous|heterogeneously)\b", re.I)

LOCATION = re.compile(r"\b(?P<side>left|right|bilateral)\s+(?P<anatomy>frontal|temporal|parietal|occipital|frontoparietal)\s+lob(?:e|es)\b",re.I)
MULTIPLE = re.compile(r"\b(?:multiple|several|two|three|numerous)\s+(?:\w+\s+){0,3}(?:foci|lesions|masses|nodules)\b",re.I)


def finding_anchor(draft, entity_id):
    """One bounded finding/group. Location is a draft search hypothesis, not truth."""
    locations = [{"side":m["side"].lower(),"anatomy":m["anatomy"].lower()+" lobe",
                  "source_span":{"start":m.start(),"end":m.end(),"text":m.group()}}
                 for m in LOCATION.finditer(draft)]
    distinct = {(v["side"],v["anatomy"]) for v in locations}
    ambiguous = len(distinct)>1 or bool(MULTIPLE.search(draft))
    hypothesis = " ".join(next(iter(distinct))) if len(distinct)==1 and not ambiguous else None
    return {"finding_id":entity_id,"location_hypothesis":hypothesis,"location_spans":locations,
            "mapping_status":"ambiguous" if ambiguous else "single_finding_hypothesis",
            "image_identity":"not_verified","location_role":"draft-derived search hypothesis only"}


def search_entities(study):
    """Share a stable neutral search across variants; never include signal values."""
    entities = deepcopy(study["entities"])
    for entity,anchor in entities.items():
        records = [finding_anchor(text,entity) for text in study["drafts"].values()]
        locations = sorted({r["location_hypothesis"] for r in records if r["location_hypothesis"]})
        ambiguous = any(r["mapping_status"]=="ambiguous" for r in records) or len(locations)>1
        anchor["location_hypothesis"] = locations[0] if len(locations)==1 and not ambiguous else None
        anchor["finding_mapping"] = "ambiguous" if ambiguous else "single_finding_hypothesis"
        anchor["finding_id"] = entity
        if anchor["location_hypothesis"]:
            anchor["description"] += " Search hypothesis: "+anchor["location_hypothesis"]+". This location comes from the draft and may be wrong; inspect wider context and defer if identity is unclear."
    return entities


def sequence_name(text):
    value = text.lower().replace("-", "").replace(" ", "")
    return "t1c" if value == "t1c" else "t1" if value.startswith("t1") else "t2" if value.startswith("t2") else "flair"


def sentence_spans(text):
    """Preserve exact offsets and decimal measurements; not a clinical parser."""
    rows, start = [], 0
    for end in [m.end() for m in re.finditer(r"[.!?](?=\s|$)|\n+", text)] + [len(text)]:
        low, high = start, end
        while low < high and text[low].isspace():low += 1
        while high > low and text[high-1].isspace():high -= 1
        if high > low:rows.append({"start": low, "end": high, "text": text[low:high]})
        start = end
    return rows


def sentence_inventory(request):
    """Expose bounded mapping; inventory is not verification of the other words."""
    slots = {s["slot_id"]: s for s in request["slots"]}
    rows = []
    for span in sentence_spans(request["draft_text"]):
        claims = [c for c in request["claims"] if c["claim_id"] != "unmapped_draft"
                  and max(span["start"], c["source_span"]["start"]) < min(span["end"], c["source_span"]["end"])]
        rows.append({**span, "claim_ids": [c["claim_id"] for c in claims],
                     "mapped_attributes": sorted({slots[s]["attribute"] for c in claims for s in c["required_slots"]}),
                     "mapping": "bounded_attributes_only" if claims else "unmapped"})
    return rows


def extract_signal_request(study_id, draft, sequences, evidence, entity_id):
    """Explicit signal+sequence phrases only; shared wording is never edited.

    Examples: 'low T1W signal', 'hyperintense signal on T2W and FLAIR'.
    Negated/uncertain/qualified or compound signal descriptions are deferred.
    The caller supplies a candidate entity, never an automatic matched identity.
    """
    require(isinstance(entity_id, str) and entity_id and entity_id != "global", "Supply one neutral candidate entity")
    require(isinstance(draft, str) and draft.strip(), "Empty Findings")
    claims, slots, ignored = [], [], []
    for match in PATTERN.finditer(draft):
        # Check this clause, not a subsequent unrelated negative sentence.
        sentence_start = max(draft.rfind(".", 0, match.start()), draft.rfind(";", 0, match.start())) + 1
        prefix = draft[sentence_start:match.start()]
        suffix = draft[match.end():].split(".", 1)[0].split(";", 1)[0]
        value = match.group("value")
        if (QUALIFIER.search(prefix) or value.lower() == "mixed"
                or re.search(rf"\b{VALUE}\s+(?:and|or)\s*$", prefix, re.I)
                or re.match(rf"\s*(?:or\s+{SEQ}|post[- ]?contrast|contrast[- ]enhanced)", suffix, re.I)
                or re.search(r"\b(?:with\s+)?(?:partial|mixed|heterogeneous|some mixed)\b", suffix, re.I)):
            ignored.append({"text": match.group(), "reason": "Qualified, uncertain, negative or heterogeneous signal; outside parser scope."})
            continue
        names = [sequence_name(s.group()) for s in re.finditer(SEQ, match.group("before") or match.group("after"), re.I)]
        names = list(dict.fromkeys(names))
        span = {"start": match.start(), "end": match.end(), "text": match.group()}
        word_span = {"start": match.start("value"), "end": match.end("value"), "text": value}
        values = ["hypointense", "hyperintense", "isointense"] if value.lower().endswith("intense") else ["low", "high", "intermediate"]
        if value.isupper():
            values = [v.upper() for v in values]
        elif value[0].isupper():
            values = [v.capitalize() for v in values]
        for sequence in names:
            key = f"claim_{len(claims) + 1:02d}"
            presence, signal = key + "_presence", key + "_signal"
            slots.extend([
                {"slot_id": presence, "attribute": "presence", "asserted_value": "present", "source_span": None,
                 "editable": False, "allowed_replacements": []},
                {"slot_id": signal, "attribute": "signal", "asserted_value": value, "source_span": word_span,
                 "editable": len(names) == 1, "allowed_replacements": values}])
            claims.append({"claim_id": key, "entity_id": entity_id, "sequence": sequence,
                           "source_span": span, "required_slots": [presence, signal]})
    if not claims:
        # Return a complete flagging transaction even for an unsupported draft.
        claims = [{"claim_id": "unmapped_draft", "entity_id": "global", "sequence": None,
                   "source_span": {"start": 0, "end": len(draft), "text": draft}, "required_slots": ["scope"]}]
        slots = [{"slot_id": "scope", "attribute": "uncertainty", "asserted_value": "outside_supported_parser_scope",
                  "source_span": None, "editable": False, "allowed_replacements": []}]
    return validate_request({"schema_version": 1, "case_id": study_id, "purpose": "development_mri",
                             "draft_text": draft, "draft_sha256": draft_hash(draft),
                             "available_sequences": list(sequences), "claims": claims, "slots": slots,
                             "evidence": deepcopy(evidence), "preparation": {
                                 "claim_mapping": VERSION, "candidate_entity": entity_id,
                                 "identity": "candidate_only; requires independent linkage review",
                                 "parser_ignored": ignored, "image_truth": "not_reviewed",
                                 "scope": "Positive explicit categorical signal phrases; all other text unassessed."}})


def extract_claim_request(study_id, draft, sequences, evidence, entity_id):
    """Retain v1 signal parsing and add one explicit, non-editable regional query.

    This is bounded rule-based routing, not a general clinical report extractor.
    'Significant' is retained as an unoperationalized qualifier, never dropped.
    All unmatched wording stays visible as unmapped text.
    """
    request = extract_signal_request(study_id, draft, sequences, evidence, entity_id)
    extra = []
    for sentence in sentence_spans(draft):
        start, end, text = sentence["start"], sentence["end"], sentence["text"]
        match = re.fullmatch(
            r"(?:There (?:is|are) )?No (?:evidence of )?(?P<qualifier>significant )?"
            r"periventricular white[ -]matter (?:signal (?:changes|abnormalities)|hyperintensities)"
            r"(?: (?:are|were) (?:noted|identified|seen|demonstrated))?[.]?", text, re.I)
        if match:
            extra.append({"span": {"start": start, "end": end, "text": text},
                          "qualifiers": [match['qualifier'].strip().lower()] if match['qualifier'] else []})
    if extra:
        if request["claims"][0]["claim_id"] == "unmapped_draft":
            request["claims"], request["slots"] = [], []
        for index, item in enumerate(extra, 1):
            key, slot = f"regional_absence_{index:02d}", f"regional_presence_{index:02d}"
            request["claims"].append({"claim_id": key, "entity_id": REGIONAL_ENTITY, "sequence": "flair",
                "source_span": item["span"], "required_slots": [slot],
                "observation_kind": "regional_presence", "anatomy": "periventricular_white_matter",
                "polarity": "absent", "scope": "region", "qualifiers": item["qualifiers"],
                "asserted_sequence": None,
                "evidence_sequence_policy": "FLAIR regional context route; sequence is not asserted by the source sentence"})
            request["slots"].append({"slot_id": slot, "attribute": "presence", "asserted_value": "absent",
                "source_span": None, "editable": False, "allowed_replacements": []})
    request["preparation"].update(claim_mapping=CLAIM_VERSION,
        scope="Explicit positive categorical signal plus bounded periventricular FLAIR absence; other text unassessed.")
    request["preparation"]["sentence_inventory"] = sentence_inventory(request)
    finding = finding_anchor(draft,entity_id)
    request["findings"] = [finding]
    for claim in request["claims"]:
        if claim["entity_id"]==entity_id:
            claim.update(finding_id=entity_id,finding_mapping=finding["mapping_status"])
    return validate_request(request)


def unmapped_spans(request):
    """Text outside substantive claim spans; placeholders never count as mapped."""
    text, covered = request["draft_text"], [False] * len(request["draft_text"])
    for claim in request["claims"]:
        if claim["claim_id"] == "unmapped_draft":
            continue
        span = claim["source_span"]
        covered[span["start"]:span["end"]] = [True] * (span["end"] - span["start"])
    spans, start = [], None
    for index in range(len(text) + 1):
        gap = index < len(text) and not covered[index]
        if gap and start is None:
            start = index
        if not gap and start is not None:
            if text[start:index].strip():
                spans.append({"start": start, "end": index, "text": text[start:index]})
            start = None
    return spans


def assessment_coverage(request, review):
    """Mapping and effective factual verdict counts; never a clinical accuracy score."""
    claims = [c for c in request["claims"] if c["claim_id"] != "unmapped_draft"]
    reviewed = {c["claim_id"]: c for c in review["claims"]}
    factual, complete = 0, 0
    for claim in claims:
        rows = {a["slot_id"]: a for a in reviewed.get(claim["claim_id"], {}).get("attributes", [])}
        count = sum(rows.get(slot, {}).get("assessability") == "sufficient"
                    and rows[slot].get("verdict") in {"supported", "contradicted"}
                    and bool(rows[slot].get("evidence_ids")) for slot in claim["required_slots"])
        factual += count
        complete += count == len(claim["required_slots"])
    return {"mapped_claims": len(claims), "placeholder_claims": len(request["claims"]) - len(claims),
            "fully_assessed_claims": complete, "factual_attribute_verdicts": factual,
            "unmapped_spans": unmapped_spans(request),
            "interpretation": "Counts report mapping and effective evidence-backed verdicts; not clinical correctness or whole-study image coverage."}
