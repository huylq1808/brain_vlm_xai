"""Scope-aware finite T2 mapping: text assertions, never image truth.

Inventory every T2 mention before selecting a whole-finding assertion. Subregions
remain separate, even when their values happen to agree with the selected value.
This module preserves exact source spans and the existing structured editor.
"""
from copy import deepcopy
import re

from brain_vlm.baseline.legacy_claims import ALIASES, entity_issues
from brain_vlm.refinement.claim_extraction import sentence_spans, sequence_name
from brain_vlm.refinement.contracts import draft_hash, validate_request

PREVIOUS_VERSION = "scope_aware_t2_v3"
RANGE_VERSION = "scope_aware_t2_v4"
VERSION = "scope_aware_t2_v5"
NATURAL_VERSION = "scope_aware_t2_v6"
SEQUENCE = r"(?:T1C|T1(?:WI?|[- ]weighted)?|T2(?:WI?|[- ]weighted)?|FLAIR)"
ATOM = rf"{SEQUENCE}(?:\s+(?:imaging|images?|sequences?))?(?:\s*\({SEQUENCE}\))?"
SEQUENCE_LIST = rf"{ATOM}(?:\s*(?:,\s*(?:and\s+)?|and\s+|/\s*|&\s*){ATOM})*"
VALUE = r"(?:hypointense|hyperintense|isointense|intermediate|low|high|mixed|heterogeneous)"
POST = re.compile(rf"\b(?P<value>{VALUE})\s+(?:signals?(?:\s+intensit(?:y|ies))?\s+)?(?:on|in)\s+(?P<sequences>{SEQUENCE_LIST})(?![\w-])", re.I)
PRE = re.compile(rf"\b(?P<value>{VALUE})\s+(?P<sequences>{SEQUENCE_LIST})\s+signals?\b", re.I)
ANCHOR = re.compile(r"\bT2(?:WI?|[- ]weighted)?\b", re.I)
DEGREE = re.compile(r"\b(slightly|mildly|predominantly|mainly|partially)\s+$", re.I)
UNCERTAIN = re.compile(r"\b(?:no|not|neither|without|possible|possibly|suspected|may|might|mixed|heterogene\w*)\b", re.I)
MULTIPLE_FINDINGS = re.compile(r"\b(?:multiple|several|numerous|two|three)\s+(?:\w+\s+){0,5}(?:lesions|masses|nodules|foci|shadows|patches)\b",re.I)
COMPOUND_V3 = re.compile(r"\b(?:iso|low|high)[- ]*\s*(?:to|and|or)\s*$", re.I)
COMPOUND = re.compile(
    r"\b(?:iso(?:intense)?|hypo(?:intense)?|hyper(?:intense)?|low|high|intermediate)"
    r"\s*(?:[-–—]?\s*(?:to|and|or)[-–—]?|[/–—-])\s*(?:(?:slightly|mildly)\s+)?$", re.I)
COMPONENTS = [
    ("central_subregion", r"\b(?:cent(?:er|re)|central|inner)\b"),
    ("cystic_component", r"\b(?:cystic|necrotic|necrosis)\b"),
    ("solid_component", r"\bsolid\s+(?:portion|component|part|area)\b"),
    ("peripheral_subregion", r"\b(?:peripher\w*|rim)\b"),
    ("perilesional", r"\b(?:peritumor\w*|perilesional|edema|oedema)\b"),
]

# A sequence inventory is not itself a categorical T2 signal assertion. This
# opt-in rule is deliberately narrow: signal-bearing bodies stay uncovered.
SEQUENCE_HEADER = re.compile(
    rf"^\s*(?:(?:sparse|axial|coronal|sagittal)\s+)*(?P<sequences>{SEQUENCE_LIST})"
    r"\s+(?:MRI\s+)?(?:images?|sequences?|scans?)\s+(?:show|shows|demonstrate|demonstrates|reveal|reveals)\b", re.I)
SIGNAL_BODY = re.compile(r"\b(?:signal\w*|intensit\w*|hypo\w*|hyper\w*|isointens\w*|low|high|intermediate|bright\w*|dark\w*|increased|decreased)\b", re.I)


def sequence_headers(text, sentences):
    headers = []
    for sentence in sentences:
        fragment = text[sentence['start']:sentence['end']]
        match = SEQUENCE_HEADER.match(fragment)
        if match and not SIGNAL_BODY.search(fragment[match.end():]):
            headers.append(_span(text, sentence['start'], sentence['start']+match.end()))
    return headers


def _span(text, start, end):
    return {"start": start, "end": end, "text": text[start:end]}


def analyze(text, *, version=VERSION):
    """Finite positive assertions plus uncovered mentions; unsupported stays unknown."""
    if version not in {PREVIOUS_VERSION, RANGE_VERSION, VERSION, NATURAL_VERSION}:
        raise ValueError("Unknown claim-scope version")
    compound = COMPOUND_V3 if version == PREVIOUS_VERSION else COMPOUND
    sentences = sentence_spans(text)
    matches = sorted([*POST.finditer(text), *PRE.finditer(text)], key=lambda m: (m.start(), -m.end()))
    records = []; covered = []
    for m in matches:
        if any(m.start() < end and m.end() > start for start, end in covered):
            continue
        covered.append((m.start(), m.end()))
        atoms = list(re.finditer(ATOM, m["sequences"], re.I))
        sequences = list(dict.fromkeys(sequence_name(re.match(SEQUENCE, a.group(), re.I).group()) for a in atoms))
        if "t2" not in sequences:
            continue
        sentence = next(s for s in sentences if s["start"] <= m.start() < s["end"])
        # Component subjects apply across comma-separated sequence descriptions.
        subject = text[sentence["start"]:m.start()]
        component = next((name for name, pattern in COMPONENTS if re.search(pattern, subject, re.I)), "whole_finding")
        clause_start = max(text.rfind(mark, sentence["start"], m.start()) for mark in [",", ";"]) + 1
        clause_start = max(clause_start, sentence["start"])
        modifier = DEGREE.search(text[clause_start:m.start("value")])
        start = clause_start + modifier.start() if modifier else m.start()
        issues = []
        if m["value"].lower() not in ALIASES or UNCERTAIN.search(text[clause_start:m.end()]):
            issues.append("Uncertain/heterogeneous direction")
        if compound.search(text[clause_start:m.start("value")]):
            issues.append("Compound direction is not one categorical value")
        if modifier and modifier[1].lower() not in {"slightly", "mildly"}:
            issues.append("Distribution/partial modifier needs a richer attribute model")
        records.append({"direction": ALIASES.get(m["value"].lower()), "component_scope": component,
            "source_span": _span(text, start, m.end()), "value_span": _span(text, m.start("value"), m.end("value")),
            "sentence_span": sentence, "sequences": sequences, "modifier": modifier[1] if modifier else None,
            "sequence_spans": [_span(text, m.start("sequences")+a.start(), m.start("sequences")+a.end()) for a in atoms],
            "pattern": "post" if m.re is POST else "pre", "issues": issues,
            "reference_tissue": "not_explicitly_identified", "image_identity": "not_adjudicated"})
    headers = sequence_headers(text, sentences) if version == NATURAL_VERSION else []
    uncovered = [_span(text, a.start(), a.end()) for a in ANCHOR.finditer(text)
                 if not any(start <= a.start() < end for start,end in covered)
                 and not any(h['start'] <= a.start() < h['end'] for h in headers)]
    whole = [r for r in records if r["component_scope"] == "whole_finding"]
    issues = entity_issues(text)
    if MULTIPLE_FINDINGS.search(text): issues.append("Explicit multiple finding/group wording needs entity-level matching")
    if uncovered: issues.append("Unparsed T2 mentions may describe additional assertions")
    if len(whole) != 1: issues.append("Need one explicit whole-finding T2 assertion")
    if len(whole) == 1: issues.extend(whole[0]["issues"])
    selected = whole[0] if not issues else None
    result = {"version": version, "mentions": records, "unparsed_t2_mentions": uncovered,
            "selected": selected, "issues": list(dict.fromkeys(issues)),
            "label_provenance": "Authored report text only; component/tissue identity not image-adjudicated"}
    if version == NATURAL_VERSION:
        result['sequence_headers'] = headers
        result['label_provenance'] = 'Draft text assertions only; component/tissue identity not image-adjudicated'
    return result


def renderings(text, record):
    """Finite sequence-preserving edits, including parenthetical sequence aliases."""
    span = record["source_span"]; value = record["value_span"]
    fragment = span["text"]
    values = ["hypointense", "isointense", "hyperintense"] if value["text"].lower().endswith("intense") else ["low", "intermediate", "high"]
    if value["text"].isupper(): values = [v.upper() for v in values]
    elif value["text"][0].isupper(): values = [v.capitalize() for v in values]
    result = {}
    for word in values:
        direction = ALIASES[word.lower()]
        if record["modifier"] and direction == "intermediate":
            result[direction] = None; continue
        a, b = value["start"]-span["start"], value["end"]-span["start"]
        if len(record["sequences"]) == 1:
            rendered = fragment[:a]+word+fragment[b:]
        elif record["pattern"] == "post":
            atoms = record["sequence_spans"]
            prefix = text[span["start"]:atoms[0]["start"]]
            changed = prefix[:a]+word+prefix[b:]
            rendered = "; ".join((changed if sequence_name(re.match(SEQUENCE,s["text"],re.I).group())=="t2" else prefix)+s["text"] for s in atoms)
        else:
            # No arbitrary rewrite of coordinated prenominal predicates.
            result[direction] = None; continue
        result[direction] = rendered
    result[record["direction"]] = fragment
    return result


def request(study_id, text, evidence, *, available_sequences=None, strict_entity=True, claim_protocol=VERSION):
    mapping = analyze(text, version=claim_protocol); chosen = mapping["selected"]
    if chosen:
        options = renderings(text, chosen)
        claims = [{"claim_id":"t2_direction","entity_id":"primary_finding_hypothesis","sequence":"t2",
                   "source_span":chosen["source_span"],"required_slots":["t2_signal"]}]
        slots = [{"slot_id":"t2_signal","attribute":"signal","asserted_value":chosen["source_span"]["text"],
                  "source_span":chosen["source_span"],"editable":True,
                  "allowed_replacements":list(dict.fromkeys(v for v in options.values() if v is not None))}]
        if chosen["modifier"]:
            claims[0]["required_slots"].append("t2_degree_unassessed")
            slots.append({"slot_id":"t2_degree_unassessed","attribute":"uncertainty","asserted_value":chosen["modifier"],
                          "source_span":None,"editable":False,"allowed_replacements":[]})
        for sequence in chosen['sequences']:
            if sequence=='t2':continue
            sid=sequence+'_source_context'
            claims.append({'claim_id':sid,'entity_id':'primary_finding_hypothesis','sequence':sequence,
                           'source_span':chosen['source_span'],'required_slots':[sid]})
            slots.append({'slot_id':sid,'attribute':'signal','asserted_value':chosen['value_span']['text'],
                          'source_span':None,'editable':False,'allowed_replacements':[]})
        semantics = {"asserted_direction":chosen["direction"],"renderings":options,
            "modifier_retained_unassessed":chosen["modifier"],"other_sequence_values_preserved":{s:chosen['value_span']['text'] for s in chosen['sequences'] if s!='t2'},
            "policy":"Whole-finding direction proxy only; subregions/reference tissue remain unassessed"}
    else:
        claims = [{"claim_id":"unmapped_draft","entity_id":"global","sequence":None,
                   "source_span":_span(text,0,len(text)),"required_slots":["scope"]}]
        slots = [{"slot_id":"scope","attribute":"uncertainty","asserted_value":"outside_supported_parser_scope",
                  "source_span":None,"editable":False,"allowed_replacements":[]}]
        semantics = None
    # Extra T2components are visible protected claims, never silently collapsed.
    for i, record in enumerate(mapping["mentions"]):
        if record is chosen: continue
        sid = f"component_{i+1:02d}_unassessed"
        claims.append({"claim_id":sid,"entity_id":sid,"sequence":"t2","source_span":record["source_span"],"required_slots":[sid]})
        slots.append({"slot_id":sid,"attribute":"uncertainty","asserted_value":record["component_scope"],
                      "source_span":None,"editable":False,"allowed_replacements":[]})
    prep = {"task":"weak_reference_t2_3d_v1","claim_protocol":claim_protocol,"mapped":chosen is not None,
            "scope_inventory":mapping,"entity_ambiguity":mapping["issues"],
            "target_scope":"whole_finding" if chosen else "unresolved",
            "observer_scope":"image-predicted source-core proxy; clinical correspondence not adjudicated",
            "reference_scope":"Report reference unknown; estimated image reference is a model assumption",
            "clinical_correctness":"not_evaluated"}
    if claim_protocol in {VERSION, NATURAL_VERSION}:
        # Scope, not the asserted direction, determines this permission. A draft
        # can contain an incorrect subregion claim; do not treat its value as truth.
        intra_core = [r for r in mapping['mentions'] if r['component_scope'] in
                      {'central_subregion','cystic_component','solid_component','peripheral_subregion'}]
        prep['observer_scope_alignment'] = {
            'status':'unresolved' if chosen and intra_core else 'bounded_proxy',
            'unmatched_intra_core_scopes':list(dict.fromkeys(r['component_scope'] for r in intra_core)),
            'reason':'A single whole-core categorical observation does not resolve explicitly described intra-core subregions; global direction needs component-aware observation',
            'clinical_correspondence':'not_adjudicated'}
    if semantics: prep["signal_semantics"] = semantics
    return validate_request({"schema_version":1,"case_id":study_id,"purpose":"development_mri",
        "draft_text":text,"draft_sha256":draft_hash(text),"available_sequences":available_sequences or ["t1","t1c","t2","flair"],
        "claims":claims,"slots":slots,"evidence":deepcopy(evidence),"preparation":prep})
