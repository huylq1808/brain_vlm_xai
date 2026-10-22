"""Optional M2: the previously working bounded MedGemma draft, without diagnostics.

Only native images and sequence labels enter the model request. No reference
report, controlled assertion, mask, entity description or PRIMA embedding.
"""
from copy import deepcopy
from pathlib import Path
from brain_vlm.common import read, write, sha, require, fingerprint
from brain_vlm.refinement.images import render_native_slices
from brain_vlm.refinement.observations import parse_model_response
from brain_vlm.refinement.local_vlm import JSON_SYSTEM_INSTRUCTION

POLICY = {"response_format":"bounded_findings_json_v1", "max_new_tokens":768,
          "max_characters":480, "max_words":180, "no_repeat_ngram_size":8, "repetition_penalty":1.05}
PROMPT = (
    "Return JSON only with one field, findings, containing a brief English brain MRI Findings paragraph. "
    "Write 2 or 3 short sentences, at most 480 characters including spaces. "
    "The images are sparse axial samples from the listed sequences of ONE study, not a complete volume. "
    "Describe only intracranial findings visible in these views, with sequence attribution and uncertainty. "
    "Do not produce an organ-by-organ checklist or discuss anatomy outside this brain examination. "
    "Do not invent history, measurements or diagnoses. Do not infer whole-study absence from unseen anatomy. "
    "Do not repeat sentences. If cross-sequence finding identity is uncertain, say so briefly. "
    "Mention only supplied sequences; do not assert diffusion or enhancement without the required images. "
    "End the paragraph after the relevant findings; no headings, lists, advice or commentary. "
    'Required format: {"findings":"Your image-based Findings text."}'
)


def prepare(studies, output, max_images=16, *, view_plan=None):
    import nibabel as nib
    jobs = []
    for study in studies:
        folder = Path(output)/study["study_id"]
        folder.mkdir(parents=True,exist_ok=True)
        rows = []
        selected = view_plan[study['study_id']] if view_plan is not None else None
        geometry = None
        for seq, path in study["images"].items():
            image = nib.as_closest_canonical(nib.load(path))
            require(len(image.shape)==3,"M2 expects 3D native MRI")
            if selected is not None:
                import numpy as np
                if geometry is None: geometry = (image.shape, image.affine)
                require(image.shape == geometry[0] and np.allclose(image.affine, geometry[1], atol=1e-4, rtol=0),
                        'Proposal views require aligned sequence grids')
            indices = selected['native_canonical_z'] if selected is not None else sorted({round((image.shape[2]-1)*f) for f in [.25,.5,.75]})
            for row in render_native_slices({seq:path},folder/"inputs"/seq/"images",indices):
                row["path"] = "inputs/"+seq+"/"+row["path"]
                row["header_world_z"],row["header_world_units"] = row.pop("world_z_mm"),image.header.get_xyzt_units()[0]
                row.update(evidence_id=f"image_{len(rows)+1:02d}",role="proposal_context" if selected is not None else "study_context",native_canonical_z=row["canonical_ras_z"])
                rows.append(row)
        require(0 < len(rows) <= max_images,"M2 views exceed max_images; select sequences explicitly")
        job = {"study_id":study["study_id"],"images":rows,"prompt":PROMPT,"reference_used":False,
               "selection":"fixed native canonical axial fractions .25/.50/.75 per sequence; no masks/reports",
               "source_sha256":{s:sha(p) for s,p in study["images"].items()}}
        if selected is not None:
            job['selection'] = 'Three separated axial support peaks of an image-only predicted core proposal; no report/mask; full-plane views, not whole-study coverage'
            job['view_proposal'] = selected
        write(folder/"request.json",job)
        jobs.append(job)
    return jobs


def generate_drafts(studies, output, jobs, generator, model_config, mode="both"):
    """Persist raw per call, then hand the exact Findings field to M3–M9."""
    import jsonschema
    require(mode in {"generated","both"},"Unknown generation handoff")
    require(len(studies)==len(jobs),"M2 study/job count mismatch")
    schema = {"type":"object","properties":{"findings":{"type":"string","minLength":1,"maxLength":POLICY["max_characters"]}},
              "required":["findings"],"additionalProperties":False}
    effective, failed = [], []
    for study, job in zip(studies,jobs):
        require(study["study_id"] == job["study_id"],"M2 job order mismatch")
        folder = Path(output)/job["study_id"]
        record = {"study_id":job["study_id"],"status":"failed","reference_used":False,"clinical_correctness":"not_evaluated",
                  "model_config":model_config,"generation_policy":POLICY,"response_schema":schema,
                  "request_fingerprint":fingerprint(job)}
        try:
            images = []
            for row in job["images"]:
                path = (folder/row["path"]).resolve()
                require(path.is_relative_to(folder.resolve()) and sha(path)==row["sha256"],"M2 evidence changed")
                images.append({**row,"resolved_path":str(path)})
            generator.configure_generation(no_repeat_ngram_size=POLICY["no_repeat_ngram_size"],repetition_penalty=POLICY["repetition_penalty"])
            generator.configure_response(constrained_decoding=True,system_instruction=JSON_SYSTEM_INSTRUCTION)
            generator.set_response_schema(schema)
            raw = generator(job["prompt"],images)
            record["raw_response"] = raw
            parsed, normalization = parse_model_response(raw)
            record.update(parsed_response=parsed,normalization=normalization)
            jsonschema.validate(parsed,schema)
            text = parsed["findings"].strip()
            metadata = deepcopy(generator.metadata)
            issues = []
            if not text:issues.append("empty_findings")
            if metadata.get("output_hit_token_cap"):issues.append("token_cap_reached")
            if metadata.get("last_stop_token_id") is None:issues.append("no_observed_stop_token")
            if len(text.split()) > POLICY["max_words"]:issues.append("word_budget_exceeded")
            record["validation"] = {"accepted_for_handoff":not issues,"issues":issues,
                "interpretation":"Format/completion only; not clinical correctness."}
            require(not issues,"Generation rejected: "+", ".join(issues))
            (folder/"draft.txt").write_text(text+"\n",encoding="utf-8")
            record.update(status="completed",draft_sha256=sha(folder/"draft.txt"))
            current = deepcopy(study)
            current["drafts"] = {} if mode=="generated" else current["drafts"]
            current["drafts"]["generated"] = text
            current["draft_provenance"] = {} if mode=="generated" else current.get("draft_provenance",{})
            current["draft_provenance"]["generated"] = {"reference_used":False,"origin":"image-only MedGemma M2",
                "generation_record":str(folder/"generation.json"),"generation_request_fingerprint":record["request_fingerprint"]}
            current["draft_origin"] = "Per-variant origins in draft_provenance; M2 uses images only"
            effective.append(current)
        except Exception as error:
            record["error"] = type(error).__name__+": "+str(error)
            failed.append(str(folder/"generation.json"))
        finally:
            record["generator_metadata"] = deepcopy(getattr(generator,"metadata",{}))
            write(folder/"generation.json",record)
    require(not failed,"M2 failed; raw/error records: "+", ".join(failed))
    return {"studies":effective,"new_model_calls":len(jobs),"reference_used":False,"traces":[],
            "note":"Generated text is a draft, not an image-truth label."}
