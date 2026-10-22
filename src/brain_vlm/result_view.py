"""Portable view of saved system artifacts; no models or clinical judgments."""
import base64
import difflib
import html
import json
from pathlib import Path
from brain_vlm.common import read
from brain_vlm.refinement.claim_extraction import sentence_inventory

esc = lambda value:html.escape(str(value))


def table(headers,rows):
    return "<table><tr>"+"".join("<th>"+esc(h)+"</th>" for h in headers)+"</tr>"+"".join(
        "<tr>"+"".join("<td>"+esc(v)+"</td>" for v in row)+"</tr>" for row in rows)+"</table>"


def details(title,value):
    return "<details><summary>"+esc(title)+"</summary><pre>"+esc(json.dumps(value,ensure_ascii=False,indent=2))+"</pre></details>"


def diff_html(before,after):
    """Escape report content before adding display-only red/green markup."""
    left,right = [],[]
    for tag,a,b,c,d in difflib.SequenceMatcher(a=before,b=after,autojunk=False).get_opcodes():
        x,y = esc(before[a:b]),esc(after[c:d])
        left.append(x if tag=="equal" else "<del>"+x+"</del>")
        right.append(y if tag=="equal" else "<ins>"+y+"</ins>")
    return "<div class='columns'><pre>"+"".join(left)+"</pre><pre>"+"".join(right)+"</pre></div>"


def gallery(root,rows):
    parts=[]
    for row in rows:
        uri="data:image/png;base64,"+base64.b64encode((root/row["path"]).read_bytes()).decode()
        caption=" · ".join(str(row.get(k,"")) for k in ["evidence_id","sequence","role"])
        caption+=" · native z="+str(row.get("native_canonical_z",row.get("canonical_ras_z","?")))
        if row.get("paired_context_id"):caption+=" · pair: "+row["paired_context_id"]
        parts.append("<figure><img src='"+uri+"'><figcaption>"+esc(caption)+"</figcaption></figure>")
    return "<div class='gallery'>"+"".join(parts)+"</div>"


def module_class(module,status):
    if any(word in status for word in ["not recorded","not implemented","not used","supplied/imported","no calls"]):return "skipped"
    return "partial" if module in {"M3","M5","M7"} else ""


def present(root):
    root=Path(root)
    summary=read(root/"summary.json")
    parts=["<!doctype html><html lang='vi'><meta charset='utf-8'><title>Brain MRI baseline · system flow</title>",
        "<style>body{font:16px/1.5 system-ui;max-width:1250px;margin:25px auto;padding:16px;background:#f3f6fa;color:#172b42}section{background:white;padding:22px;margin:20px 0;border-radius:10px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f1f4f8;padding:14px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #ccd5df;padding:9px;text-align:left;vertical-align:top}img{width:245px;max-width:100%}figure{margin:8px;max-width:260px}figcaption{font-size:12px;overflow-wrap:anywhere}.gallery,.modules{display:flex;flex-wrap:wrap}.module{padding:9px;margin:4px;background:#e4f3e9;border-radius:5px}.partial{background:#fff0cb}.skipped{background:#e7ebf0}.columns{display:grid;grid-template-columns:1fr 1fr;gap:14px}del{background:#ffdada;text-decoration:line-through}ins{background:#d6f2e0;text-decoration:none;font-weight:bold}.keep{background:#e1f3e9}.correct{background:#e2edff}.flag,.note{background:#fff0cf}.note{padding:12px}details{margin:12px 0}summary{cursor:pointer}nav a{display:inline-block;margin:6px}@media(max-width:650px){.columns{grid-template-columns:1fr}section{padding:12px}}</style>",
        "<h1>Brain MRI VLM · Study → evidence → edit</h1><p>"+esc(summary["mode"])+" · New M2 calls: "+esc(summary.get("new_generation_calls",0))+" · New M7 calls: "+esc(summary["new_model_calls"])+"</p>",
        "<p class='note'>Research output. Module colors show implementation/execution, not medical accuracy. Cached replay is not new inference; scope/uncertainty edits are separate from disease-fact corrections.</p>"]
    parts.append("<nav>"+"".join("<a href='#case"+str(i)+"'>"+esc(e["study"]+" / "+e["variant"])+"</a>" for i,e in enumerate(summary["studies"]))+"</nav>")
    for index,entry in enumerate(summary["studies"]):
        folder=(root/entry["result"]).parent
        result,request=read(folder/"result.json"),read(folder/"request.json")
        native=read(folder/"input_study.json") if (folder/"input_study.json").exists() else {}
        records=read(folder/"observations.json")
        queries=native.get("grounding",[])
        gen=root/"generation"/entry["study"]
        unique_localizations = {q.get("localization_record") or q["query"]["query_id"]:q for q in queries}
        groups=sum(len(q["m10_groups"]) for q in unique_localizations.values())
        learned = any(q.get("manifest", {}).get("method") == "s2_gli_flair_top8_v1" for q in queries)
        m5_status = ("proposed region" if any(q.get("manifest",{}).get("files") for q in queries)
                     else "no positive region" if groups else "not used / unavailable")
        modules=[("M0","native study"),("M1","saved features" if native.get("prima_outputs") else "not recorded"),
            ("M2","generated" if (gen/"generation.json").exists() else "supplied/imported draft"),
            ("M3","bounded attributes"),("M4","not implemented"),("M5",m5_status),
            ("M6",str(len(request["evidence"]))+" PNGs"),("M7","saved observations" if records else "no calls"),
            ("M8","one edit plan" if result.get("edit_plan",{}).get("policy") else "legacy plan"),
            ("M9",str(len(result["applied_edits"]))+" edits"),("M10","not used: learned proposal route" if learned else str(groups)+" PRIMA groups" if groups else "not used / unavailable")]
        parts.append("<section id='case"+str(index)+"'><h2>"+esc(entry["study"]+" / "+entry["variant"])+"</h2><div class='modules'>"+
            "".join("<div class='module "+module_class(m,s)+"'><b>"+m+"</b><br>"+esc(s)+"</div>" for m,s in modules)+"</div>")
        parts.append("<h3>0 · M0/M1 — MRI và representations</h3>"+table(["Sequence","Native NIfTI"],native.get("native_images",{}).items()))
        parts.append(details("Geometry, actual PRIMA shapes, weights and source provenance",native))
        role=native.get("sample_provenance",{}).get("clinical_role")
        if role:parts.append("<p class='note'>"+esc(role)+"</p>")
        parts.append("<h3>1 · Draft / optional M2</h3><p>"+esc(native.get("draft_origin","supplied draft"))+"</p><pre>"+esc(request["draft_text"])+"</pre>")
        if (gen/"generation.json").exists():
            generation=read(gen/"generation.json")
            parts.append(details("M2 raw response, validation and resources (not clinical truth)",generation))
            job=read(gen/"request.json")
            parts.append(details("Exact image-only M2 prompt",job["prompt"])+gallery(gen,job["images"]))
        parts.append("<h3>2 · M3 — Mapped claims / unassessed sentences</h3>"+table(
            ["Sentence","Claim IDs","Mapped attributes","Mapping"],[(s["text"],s["claim_ids"],s["mapped_attributes"],s["mapping"]) for s in sentence_inventory(request)]))
        parts.append(details("Exact claims, slots and source spans",{k:request[k] for k in ["claims","slots"]}))
        if request.get("findings"):parts.append(details("Finding IDs / location hypotheses from draft (not image truth)",request["findings"]))
        parts.append("<h3>3 · M5/M6/M10 — Selection and evidence</h3><p>PRIMA group score is a selection aid, not a crop/voxel score or MedGemma explanation. Full context and paired crop use the same source window in new runs.</p>")
        policies=sorted({e["evidence_sampling_policy"] for e in request["evidence"] if e.get("evidence_sampling_policy")})
        if policies:parts.append("<p>Evidence sampling: "+esc(", ".join(policies))+". Broad full views are retained independently of the proposed ROI; sampled views do not establish whole-study absence.</p>")
        plans = {e["sequence"]: e["support_plane_selection"] for e in request["evidence"] if e.get("support_plane_selection")}
        if plans:
            parts.append(table(["Sequence", "Actual support planes (canonical z)", "Support bands not represented", "Scope"],
                [(seq, plan["canonical_z"], plan["unrepresented_band_count"], plan["interpretation"]) for seq,plan in plans.items()]))
        shown = set()
        for q in queries:
            parts.append(details("Query / proposed region / attribution source",q))
            key = q.get("localization_record") or q["query"]["query_id"]
            if q["m10_groups"] and key not in shown:
                if q.get("manifest", {}).get("method") == "s2_gli_flair_top8_v1":
                    parts.append("<p>Learned GLI/FLAIR proposal. Top8 patch union box; same-grid projection only. Scores are uncalibrated foreground scores, not M10 cosine drops or clinical confidence. Finding identity is unresolved.</p>")
                    parts.append(table(["Proposal", "Native box", "Foreground score", "Scope"],
                        [(g["region_id"],g["native_box_xyz"],g["foreground_score"],g["score_scope"]) for g in q["m10_groups"]]))
                elif q.get("localization_record"):
                    parts.append("<p>Shared spatial search: scores below are computed once and reused for same-grid sequences. Region IDs are proposals, not confirmed lesion IDs.</p>")
                    parts.append(table(["Region / parent","Depth","Native token-support box","Tokens by sequence","Cosine drop"],
                        [(g["region_id"]+" / "+str(g["parent_id"]),g["depth"],g["native_box_xyz"],g["masked_series_token_counts"],g["similarity_drop"]) for g in q["m10_groups"]]))
                else:parts.append(table(["Series index","Group tokens","Cosine baseline","Drop"],
                    [(g["series_index"],g["token_count"],g["baseline"],g["similarity_drop"]) for g in q["m10_groups"]]))
                shown.add(key)
        parts.append(gallery(folder,request["evidence"]))
        parts.append("<h3>4 · M7 — Actual observations</h3>")
        for item in records:
            record=item["record"]
            parts.append("<p>Execution here: "+esc(item["execution_in_this_run"])+" · Original response status: "+esc(record.get("response_status"))+"</p><pre>"+esc(record.get("raw_response") or record.get("execution"))+"</pre>")
            parts.append(details("Actual prompt / schema / forward tensors / original call resources",{k:record.get(k) for k in ["task","prompt","response_schema","generator_metadata"]}))
        if result.get("readiness"):
            parts.append(table(["Sequence","Model abstained","Pipeline deferred","Edit eligible","Causes","Reason image pointers"],
                [(r["sequence"],r.get("model_abstained"),r.get("pipeline_blocked"),r.get("correction_eligible"),r.get("causes",[]),
                  r.get("evidence_reference_audit", "not recorded")) for r in result["readiness"]]))
        parts.append("<h3>5 · M8 — Policy plan and decisions</h3>"+details("One plan applied by code; clinical correctness not established",result["edit_plan"]))
        parts.append("<table><tr><th>Action / type</th><th>Input → output</th><th>Reason / evidence</th></tr>"+"".join(
            "<tr class='"+d["action"]+"'><td>"+esc(d["action"]+" / "+d["edit_type"])+"</td><td>"+esc(d["input"])+"<hr>"+esc(d["output"])+"</td><td>"+esc(d["reason"])+"<br>"+esc(d["evidence_ids"])+"</td></tr>" for d in result["report_decisions"])+"</table>")
        parts.append("<h3>6 · M9 — Draft / revised Findings</h3><p>Removed text in red; added text in green.</p>"+diff_html(result["original_findings"],result["revised_findings"])+"<p>Edits: "+esc(result["edit_counts"])+"</p>")
        parts.append("<h3>7 · Unresolved</h3><p>An edited sentence may still contain unresolved attributes. Kept text is not automatically verified.</p><pre>"+esc(json.dumps(result["unresolved_items"],ensure_ascii=False,indent=2))+"</pre></section>")
    target=root/"presentation.html"
    target.write_text("\n".join(parts)+"</html>",encoding="utf-8")
    return target
