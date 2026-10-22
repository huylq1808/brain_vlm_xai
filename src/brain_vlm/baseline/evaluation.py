"""Evaluate saved native outputs; labels are offline inputs, never model inputs.

This bounded endpoint is authored-report-relative primary-finding T2 direction.
It does not score lesion presence, location, measurements or clinical accuracy.
"""
from collections import Counter,defaultdict
from pathlib import Path
import csv
import html
import json
import re

from brain_vlm.common import read,write,sha,require
from brain_vlm.baseline.claims import analyze,NATURAL_VERSION
from brain_vlm.refinement.claim_extraction import sentence_spans

VERSION='native_t2_evaluation_v1'


def text_quality(text,available_sequences=()):
    """Observable text/prompt compliance only, not image truth."""
    sentences=sentence_spans(text);issues=[]
    if text.strip() and not re.search(r'[.!?][\"\u201d\u2019\']?$',text.strip()):issues.append('unfinished_sentence_tail')
    if not 2<=len(sentences)<=3:issues.append('outside_requested_2_to_3_sentences')
    measurements=[m.group() for m in re.finditer(r'\b\d+(?:\.\d+)?\s*(?:mm|cm)\b',text,re.I)]
    if measurements:issues.append('measurement_despite_generator_prompt')
    diffusion=bool(re.search(r'\b(?:restrict\w*\s+diffusion|diffusion\s+restrict\w*|DWI|ADC)\b',text,re.I))
    if diffusion and not {'dwi','adc'}<=set(available_sequences):issues.append('diffusion_statement_without_supplied_DWI_ADC')
    return {'sentence_count':len(sentences),'character_count':len(text),'issues':issues,
        'measurements':measurements,'interpretation':'Text structure/prompt/evidence-requirement diagnostics, not clinical error adjudication. No text is modified.'}


def endpoint(before,after,reference,edited):
    """Retained/flagged wrong claims remain wrong; absent final direction is error."""
    if reference not in {'low','intermediate','high'} or before is None:return {'scored':False}
    wb,wa=before!=reference,after!=reference
    return {'scored':True,'wrong_before':wb,'wrong_after':wa,
        'correct_repair':wb and not wa,'new_error':not wb and wa,
        'wrong_to_wrong_edit':bool(edited) and wb and wa,'missing_final_direction':after is None}


def totals(rows):
    scored=[r for r in rows if r.get('scored')]
    out={'studies':len({r['study_id'] for r in rows}),'drafts':len(rows),
        'completed':sum(r['status']=='completed' for r in rows),'scored_drafts':len(scored),
        'mapped_drafts':sum(r.get('mapped',False) for r in rows),
        'accepted_drafts':sum(r.get('accepted',False) for r in rows),
        'edited_drafts':sum(r.get('edits',0)>0 for r in rows),'exact_edits':sum(r.get('edits',0) for r in rows)}
    for key in ['wrong_before','wrong_after','correct_repair','new_error','wrong_to_wrong_edit','missing_final_direction']:
        out[key]=sum(r.get(key,False) for r in scored) if scored else None
    def ratio(a,b):return a/b if b else None
    out.update(mapping_coverage=ratio(out['mapped_drafts'],len(rows)),
        decision_coverage=ratio(out['accepted_drafts'],len(rows)),edit_coverage=ratio(out['edited_drafts'],len(rows)),
        repair_precision=ratio(out['correct_repair'],sum(r.get('edits',0)>0 for r in scored)) if scored else None,
        repair_recall=ratio(out['correct_repair'],out['wrong_before']) if scored else None,
        error_rate_before=ratio(out['wrong_before'],len(scored)) if scored else None,
        error_rate_after=ratio(out['wrong_after'],len(scored)) if scored else None)
    return out


def evaluate(run,output,labels=None):
    run,out=Path(run).resolve(),Path(output).resolve()
    require(not out.is_relative_to(run) and not run.is_relative_to(out),'Keep evaluation separate from the raw run')
    require(not out.exists(),'Choose a new evaluation output; preserve completed results')
    native=run/'refinement' if (run/'refinement/summary.json').exists() else run
    summary=read(native/'summary.json')
    require(summary['baseline'] in {'native_component_t2_baseline_v1','pretrained_medgemma_signal_v1'},'Expected bounded T2 artifacts')
    refs={};label_info=None
    if labels is not None:
        label_info=read(labels)
        require(label_info['version']=='weak_primary_t2_reference_v1','Use explicit weak T2 label provenance')
        for row in label_info['rows']:
            key=(row['study_id'],row['variant']);require(key not in refs,'Duplicate evaluation label')
            require(row['reference'] in {None,'low','intermediate','high'},'Invalid categorical reference')
            refs[key]=row
    inputs={str((native/'summary.json').relative_to(run)):sha(native/'summary.json')};rows=[]
    for item in summary['outputs']:
        sid=item['study_id'];variant=item.get('variant','unknown');label=refs.get((sid,variant),{})
        row={'study_id':sid,'variant':variant,'track':label.get('track','generated' if variant.startswith('generated') else 'controlled' if variant.startswith('controlled') else 'source'),
            'cohort':label.get('cohort','unlabelled'),'status':item['status'],'scored':False}
        if item['status']=='completed':
            path=native/item['result_path'];require(path.resolve().is_relative_to(native),'Unsafe result path')
            request=read(path.with_name('request.json'));result=read(path);policy=read(path.with_name('policy.json'))
            require(result['original_findings']==request['draft_text'],'Changed draft/result handoff')
            text=result['original_findings']
            for edit in sorted(result['applied_edits'],key=lambda e:e['start'],reverse=True):
                require(text[edit['start']:edit['end']]==edit['text'],'Edit span mismatch')
                text=text[:edit['start']]+edit['replacement']+text[edit['end']:]
            require(text==result['revised_findings'],'Final text has changes outside its declared edits')
            for p in [path,path.with_name('request.json'),path.with_name('policy.json')]:inputs[p.relative_to(run).as_posix()]=sha(p)
            def direction(text):
                selected=analyze(text,version=NATURAL_VERSION)['selected']
                return selected['direction'] if selected else None
            before,after=direction(result['original_findings']),direction(result['revised_findings'])
            row.update(mapped=bool(request['preparation']['mapped']),accepted=policy['accepted'],edits=len(result['applied_edits']),
                before=before,after=after,reference=label.get('reference'),reasons=policy['reasons'],
                result_path=path.relative_to(run).as_posix(),original_findings=result['original_findings'],revised_findings=result['revised_findings'])
            row.update(endpoint(before,after,label.get('reference'),row['edits']))
        rows.append(row)
    unused=set(refs)-{(r['study_id'],r['variant']) for r in rows}
    require(not unused,'Evaluation labels include absent draft variants')
    generation=[]
    for path in sorted((run/'generation').glob('*/generation.json')):
        record=read(path);request=read(path.with_name('request.json'))
        text=record.get('parsed_response',{}).get('findings','') if isinstance(record.get('parsed_response'),dict) else ''
        meta=record.get('generator_metadata',{})
        generation.append({'study_id':record['study_id'],'status':record['status'],
            'accepted_for_handoff':record.get('validation',{}).get('accepted_for_handoff',False),
            'text_quality':text_quality(text,{r['sequence'] for r in request['images']}) if text else None,
            'seconds':meta.get('last_generation_seconds'),'pixel_shape':meta.get('last_pixel_values_shape'),
            'cuda_peak_gib':meta.get('last_resources',{}).get('gpu_allocated_peak_gib'),
            'RSS_peak_gib':meta.get('last_resources',{}).get('process_rss_peak_sampled_gib')})
        inputs[path.relative_to(run).as_posix()]=sha(path);inputs[path.with_name('request.json').relative_to(run).as_posix()]=sha(path.with_name('request.json'))
    groups=defaultdict(list)
    for r in rows:groups[(r['cohort'],r['track'])].append(r)
    counts=Counter(issue for g in generation if g['text_quality'] for issue in g['text_quality']['issues'])
    metrics={'version':VERSION,'endpoint':'Weak authored-report-relative primary-finding T2 direction; scope/identity are proxies, not clinical truth',
        'scoring_parser':NATURAL_VERSION,'clinical_correctness':'not_evaluated','totals':totals(rows),
        'by_cohort_track':{c+'/'+t:totals(rs) for (c,t),rs in groups.items()},
        'generation':{'calls_recorded':len(generation),'completed':sum(g['status']=='completed' for g in generation),
            'handoff_accepted':sum(g['accepted_for_handoff'] for g in generation),'text_quality_issue_counts':dict(counts),'cases':generation},
        'labels_provenance':label_info.get('provenance') if label_info else None,
        'rows':rows,'notes':['No label or reference report is passed to inference by this evaluator.',
            'Failed/unmapped drafts remain in coverage denominators. Only an extractable pre-edit assertion with a declared reference is scored.',
            'Scoring uses the same fixed v6 parser across runs; execution mapping remains the recorded pipeline mapping.',
            'Flags retain errors. Zero-edit precision is null. Source/controlled/generated tracks must be reported separately.']}
    write(out/'metrics.json',metrics)
    write(out/'provenance.json',{'run':str(run),'input_sha256':inputs,'labels_sha256':sha(labels) if labels else None,
        'evaluator_sha256':sha(Path(__file__)),'new_model_calls':0})
    with (out/'per_draft.csv').open('w',newline='',encoding='utf-8-sig') as f:
        keys=['study_id','variant','track','cohort','status','mapped','accepted','edits','scored','reference','before','after','wrong_before','wrong_after','correct_repair','new_error','wrong_to_wrong_edit']
        writer=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    esc=lambda s:html.escape(str(s),quote=True)
    table='<table><tr><th>Cohort / track</th><th>Drafts / scored</th><th>Errors before→after</th><th>Repair / new / wrong→wrong</th><th>Accepted / edited</th></tr>'
    for name,m in metrics['by_cohort_track'].items():
        table+='<tr>'+''.join('<td>'+esc(v)+'</td>' for v in [name,f"{m['drafts']} / {m['scored_drafts']}",f"{m['wrong_before']} → {m['wrong_after']}",f"{m['correct_repair']} / {m['new_error']} / {m['wrong_to_wrong_edit']}",f"{m['accepted_drafts']} / {m['edited_drafts']}"])+'</tr>'
    page='<!doctype html><html lang="vi"><meta charset="utf-8"><title>Baseline quantitative evaluation</title><style>body{font:16px/1.6 system-ui;max-width:1250px;margin:auto;padding:24px;color:#17304b}pre{white-space:pre-wrap;overflow-wrap:anywhere}td,th{padding:8px;border:1px solid #cbd5e1}table{border-collapse:collapse;width:100%}details{margin:15px 0}</style><h1>Baseline: kết quả định lượng</h1><p>'+esc(metrics['endpoint'])+'</p><p>Controlled errors are artificial; source references are weak authored-report labels. Natural claims with uncertain finding correspondence are proxy diagnostics. None establishes clinical accuracy.</p>'+table+'</table><h2>Generation: completion và chất lượng văn bản tách riêng</h2><pre>'+esc(json.dumps(metrics['generation'],ensure_ascii=False,indent=2))+'</pre><h2>Từng draft</h2>'
    for r in rows:page+='<details><summary>'+esc(r['study_id']+' / '+r['variant'])+'</summary><pre>'+esc(json.dumps(r,ensure_ascii=False,indent=2))+'</pre></details>'
    (out/'index.html').write_text(page+'</html>',encoding='utf-8')
    return out/'index.html'
