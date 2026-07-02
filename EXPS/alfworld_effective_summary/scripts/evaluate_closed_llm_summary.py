#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

STOP = {'a','an','and','in','into','on','onto','put','find','two','some','the','with','to','of','them','it','then','task','your','is'}
ACTION_WORDS = {'go','open','take','put','move','examine','inventory','look','cool','heat','clean','use','slice'}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def task_terms(task: str) -> set[str]:
    out=set()
    for tok in re.split(r'[^A-Za-z0-9]+', task.lower()):
        if len(tok)>=3 and tok not in STOP:
            out.add(tok)
    return out


def object_names(text: str) -> set[str]:
    names=set()
    for m in re.finditer(r'\b([a-z]+(?:[a-z]+)?)(?:\s+\d+)?\b', text.lower()):
        tok=m.group(1)
        if len(tok)>=3 and tok not in STOP and tok not in ACTION_WORDS:
            names.add(tok)
    return names


def score_summary(summary: str, task: str, trace: str) -> dict[str, Any]:
    text=str(summary or '').strip()
    lower=text.lower()
    lines=[ln.strip() for ln in text.splitlines() if ln.strip()]
    terms=task_terms(task)
    hits=sum(1 for t in terms if t in lower)
    trace_names=object_names(trace + '\n' + task)
    summary_names=object_names(text)
    suspicious=sorted(n for n in summary_names if n not in trace_names)[:20]
    starts_ok = all(any(line.startswith(prefix) for prefix in ('Experience:', 'Next try:', 'Avoid:')) for line in lines[:3]) if lines else False
    has_experience='experience:' in lower
    has_next='next try:' in lower or 'next:' in lower
    has_avoid='avoid:' in lower
    action_hits=sum(1 for w in ACTION_WORDS if re.search(r'\b'+re.escape(w)+r'\b', lower))
    word_count=len(text.split())
    no_wrapper=not any(x in text for x in ['```','<think>','</think>','<action>','</action>'])
    concise=word_count <= 95
    return {
        'word_count': word_count,
        'has_three_fields': bool(has_experience and has_next and has_avoid),
        'starts_with_requested_fields': bool(starts_ok),
        'task_term_overlap': hits / max(1, min(len(terms), 4)),
        'action_vocab_hits': action_hits,
        'no_wrapper': bool(no_wrapper),
        'concise': bool(concise),
        'suspicious_new_names': suspicious,
        'suspicious_new_name_count': len(suspicious),
        'heuristic_score': round(
            0.25*float(has_experience and has_next and has_avoid)
            +0.20*float(starts_ok)
            +0.20*min(1.0, hits / max(1, min(len(terms), 4)))
            +0.15*float(action_hits > 0)
            +0.10*float(no_wrapper)
            +0.10*float(concise),
            4,
        ),
    }


def pct(x: float) -> str:
    return f'{100*x:.1f}%'


def main() -> None:
    parser=argparse.ArgumentParser(description='Evaluate closed LLM ALFWorld summaries with lightweight groundedness heuristics.')
    parser.add_argument('--input-jsonl', required=True)
    parser.add_argument('--output-md', required=True)
    parser.add_argument('--output-json', default='')
    args=parser.parse_args()
    rows=read_jsonl(Path(args.input_jsonl))
    scored=[]
    for row in rows:
        closed=score_summary(row.get('closed_summary',''), row.get('task',''), row.get('trace',''))
        old=score_summary(row.get('old_policy_summary',''), row.get('task',''), row.get('trace',''))
        scored.append({'row':row,'closed_score':closed,'old_score':old})

    def avg(key: str, which: str) -> float:
        vals=[float(x[f'{which}_score'][key]) for x in scored]
        return sum(vals)/len(vals) if vals else 0.0
    def rate(key: str, which: str) -> float:
        vals=[bool(x[f'{which}_score'][key]) for x in scored]
        return sum(vals)/len(vals) if vals else 0.0

    out=[]
    out.append('# Closed LLM ALFWorld Summary Probe Report')
    out.append('')
    out.append(f'- Samples: {len(scored)}')
    if scored:
        models=Counter(str(x['row'].get('closed_model')) for x in scored)
        out.append(f'- Models: {dict(models)}')
    out.append('')
    out.append('## Aggregate Heuristics')
    out.append('')
    out.append('| Metric | Closed LLM | Old policy summary |')
    out.append('|---|---:|---:|')
    out.append(f"| heuristic_score mean | {avg('heuristic_score','closed'):.3f} | {avg('heuristic_score','old'):.3f} |")
    out.append(f"| requested 3 fields rate | {pct(rate('has_three_fields','closed'))} | {pct(rate('has_three_fields','old'))} |")
    out.append(f"| starts with requested fields | {pct(rate('starts_with_requested_fields','closed'))} | {pct(rate('starts_with_requested_fields','old'))} |")
    out.append(f"| task-term overlap mean | {avg('task_term_overlap','closed'):.3f} | {avg('task_term_overlap','old'):.3f} |")
    out.append(f"| no wrapper/tag rate | {pct(rate('no_wrapper','closed'))} | {pct(rate('no_wrapper','old'))} |")
    out.append(f"| concise rate | {pct(rate('concise','closed'))} | {pct(rate('concise','old'))} |")
    out.append(f"| suspicious new name count mean | {avg('suspicious_new_name_count','closed'):.2f} | {avg('suspicious_new_name_count','old'):.2f} |")
    out.append('')
    out.append('## Per-Sample Review')
    out.append('')
    for item in scored:
        row=item['row']
        out.append(f"### {row.get('sample_id')} · {row.get('attempt_status')} · task: `{row.get('task')}`")
        out.append('')
        out.append(f"- Closed score: {item['closed_score']['heuristic_score']} ; old score: {item['old_score']['heuristic_score']}")
        out.append(f"- Old retry delta mode from previous run: {row.get('old_retry_delta_mode')} ; old traj2 top task: `{row.get('old_traj2_top_task')}`")
        if item['closed_score']['suspicious_new_names']:
            out.append(f"- Closed suspicious new names: {', '.join(item['closed_score']['suspicious_new_names'])}")
        out.append('')
        out.append('Closed LLM summary:')
        out.append('')
        out.append('```text')
        out.append(str(row.get('closed_summary','')).strip())
        out.append('```')
        out.append('')
        out.append('Old policy summary:')
        out.append('')
        out.append('```text')
        out.append(str(row.get('old_policy_summary','')).strip()[:1200])
        out.append('```')
        out.append('')
    md=Path(args.output_md)
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text('\n'.join(out), encoding='utf-8')
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(scored, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'[OK] wrote {md}')


if __name__ == '__main__':
    main()
