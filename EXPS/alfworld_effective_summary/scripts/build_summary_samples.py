#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

DEFAULT_TRACE = Path('EXPS/analysis/024_ssca_retry_full/final_step_150_static_trace_bundle/full_trace_data.json')
DEFAULT_PROMPT = Path('EXPS/alfworld_effective_summary/prompts/simple_experience_summary_v1.txt')


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def extract_block(text: str, start_marker: str, end_marker: str | None = None, *, start_pos: int = 0) -> tuple[str, int]:
    start = text.find(start_marker, start_pos)
    if start < 0:
        return '', -1
    start += len(start_marker)
    if end_marker is None:
        end = len(text)
    else:
        end = text.find(end_marker, start)
        if end < 0:
            end = len(text)
    return text[start:end].strip(), end


def parse_summary_input(text: str) -> tuple[str, str]:
    anchor = text.rfind('[Compact first attempt trace]')
    if anchor < 0:
        anchor = 0
    trace, end_trace = extract_block(text, '[Compact first attempt trace]', '[Outcome feedback]', start_pos=anchor)
    outcome, _ = extract_block(text, '[Outcome feedback]', 'Write only', start_pos=max(end_trace, anchor))
    if not outcome:
        outcome, _ = extract_block(text, '[Outcome feedback]', None, start_pos=max(end_trace, anchor))
    return trace.strip(), outcome.strip()


def compact(text: str, max_chars: int) -> str:
    text = re.sub(r'\n{3,}', '\n\n', str(text or '').strip())
    if max_chars > 0 and len(text) > max_chars:
        return text[: max_chars - 120].rstrip() + f"\n... <truncated {len(text) - max_chars + 120} chars>"
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description='Build ALFWorld summary samples from final SSCA trace data.')
    parser.add_argument('--trace-json', default=str(DEFAULT_TRACE))
    parser.add_argument('--prompt-template', default=str(DEFAULT_PROMPT))
    parser.add_argument('--output-jsonl', required=True)
    parser.add_argument('--max-samples', type=int, default=12)
    parser.add_argument('--max-trace-chars', type=int, default=5200)
    parser.add_argument('--max-outcome-chars', type=int, default=1600)
    parser.add_argument('--prefer-mixed-status', action='store_true', default=True)
    args = parser.parse_args()

    data = read_json(Path(args.trace_json))
    rows = data['rows']
    chains = {int(c['summary_idx']): c for c in data.get('chains', [])}
    template = Path(args.prompt_template).read_text(encoding='utf-8')

    candidates = []
    for row in rows:
        if row.get('phase') != 'summary':
            continue
        idx = int(row.get('idx', len(candidates)))
        trace, outcome = parse_summary_input(str(row.get('input', '')))
        chain = chains.get(idx, {})
        if not trace:
            continue
        task = str(row.get('task') or chain.get('summary_input_task') or 'unknown')
        status = str(chain.get('attempt_status') or ('SUCCEEDED' if float(row.get('score') or 0) >= 10 else 'FAILED'))
        sample = {
            'sample_id': f'summary_row_{idx}',
            'summary_idx': idx,
            'task': task,
            'attempt_status': status,
            'reward1_env': chain.get('reward1_env'),
            'old_policy_summary': str(row.get('output') or '').strip(),
            'old_policy_summary_score': row.get('score'),
            'old_retry_delta_mode': chain.get('reward_delta_mode'),
            'old_traj2_top_task': chain.get('traj2_top_task'),
            'trace': compact(trace, args.max_trace_chars),
            'outcome': compact(outcome, args.max_outcome_chars),
        }
        sample['prompt'] = template.format(task=sample['task'], trace=sample['trace'], outcome=sample['outcome'])
        candidates.append(sample)

    if args.max_samples > 0:
        if args.prefer_mixed_status:
            succ = [x for x in candidates if x['attempt_status'].upper().startswith('SUCCEEDED')]
            fail = [x for x in candidates if not x['attempt_status'].upper().startswith('SUCCEEDED')]
            selected = []
            half = max(1, args.max_samples // 2)
            for bucket, limit in ((succ, half), (fail, args.max_samples - half)):
                seen_tasks = set()
                count = 0
                for item in bucket:
                    if item['task'] in seen_tasks:
                        continue
                    selected.append(item)
                    seen_tasks.add(item['task'])
                    count += 1
                    if count >= limit:
                        break
            if len(selected) < args.max_samples:
                used = {x['sample_id'] for x in selected}
                selected.extend(x for x in candidates if x['sample_id'] not in used)
            candidates = selected[: args.max_samples]
        else:
            candidates = candidates[: args.max_samples]

    out = Path(args.output_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8') as f:
        for sample in candidates:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    print(f'[OK] wrote {len(candidates)} samples -> {out}')


if __name__ == '__main__':
    main()
