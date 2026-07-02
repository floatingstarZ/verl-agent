#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

DATAFACTORY_SML = Path('/prj/corp/crd/morpheus/lasvegas/china-scratch/ziyuhuan/Projects/DataFactory_SML/SML')
if DATAFACTORY_SML.exists():
    sys.path.insert(0, str(DATAFACTORY_SML))

from sml.llm_client import LLMClient  # type: ignore  # noqa: E402


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding='utf-8').splitlines():
        line = raw.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids = set()
    with path.open(encoding='utf-8') as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get('sample_id'):
                ids.add(str(obj['sample_id']))
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description='Run closed-source LLM summaries for ALFWorld retry memory samples.')
    parser.add_argument('--input-jsonl', required=True)
    parser.add_argument('--output-jsonl', required=True)
    parser.add_argument('--env-file', default='/prj/corp/crd/morpheus/lasvegas/china-scratch/ziyuhuan/Projects/DataFactory_SML/SML/sml/.env')
    parser.add_argument('--model', default=os.environ.get('MODEL', 'azure::gpt-5.5'))
    parser.add_argument('--base-url', default=os.environ.get('QGENIE_API_ENDPOINT') or os.environ.get('OPENAI_BASE_URL') or 'https://qgenie-api.qualcomm.com/v1')
    parser.add_argument('--api-key-env', default='QGENIE_API_KEY')
    parser.add_argument('--max-samples', type=int, default=0)
    parser.add_argument('--max-tokens', type=int, default=260)
    parser.add_argument('--temperature', type=float, default=-1.0, help='negative means omit temperature')
    parser.add_argument('--sleep', type=float, default=0.2)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    load_env_file(Path(args.env_file))
    api_key = os.environ.get(args.api_key_env) or os.environ.get('OPENAI_API_KEY') or ''
    if not api_key:
        raise SystemExit(f'Missing API key. Set {args.api_key_env} or provide --env-file.')

    rows = read_jsonl(Path(args.input_jsonl))
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    out = Path(args.output_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = existing_ids(out) if args.resume else set()

    client = LLMClient(model=args.model, api_key=api_key, base_url=args.base_url, timeout=180, verify_ssl=False)
    temp = None if args.temperature < 0 else args.temperature

    mode = 'a' if args.resume else 'w'
    with out.open(mode, encoding='utf-8') as f:
        for i, sample in enumerate(rows, start=1):
            sid = str(sample.get('sample_id') or i)
            if sid in done:
                print(f'[SKIP] {sid}')
                continue
            print(f'[CALL] {i}/{len(rows)} {sid} task={sample.get("task")}')
            res = client.run(
                [
                    {'role': 'system', 'content': 'You write concise grounded ALFWorld retry memories.'},
                    {'role': 'user', 'content': sample['prompt']},
                ],
                max_tokens=args.max_tokens,
                temperature=temp,
                retries=3,
            )
            record = dict(sample)
            record.update({
                'closed_model': args.model,
                'closed_base_url': args.base_url,
                'closed_summary': str(res.get('text') or '').strip(),
                'closed_usage': res.get('usage'),
                'closed_response_time': res.get('response_time'),
                'closed_timestamp': res.get('timestamp'),
            })
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            f.flush()
            if args.sleep > 0:
                time.sleep(args.sleep)
    print(f'[OK] wrote -> {out}')


if __name__ == '__main__':
    main()
