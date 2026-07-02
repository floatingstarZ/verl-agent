#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
PLATFORM_HOME = Path('/prj/corp/crd/morpheus/lasvegas/china-scratch/ziyuhuan')
DEFAULT_ALFWORLD_DATA = PLATFORM_HOME / '.cache/alfworld'
os.environ.setdefault('ALFWORLD_DATA', str(DEFAULT_ALFWORLD_DATA))
DATAFACTORY_SML = PLATFORM_HOME / 'Projects/DataFactory_SML/SML'
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if DATAFACTORY_SML.exists() and str(DATAFACTORY_SML) not in sys.path:
    sys.path.insert(0, str(DATAFACTORY_SML))

from agent_system.environments.env_manager import AlfWorldEnvironmentManager  # noqa: E402
from agent_system.environments.env_package.alfworld import alfworld_projection, build_alfworld_envs  # noqa: E402

ALF_CONFIG = ROOT / 'agent_system/environments/env_package/alfworld/configs/config_tw.yaml'
ACTION_SYSTEM = (
    'You are an ALFWorld TextWorld agent. Return exactly one action in this format: '
    '<think>brief reason</think>\n<action>one admissible action copied exactly</action>. '
    'The action must be one of the admissible actions in the user prompt. No extra text.'
)

SUCCESS_SUMMARY_PROMPT = '''The previous attempt succeeded.

Write a short experience summary that helps the agent perform the same ALFWorld task better next time.
Use only what actually worked in the previous attempt.
Focus on the useful experience: where the needed object/tool/receptacle was, what key action order worked, and what can be skipped.
Do not include failed detours, loops, or actions that undo the success.

Task:
{task}

Successful attempt:
{context}

Return a short memory:
Experience:'''

FAILURE_SUMMARY_PROMPT = '''The previous attempt failed.

Write a short failure summary that helps the agent choose better actions in the next attempt.
Use only what was actually seen or tried.
Focus on what went wrong: wrong places searched, missing object/tool/receptacle, repeated loops, invalid or useless actions, or an unfinished subgoal.
If something important is still unknown, say it is unknown.

Task:
{task}

Failed attempt:
{context}

Return a short memory:
Failure:'''

TASK_STOP_WORDS = {
    'a', 'an', 'and', 'in', 'into', 'on', 'onto', 'put', 'find', 'two', 'some', 'the',
    'with', 'to', 'of', 'them', 'it', 'then', 'look', 'at', 'under', 'hot', 'clean',
    'cool', 'heat', 'slice', 'examine', 'task', 'your', 'is', 'that', 'this', 'from',
}


def compact(text: Any, limit: int) -> str:
    value = re.sub(r'\n{3,}', '\n\n', str(text or '').strip())
    if limit > 0 and len(value) > limit:
        return value[: limit - 80].rstrip() + f'\n... <truncated {len(value) - limit + 80} chars>'
    return value


def extract_task(text: str) -> str:
    marker = 'Your task is to: '
    start = str(text or '').find(marker)
    if start < 0:
        return 'unknown'
    start += len(marker)
    end = str(text).find('\n', start)
    if end < 0:
        end = str(text).find('.', start)
    if end < 0:
        end = len(str(text))
    return str(text)[start:end].strip().rstrip('.') or 'unknown'


def get_action_text(response: str) -> str:
    match = re.search(r'<action>(.*?)</action>', response or '', flags=re.I | re.S)
    if match:
        return re.sub(r'\s+', ' ', match.group(1)).strip().lower()
    return ''


def infer_won(info: dict[str, Any] | None, reward: float) -> bool:
    if isinstance(info, dict) and 'won' in info:
        return bool(info.get('won'))
    return reward >= 10.0


def task_terms(task: str) -> list[str]:
    terms: list[str] = []
    for token in re.split(r'[^A-Za-z0-9]+', str(task or '').lower()):
        if len(token) >= 3 and token not in TASK_STOP_WORDS and token not in terms:
            terms.append(token)
    return terms


def parse_seen_items(text: str) -> list[str]:
    items: list[str] = []
    for match in re.findall(r'you see (.*?)(?:\.|$)', str(text or ''), flags=re.I):
        cleaned = re.sub(r'\b(?:a|an|the)\b\s+', '', match.lower())
        for part in re.split(r',|\band\b', cleaned):
            item = part.strip(" .;:'\"()[]")
            if item and item != 'nothing' and len(item) <= 80 and item not in items:
                items.append(item)
    return items


def maybe_target_related(action: str, text: str, terms: list[str]) -> bool:
    combined = f'{action}\n{text}'.lower()
    return any(term in combined for term in terms)


def build_prev_context(task: str, trace: list[dict[str, Any]], won: bool, reward: float, max_events: int = 18) -> str:
    terms = task_terms(task)
    visited: list[str] = []
    opened: list[str] = []
    target_events: list[str] = []
    action_events: list[str] = []
    empty_events: list[str] = []
    invalid_events: list[str] = []
    terminal_event = ''
    repeated_counter: dict[str, int] = {}

    initial_places = parse_seen_items(trace[0].get('obs', '') if trace else '')
    for item in trace:
        step = int(item.get('step') or 0)
        action = str(item.get('projected_action') or item.get('action') or '').strip().lower()
        feedback = str(item.get('feedback') or '')
        obs = str(item.get('obs') or '')
        if action:
            repeated_counter[action] = repeated_counter.get(action, 0) + 1
        if action.startswith('go to '):
            place = action[len('go to '):].strip()
            if place and place not in visited:
                visited.append(place)
        if action.startswith('open '):
            place = action[len('open '):].strip()
            if place and place not in opened:
                opened.append(place)
        if maybe_target_related(action, f'{obs}\n{feedback}', terms):
            target_events.append(f'S{step}: {action}; {compact(feedback or obs, 220)}')
        if re.match(r'^(take|move|put|use|cool|heat|clean|slice|examine)\b', action):
            action_events.append(f'S{step}: {action}; {compact(feedback, 180)}')
        if re.search(r'you see nothing|empty|is closed|nothing happens', feedback.lower()):
            empty_events.append(f'S{step}: {action}; {compact(feedback, 150)}')
        if not bool(item.get('valid', True)):
            invalid_events.append(f'S{step}: invalid {action}')
        if bool(item.get('done')) or float(item.get('reward') or 0.0) > 0.0:
            terminal_event = f'S{step}: {action}; reward={item.get("reward")}; {compact(feedback, 220)}'

    repeated = [f'{act} x{cnt}' for act, cnt in sorted(repeated_counter.items(), key=lambda pair: (-pair[1], pair[0])) if cnt >= 3]
    unvisited = [place for place in initial_places if place not in visited]
    selected_events: list[str] = []
    for event in target_events + action_events + empty_events + invalid_events:
        if event not in selected_events:
            selected_events.append(event)
    if len(selected_events) > max_events:
        selected_events = selected_events[: max_events // 2] + selected_events[-(max_events - max_events // 2):]

    header = 'Succeeded' if won else 'Failed'
    lines = [
        f'{header} after {len(trace)} steps with reward {reward:.1f}.',
        'Task terms: ' + (', '.join(terms) or 'unknown'),
    ]
    if visited:
        lines.append('Visited: ' + ', '.join(visited[:24]))
    if opened:
        lines.append('Opened: ' + ', '.join(opened[:18]))
    if unvisited and not won:
        lines.append('Still unvisited from initial scene: ' + ', '.join(unvisited[:18]))
    if terminal_event:
        lines.append('Final important event: ' + terminal_event)
    if selected_events:
        lines.append('Key steps:')
        lines.extend('- ' + event for event in selected_events)
    if repeated:
        lines.append('Repeated/loop actions: ' + '; '.join(repeated[:10]))
    return compact('\n'.join(lines), 3600)


@dataclass
class EpisodeState:
    local_id: int
    split: str
    task: str = ''
    gamefile: str = ''
    done: bool = False
    won: bool = False
    reward: float = 0.0
    steps: int = 0
    invalid_actions: int = 0
    final_info: dict[str, Any] = field(default_factory=dict)
    trace: list[dict[str, Any]] = field(default_factory=list)


class BasePolicy:
    def generate(self, prompts: list[str]) -> list[tuple[str, dict[str, Any]]]:
        raise NotImplementedError


class QGeniePolicy(BasePolicy):
    def __init__(self, model: str, base_url: str, api_key: str, max_tokens: int, max_workers: int):
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from sml.llm_client import LLMClient  # type: ignore
        self.client = LLMClient(model=model, api_key=api_key, base_url=base_url, timeout=180, verify_ssl=False)
        self.model = model
        self.max_tokens = max_tokens
        self.max_workers = max_workers
        self.ThreadPoolExecutor = ThreadPoolExecutor
        self.as_completed = as_completed

    def _one(self, prompt: str) -> tuple[str, dict[str, Any]]:
        messages = [{'role': 'system', 'content': ACTION_SYSTEM}, {'role': 'user', 'content': prompt}]
        res = self.client.run(messages, max_tokens=self.max_tokens, temperature=None, retries=3)
        return str(res.get('text') or '').strip(), {
            'usage': res.get('usage'),
            'response_time': res.get('response_time'),
            'timestamp': res.get('timestamp'),
        }

    def generate(self, prompts: list[str]) -> list[tuple[str, dict[str, Any]]]:
        outputs: list[tuple[str, dict[str, Any]] | None] = [None] * len(prompts)
        with self.ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(self._one, prompt): idx for idx, prompt in enumerate(prompts)}
            for future in self.as_completed(futures):
                idx = futures[future]
                try:
                    outputs[idx] = future.result()
                except Exception as exc:  # noqa: BLE001
                    outputs[idx] = ('<think>fallback</think>\n<action>look</action>', {'error': f'{type(exc).__name__}: {exc}'})
        return [item if item is not None else ('<think>missing</think>\n<action>look</action>', {'error': 'missing'}) for item in outputs]


class HFPolicy(BasePolicy):
    def __init__(self, model_path: str, max_new_tokens: int, temperature: float, top_p: float, batch_size: int, dtype: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        torch_dtype = {'auto': 'auto', 'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}.get(dtype, 'auto')
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map='auto',
            trust_remote_code=True,
            attn_implementation=os.environ.get('HF_ATTN_IMPLEMENTATION') or None,
        )
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.batch_size = batch_size

    def _format(self, prompt: str) -> str:
        messages = [{'role': 'system', 'content': ACTION_SYSTEM}, {'role': 'user', 'content': prompt}]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def generate(self, prompts: list[str]) -> list[tuple[str, dict[str, Any]]]:
        outputs: list[tuple[str, dict[str, Any]]] = []
        for start in range(0, len(prompts), self.batch_size):
            batch_prompts = [self._format(prompt) for prompt in prompts[start:start + self.batch_size]]
            encoded = self.tokenizer(batch_prompts, return_tensors='pt', padding=True, truncation=True, max_length=4096)
            encoded = {key: value.to(self.model.device) for key, value in encoded.items()}
            started = time.perf_counter()
            with self.torch.inference_mode():
                generated = self.model.generate(
                    **encoded,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=self.temperature > 0,
                    temperature=self.temperature if self.temperature > 0 else None,
                    top_p=self.top_p if self.temperature > 0 else None,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            input_len = encoded['input_ids'].shape[1]
            decoded = self.tokenizer.batch_decode(generated[:, input_len:], skip_special_tokens=True)
            elapsed = round(time.perf_counter() - started, 4)
            outputs.extend((text.strip(), {'response_time': elapsed}) for text in decoded)
        return outputs


def make_env(split: str, env_num: int, seed: int, ray_num_cpus: int, worker_cpus: float, eval_dataset: str) -> AlfWorldEnvironmentManager:
    import ray
    if ray.is_initialized():
        ray.shutdown()
    ray.init(num_cpus=ray_num_cpus, ignore_reinit_error=True, include_dashboard=False)
    is_train = split == 'train'
    env_kwargs = {} if is_train else {'eval_dataset': eval_dataset}
    envs = build_alfworld_envs(
        str(ALF_CONFIG),
        seed=seed,
        env_num=env_num,
        group_n=1,
        resources_per_worker={'num_cpus': worker_cpus},
        is_train=is_train,
        env_kwargs=env_kwargs,
    )
    config = type('Cfg', (), {'env': type('EnvCfg', (), {'history_length': 2})()})()
    return AlfWorldEnvironmentManager(envs, alfworld_projection, config)


def finalize_record(state: EpisodeState, model_name: str, backend: str, run_seed: int, attempt_index: int) -> dict[str, Any]:
    outcome = 'success' if state.won else 'failure'
    context = build_prev_context(state.task, state.trace, state.won, state.reward)
    prompt_template = SUCCESS_SUMMARY_PROMPT if state.won else FAILURE_SUMMARY_PROMPT
    return {
        'id': f'{state.split}_{outcome}_{attempt_index:05d}_{state.local_id}',
        'split': state.split,
        'outcome': outcome,
        'won': state.won,
        'reward': state.reward,
        'steps': state.steps,
        'invalid_actions': state.invalid_actions,
        'task': state.task,
        'gamefile': state.gamefile,
        'backend': backend,
        'model': model_name,
        'seed': run_seed,
        'attempt_index': attempt_index,
        'prev_context': context,
        'summary_prompt': prompt_template.format(task=state.task, context=context),
        'trace': state.trace,
    }


def collect_split(args: argparse.Namespace, split: str, policy: BasePolicy, out_dir: Path) -> list[dict[str, Any]]:
    env = make_env(split, args.env_num, args.seed + (0 if split == 'train' else 100000), args.ray_num_cpus, args.worker_cpus, args.eval_dataset)
    wanted = {'success': args.target_success, 'failure': args.target_failure}
    buckets: dict[str, list[dict[str, Any]]] = {'success': [], 'failure': []}
    seen_gamefiles: dict[str, set[str]] = {'success': set(), 'failure': set()}
    attempts = 0
    max_attempt_batches = args.max_attempt_batches
    try:
        while any(len(buckets[key]) < wanted[key] for key in wanted) and attempts < max_attempt_batches:
            attempts += 1
            obs, infos = env.reset({})
            states: list[EpisodeState] = []
            raw_obs = list(obs.get('anchor') or obs.get('text') or [''] * args.env_num)
            for idx in range(args.env_num):
                info = infos[idx] if idx < len(infos) and isinstance(infos[idx], dict) else {}
                states.append(EpisodeState(
                    local_id=idx,
                    split=split,
                    task=extract_task(str(raw_obs[idx])),
                    gamefile=str(info.get('extra.gamefile') or ''),
                ))

            for step_idx in range(args.max_steps):
                active = [idx for idx, state in enumerate(states) if not state.done]
                if not active:
                    break
                prompts = list(obs.get('text') or [''] * args.env_num)
                active_prompts = [prompts[idx] for idx in active]
                generations = policy.generate(active_prompts)
                raw_actions = ['<think>done</think>\n<action>look</action>'] * args.env_num
                meta_by_idx: dict[int, dict[str, Any]] = {}
                for offset, idx in enumerate(active):
                    text, meta = generations[offset]
                    raw_actions[idx] = text
                    meta_by_idx[idx] = meta

                prev_obs = list(obs.get('anchor') or obs.get('text') or [''] * args.env_num)
                next_obs, rewards, dones, infos = env.step(raw_actions)
                next_anchor = list(next_obs.get('anchor') or next_obs.get('text') or [''] * args.env_num)
                rewards_np = np.asarray(rewards).reshape(-1)
                dones_np = np.asarray(dones).reshape(-1)

                for idx in active:
                    state = states[idx]
                    info = infos[idx] if idx < len(infos) and isinstance(infos[idx], dict) else {}
                    reward = float(rewards_np[idx])
                    valid = bool(info.get('is_action_valid', False))
                    if not valid:
                        state.invalid_actions += 1
                    state.reward += reward
                    state.steps += 1
                    state.final_info = dict(info)
                    projected_action = get_action_text(raw_actions[idx])
                    state.trace.append({
                        'step': step_idx + 1,
                        'obs': prev_obs[idx],
                        'raw_action': raw_actions[idx],
                        'projected_action': projected_action,
                        'valid': valid,
                        'reward': reward,
                        'done': bool(dones_np[idx]),
                        'won': infer_won(info, state.reward),
                        'feedback': next_anchor[idx],
                        'llm_meta': meta_by_idx.get(idx, {}),
                    })
                    if bool(dones_np[idx]):
                        state.done = True
                        state.won = infer_won(info, state.reward)
                obs = next_obs

            for state in states:
                if not state.done:
                    state.done = True
                    state.won = infer_won(state.final_info, state.reward)
                outcome = 'success' if state.won else 'failure'
                if len(buckets[outcome]) >= wanted[outcome]:
                    continue
                gamefile_key = state.gamefile or f'{state.task}:{attempts}:{state.local_id}'
                if args.unique_gamefiles and gamefile_key in seen_gamefiles[outcome]:
                    continue
                seen_gamefiles[outcome].add(gamefile_key)
                buckets[outcome].append(finalize_record(state, args.model_path or args.model, args.backend, args.seed, attempts))

            counts = {key: len(value) for key, value in buckets.items()}
            print(f'[COLLECT] split={split} batch={attempts} counts={counts}', flush=True)

        records = buckets['success'][:wanted['success']] + buckets['failure'][:wanted['failure']]
        split_path = out_dir / f'{split}_records.jsonl'
        split_path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in records), encoding='utf-8')
        return records
    finally:
        try:
            env.envs.close()
        except Exception:
            pass
        try:
            import ray
            if ray.is_initialized():
                ray.shutdown()
        except Exception:
            pass


def load_env_file(path: str) -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding='utf-8').splitlines():
        line = raw.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def build_policy(args: argparse.Namespace) -> BasePolicy:
    if args.backend == 'qgenie':
        load_env_file(args.env_file)
        api_key = os.environ.get(args.api_key_env) or os.environ.get('OPENAI_API_KEY') or ''
        if not api_key:
            raise SystemExit(f'Missing API key: set {args.api_key_env} or use --env-file')
        return QGeniePolicy(args.model, args.base_url, api_key, args.action_max_tokens, args.max_workers)
    return HFPolicy(args.model_path, args.action_max_tokens, args.temperature, args.top_p, args.hf_batch_size, args.dtype)


def main() -> None:
    parser = argparse.ArgumentParser(description='Collect ALFWorld previous success/failure trajectories for summary-memory experiments.')
    parser.add_argument('--backend', choices=['hf', 'qgenie'], default=os.environ.get('COLLECT_BACKEND', 'hf'))
    parser.add_argument('--model-path', default=os.environ.get('MODEL_PATH', ''))
    parser.add_argument('--model', default=os.environ.get('MODEL', 'azure::gpt-5.4-mini'))
    parser.add_argument('--base-url', default=os.environ.get('BASE_URL') or os.environ.get('QGENIE_API_ENDPOINT') or 'https://qgenie-api.qualcomm.com/v1')
    parser.add_argument('--api-key-env', default=os.environ.get('API_KEY_ENV', 'QGENIE_API_KEY'))
    parser.add_argument('--env-file', default=os.environ.get('ENV_FILE', str(DATAFACTORY_SML / 'sml/.env')))
    parser.add_argument('--splits', default=os.environ.get('SPLITS', 'train,test'))
    parser.add_argument('--eval-dataset', default=os.environ.get('EVAL_DATASET', 'eval_in_distribution'))
    parser.add_argument('--target-success', type=int, default=int(os.environ.get('TARGET_SUCCESS', '16')))
    parser.add_argument('--target-failure', type=int, default=int(os.environ.get('TARGET_FAILURE', '16')))
    parser.add_argument('--env-num', type=int, default=int(os.environ.get('ENV_NUM', '8')))
    parser.add_argument('--max-steps', type=int, default=int(os.environ.get('MAX_STEPS', '50')))
    parser.add_argument('--max-attempt-batches', type=int, default=int(os.environ.get('MAX_ATTEMPT_BATCHES', '80')))
    parser.add_argument('--seed', type=int, default=int(os.environ.get('SEED', '2026')))
    parser.add_argument('--ray-num-cpus', type=int, default=int(os.environ.get('RAY_NUM_CPUS', '16')))
    parser.add_argument('--worker-cpus', type=float, default=float(os.environ.get('ENV_WORKER_CPUS', '0.2')))
    parser.add_argument('--max-workers', type=int, default=int(os.environ.get('MAX_WORKERS', '2')))
    parser.add_argument('--hf-batch-size', type=int, default=int(os.environ.get('HF_BATCH_SIZE', '4')))
    parser.add_argument('--action-max-tokens', type=int, default=int(os.environ.get('ACTION_MAX_TOKENS', '160')))
    parser.add_argument('--temperature', type=float, default=float(os.environ.get('TEMPERATURE', '0.4')))
    parser.add_argument('--top-p', type=float, default=float(os.environ.get('TOP_P', '0.95')))
    parser.add_argument('--dtype', choices=['auto', 'bf16', 'fp16', 'fp32'], default=os.environ.get('DTYPE', 'bf16'))
    parser.add_argument('--out-dir', default=os.environ.get('OUT_DIR', ''))
    parser.add_argument('--unique-gamefiles', action=argparse.BooleanOptionalAction, default=os.environ.get('UNIQUE_GAMEFILES', '1') != '0')
    args = parser.parse_args()

    if args.backend == 'hf' and not args.model_path:
        ref = PLATFORM_HOME / '.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/refs/main'
        if ref.exists():
            snapshot = PLATFORM_HOME / '.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots' / ref.read_text().strip()
            if (snapshot / 'config.json').exists():
                args.model_path = str(snapshot)
        if not args.model_path:
            args.model_path = 'Qwen/Qwen2.5-1.5B-Instruct'

    run_tag = time.strftime('%Y%m%d_%H%M%S')
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / 'EXPS/alfworld_effective_summary/outputs' / f'prev_traj_collect_{args.backend}_{run_tag}'
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'config.json').write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'[INFO] out_dir={out_dir}', flush=True)
    print(f'[INFO] backend={args.backend} model={args.model_path if args.backend == "hf" else args.model}', flush=True)

    policy = build_policy(args)
    all_records: list[dict[str, Any]] = []
    for split in [item.strip() for item in args.splits.split(',') if item.strip()]:
        all_records.extend(collect_split(args, split, policy, out_dir))

    (out_dir / 'all_records.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in all_records), encoding='utf-8')
    metrics: dict[str, Any] = {'total_records': len(all_records), 'by_split_outcome': {}}
    for row in all_records:
        key = f"{row['split']}/{row['outcome']}"
        metrics['by_split_outcome'][key] = metrics['by_split_outcome'].get(key, 0) + 1
    (out_dir / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    print('[DONE] ' + str(out_dir), flush=True)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
