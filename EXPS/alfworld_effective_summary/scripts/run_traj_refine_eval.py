#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ALFWORLD_DATA = Path('/prj/corp/crd/morpheus/lasvegas/china-scratch/ziyuhuan/.cache/alfworld')
os.environ.setdefault('ALFWORLD_DATA', str(DEFAULT_ALFWORLD_DATA))
DATAFACTORY_SML = Path('/prj/corp/crd/morpheus/lasvegas/china-scratch/ziyuhuan/Projects/DataFactory_SML/SML')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if DATAFACTORY_SML.exists() and str(DATAFACTORY_SML) not in sys.path:
    sys.path.insert(0, str(DATAFACTORY_SML))

from agent_system.environments.env_manager import AlfWorldEnvironmentManager  # noqa: E402
from agent_system.environments.env_package.alfworld import alfworld_projection, build_alfworld_envs  # noqa: E402
from sml.llm_client import LLMClient  # type: ignore  # noqa: E402

ALF_CONFIG = ROOT / 'agent_system/environments/env_package/alfworld/configs/config_tw.yaml'

ACTION_SYSTEM = (
    'You are an ALFWorld TextWorld agent. Return exactly one action in this format: '
    '<think>brief reason</think>\n<action>one admissible action copied exactly</action>. '
    'The action must be one of the admissible actions in the user prompt. No extra text.'
)

SUMMARY_SYSTEM = 'You write short grounded retry memories for ALFWorld agents. No hidden reasoning.'

SUMMARY_TEMPLATE = '''Write a short retry memory for the SAME ALFWorld task.
Use only the structured evidence below; do not invent object locations.
Next try must be an executable route using concrete verbs: go to, open, take X from Y, move/put X to Y, use, cool, heat, clean, slice, examine.
Before every take, include go to the source; before every move/put, include go to the destination; open closed containers before taking from them.
If the target object was not observed, say unknown and use the listed unvisited places as candidates.
For examine-with-desklamp tasks: use the desklamp, go to the object, then examine the object; do not relocate the object unless evidence proves it is needed.
For succeeded trajectories, prefer completed placements or terminal successful operations over earlier failed loops.
For two-object tasks, mention both objects if observed; otherwise say second object unknown. Never take an object back out of the final target after placing it.

Task: {task}
Outcome: {status}, reward={reward}, steps={steps}, invalid_actions={invalid}
Evidence:
{trace}

Return exactly three short lines:
Experience: <reliable facts or route; include unknowns explicitly>
Next try: <concrete retry plan from start>
Avoid: <bad actions/places, loops, or none>
'''


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding='utf-8').splitlines():
        line = raw.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


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


def compact(text: str, limit: int) -> str:
    text = re.sub(r'\n{3,}', '\n\n', str(text or '').strip())
    if limit > 0 and len(text) > limit:
        return text[: limit - 80].rstrip() + f'\n... <truncated {len(text) - limit + 80} chars>'
    return text


def get_action_text(response: str) -> str:
    m = re.search(r'<action>(.*?)</action>', response or '', flags=re.I | re.S)
    return m.group(1).strip().lower() if m else ''


def infer_won(info: dict[str, Any] | None, reward: float) -> bool:
    if isinstance(info, dict) and 'won' in info:
        return bool(info.get('won'))
    return reward >= 10.0


@dataclass
class EpisodeState:
    task_id: int
    task: str = ''
    gamefile: str = ''
    done: bool = False
    won: bool = False
    reward: float = 0.0
    steps: int = 0
    invalid_actions: int = 0
    final_info: dict[str, Any] = field(default_factory=dict)
    trace: list[dict[str, Any]] = field(default_factory=list)


def call_llm(client: LLMClient, messages: list[dict[str, str]], max_tokens: int, retries: int = 3) -> tuple[str, dict[str, Any]]:
    res = client.run(messages, max_tokens=max_tokens, temperature=None, retries=retries)
    return str(res.get('text') or '').strip(), {
        'usage': res.get('usage'),
        'response_time': res.get('response_time'),
        'timestamp': res.get('timestamp'),
    }


def call_actions_parallel(
    client: LLMClient,
    prompts: list[str],
    active_indices: list[int],
    max_workers: int,
    max_tokens: int,
) -> dict[int, tuple[str, dict[str, Any]]]:
    outputs: dict[int, tuple[str, dict[str, Any]]] = {}
    if not active_indices:
        return outputs
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {}
        for idx in active_indices:
            messages = [
                {'role': 'system', 'content': ACTION_SYSTEM},
                {'role': 'user', 'content': prompts[idx]},
            ]
            futures[pool.submit(call_llm, client, messages, max_tokens, 3)] = idx
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                outputs[idx] = fut.result()
            except Exception as exc:  # noqa: BLE001
                fallback = '<think>API call failed; choose look to recover.</think>\n<action>look</action>'
                outputs[idx] = (fallback, {'error': f'{type(exc).__name__}: {exc}'})
    return outputs


TASK_STOP_WORDS = {
    'a', 'an', 'and', 'in', 'into', 'on', 'onto', 'put', 'find', 'two', 'some', 'the',
    'with', 'to', 'of', 'them', 'it', 'then', 'look', 'at', 'under', 'hot', 'clean',
    'cool', 'heat', 'slice', 'examine', 'task', 'your', 'is', 'that', 'this', 'from',
}
CONTAINER_PREFIXES = (
    'cabinet', 'drawer', 'safe', 'box', 'fridge', 'microwave', 'garbagecan', 'toilet', 'sinkbasin'
)


def task_terms(task: str) -> list[str]:
    terms = []
    for token in re.split(r'[^A-Za-z0-9]+', str(task or '').lower()):
        if len(token) >= 3 and token not in TASK_STOP_WORDS and token not in terms:
            terms.append(token)
    return terms


def split_seen_items(text: str) -> list[str]:
    text = str(text or '').replace('\n', ' ')
    matches = re.findall(r'you see (.*?)(?:\.|$)', text, flags=re.I)
    items: list[str] = []
    for match in matches:
        cleaned = re.sub(r'\b(?:a|an|the)\b\s+', '', match.lower())
        parts = re.split(r',|\band\b', cleaned)
        for part in parts:
            item = part.strip(" .;:'\"()[]")
            if item and item != 'nothing' and len(item) <= 60 and item not in items:
                items.append(item)
    return items


def extract_initial_candidates(state: EpisodeState) -> list[str]:
    if not state.trace:
        return []
    initial = str(state.trace[0].get('obs') or '')
    items = split_seen_items(initial)
    candidates = []
    for item in items:
        item = re.sub(r'\s+', ' ', item).strip()
        if item and item not in candidates:
            candidates.append(item)
    return candidates




def is_target_related(action: str, text: str, terms: list[str]) -> bool:
    combined = f'{action}\n{text}'.lower()
    return any(term in combined for term in terms)


def dedupe_action_path(actions: list[str]) -> list[str]:
    out: list[str] = []
    for action in actions:
        action = str(action or '').strip().lower()
        if not action:
            continue
        if out and out[-1] == action:
            continue
        # Drop immediate undo loops like take X from Y right after move X to Y only when same full action repeats later.
        out.append(action)
    return out


def base_name(name: str) -> str:
    return re.sub(r'\s+\d+$', '', re.sub(r'\s+', ' ', str(name or '').strip().lower()))


def parse_task_intent(task: str) -> dict[str, Any]:
    lower_task = str(task or '').lower()
    object_terms: list[str] = []
    tool_terms: list[str] = []
    target_location = ''

    for pattern in (
        r'find two\s+([a-z0-9_]+)\b',
        r'put\s+(?:some|a|an|two)\s+([a-z0-9_]+)\b',
        r'(?:cool|heat|clean|slice)\s+(?:some|a|an|the)?\s*([a-z0-9_]+)\b',
        r'examine\s+the\s+([a-z0-9_]+)\s+with\s+the\s+([a-z0-9_]+)\b',
    ):
        match = re.search(pattern, lower_task)
        if not match:
            continue
        if match.group(1) and match.group(1) not in object_terms:
            object_terms.append(match.group(1))
        if len(match.groups()) > 1 and match.group(2) and match.group(2) not in tool_terms:
            tool_terms.append(match.group(2))
        break

    target_match = re.search(r'put\s+(?:them|it|.+?)\s+(?:in|on|into|onto)\s+([a-z0-9_]+)\b', lower_task)
    if target_match:
        target_location = target_match.group(1)

    if not object_terms:
        for term in task_terms(task):
            if term != target_location and term not in object_terms:
                object_terms.append(term)
                break
    return {
        'object_terms': object_terms,
        'tool_terms': tool_terms,
        'target_location': target_location,
    }


def action_matches_terms(name: str, terms: list[str]) -> bool:
    normalized = base_name(name)
    return any(term == normalized or term in normalized for term in terms)


def parse_take_action(action: str) -> tuple[str, str] | None:
    match = re.match(r'^take\s+(.+?)\s+from\s+(.+)$', str(action or '').strip().lower())
    if not match:
        return None
    return match.group(1).strip(), match.group(2).strip()


def parse_move_action(action: str) -> tuple[str, str] | None:
    match = re.match(r'^(?:move|put)\s+(.+?)\s+to\s+(.+)$', str(action or '').strip().lower())
    if not match:
        return None
    return match.group(1).strip(), match.group(2).strip()


def build_success_plan_evidence(state: EpisodeState, obs_chars: int, feedback_chars: int) -> str:
    intent = parse_task_intent(state.task)
    object_terms = list(intent['object_terms'])
    tool_terms = list(intent['tool_terms'])
    target_location = str(intent['target_location'])
    opened: set[str] = set()
    object_states: dict[str, dict[str, Any]] = {}
    critical_events: list[str] = []
    undo_events: list[str] = []
    repeated_counter: dict[str, int] = {}
    terminal_event = 'unknown'

    for item in state.trace:
        step = int(item.get('step') or 0)
        action = str(item.get('projected_action') or item.get('action') or '').strip().lower()
        feedback = str(item.get('feedback') or '')
        combined = f"{item.get('obs', '')}\n{action}\n{feedback}".lower()
        if action:
            repeated_counter[action] = repeated_counter.get(action, 0) + 1
        if action.startswith('open '):
            opened.add(action[len('open '):].strip())

        take = parse_take_action(action)
        move = parse_move_action(action)
        if take:
            object_name, source = take
            if action_matches_terms(object_name, object_terms):
                entry = object_states.setdefault(object_name, {
                    'first_source': '',
                    'last_location': '',
                    'last_step': step,
                    'moves_to_target': [],
                })
                entry['last_location'] = 'inventory'
                entry['last_step'] = step
                if target_location and base_name(source) == target_location:
                    undo_events.append(f'S{step}: {action}; took a placed target object back out')
                elif not entry.get('first_source'):
                    entry['first_source'] = source
                critical_events.append(f'S{step}: {action}; feedback={compact(feedback, feedback_chars)}')
        elif move:
            object_name, destination = move
            if action_matches_terms(object_name, object_terms):
                entry = object_states.setdefault(object_name, {
                    'first_source': '',
                    'last_location': '',
                    'last_step': step,
                    'moves_to_target': [],
                })
                entry['last_location'] = destination
                entry['last_step'] = step
                if target_location and base_name(destination) == target_location:
                    entry['moves_to_target'].append((step, action))
                critical_events.append(f'S{step}: {action}; feedback={compact(feedback, feedback_chars)}')
        elif re.match(r'^(use|examine|cool|heat|clean|slice)\b', action) and any(term in combined for term in object_terms + tool_terms):
            critical_events.append(f'S{step}: {action}; feedback={compact(feedback, feedback_chars)}')

        if bool(item.get('done')) or float(item.get('reward') or 0.0) > 0.0:
            terminal_event = f'S{step}: {action}; reward={item.get("reward")}; feedback={compact(feedback, feedback_chars)}'

    completed = []
    if target_location:
        for object_name, entry in object_states.items():
            last_location = str(entry.get('last_location') or '')
            if base_name(last_location) != target_location:
                continue
            source = str(entry.get('first_source') or 'unknown')
            retry_steps = []
            if source != 'unknown':
                retry_steps.append(f'go to {source}')
                if source in opened:
                    retry_steps.append(f'open {source}')
                retry_steps.append(f'take {object_name} from {source}')
            retry_steps.append(f'go to {last_location}')
            retry_steps.append(f'move {object_name} to {last_location}')
            completed.append({
                'object': object_name,
                'source': source,
                'target': last_location,
                'last_step': int(entry.get('last_step') or 0),
                'retry': ' -> '.join(retry_steps),
            })
        completed.sort(key=lambda row: row['last_step'])

    loops = [f'{action} x{count}' for action, count in sorted(repeated_counter.items(), key=lambda pair: (-pair[1], pair[0])) if count >= 3]
    lines = [
        'Target terms: ' + (', '.join(task_terms(state.task)) or 'unknown'),
        'Parsed intent: objects=' + (', '.join(object_terms) or 'unknown')
        + '; final_receptacle=' + (target_location or 'none')
        + '; tool=' + (', '.join(tool_terms) or 'none'),
        'Outcome-aware note: traj1 succeeded; summarize only the final working recipe, not earlier exploration.',
    ]
    if completed:
        lines.append('Completed final placements:')
        for row in completed[:4]:
            lines.append(
                f"- {row['object']}: source={row['source']} -> final target={row['target']} at S{row['last_step']}; retry={row['retry']}"
            )
    lines.append('Terminal reward event: ' + terminal_event)
    lines.append('Critical target/tool operations:')
    for event in critical_events[-14:]:
        lines.append('- ' + event)
    if undo_events:
        lines.append('Undo actions that should NOT be copied: ' + '; '.join(undo_events[:8]))
    if loops:
        lines.append('Likely loops to avoid: ' + '; '.join(loops[:8]))
    return compact('\n'.join(lines), 4200)


def build_adaptive_evidence(state: EpisodeState, max_trace_steps: int, obs_chars: int, feedback_chars: int) -> str:
    if not state.won:
        return build_structured_evidence(state, max_trace_steps, obs_chars, feedback_chars)
    return build_success_plan_evidence(state, obs_chars, feedback_chars)

def build_structured_evidence(state: EpisodeState, max_trace_steps: int, obs_chars: int, feedback_chars: int) -> str:
    terms = task_terms(state.task)
    visited: list[str] = []
    opened: list[str] = []
    target_events: list[str] = []
    manipulation_events: list[str] = []
    empty_events: list[str] = []
    bad_events: list[str] = []
    all_actions: list[str] = []
    repeated_counter: dict[str, int] = {}

    for item in state.trace:
        step = int(item.get('step') or 0)
        action = str(item.get('projected_action') or item.get('action') or '').strip().lower()
        obs = str(item.get('obs') or '')
        feedback = str(item.get('feedback') or '')
        combined = f'{obs}\n{action}\n{feedback}'.lower()
        if action:
            all_actions.append(action)
            repeated_counter[action] = repeated_counter.get(action, 0) + 1
        if action.startswith('go to '):
            loc = action[len('go to '):].strip()
            if loc and loc not in visited:
                visited.append(loc)
        if action.startswith('open '):
            loc = action[len('open '):].strip()
            if loc and loc not in opened:
                opened.append(loc)
        if any(term in combined for term in terms):
            excerpt = compact(feedback or obs, feedback_chars)
            target_events.append(f'S{step}: {action}; seen/feedback: {excerpt}')
        if re.match(r'^(take|move|put|use|cool|heat|clean|slice|examine)\b', action):
            manipulation_events.append(f'S{step}: {action}; reward={item.get("reward")} done={bool(item.get("done"))}')
        if re.search(r'you see nothing|empty|drawer \d+ is closed|cabinet \d+ is closed', feedback.lower()):
            empty_events.append(f'S{step}: {action}; {compact(feedback, 160)}')
        if not bool(item.get('valid', True)):
            bad_events.append(f'S{step}: invalid {action}')

    initial_candidates = extract_initial_candidates(state)
    unvisited = [cand for cand in initial_candidates if cand not in visited]
    repeated = [f'{act} x{cnt}' for act, cnt in sorted(repeated_counter.items(), key=lambda kv: (-kv[1], kv[0])) if cnt >= 3]
    recent = all_actions[-max_trace_steps:] if max_trace_steps > 0 else all_actions
    success_path = all_actions if state.won else []

    if state.won and len(success_path) > 28:
        success_path = success_path[:18] + ['...'] + success_path[-8:]

    lines = []
    lines.append('Target terms: ' + (', '.join(terms) or 'unknown'))
    lines.append('Visited places: ' + (', '.join(visited[:30]) or 'none'))
    lines.append('Opened containers: ' + (', '.join(opened[:20]) or 'none'))
    lines.append('Unvisited initial places: ' + (', '.join(unvisited[:18]) or 'none'))
    if state.won:
        lines.append('Successful action path: ' + ' -> '.join(success_path))
    lines.append('Target-related evidence:')
    for event in target_events[:14]:
        lines.append('- ' + event)
    if len(target_events) > 14:
        lines.append(f'- ... {len(target_events) - 14} more target-related events')
    lines.append('Manipulation/action evidence:')
    for event in manipulation_events[:18]:
        lines.append('- ' + event)
    if len(manipulation_events) > 18:
        lines.append(f'- ... {len(manipulation_events) - 18} more manipulation events')
    lines.append('Empty or low-value evidence:')
    for event in empty_events[:10]:
        lines.append('- ' + event)
    if repeated:
        lines.append('Repeated action loops: ' + '; '.join(repeated[:10]))
    if bad_events:
        lines.append('Invalid actions: ' + '; '.join(bad_events[:8]))
    lines.append('Recent action tail: ' + (' -> '.join(recent) or 'none'))
    return compact('\n'.join(lines), 5200)


def build_raw_trace_evidence(state: EpisodeState, max_trace_steps: int, obs_chars: int, feedback_chars: int) -> str:
    selected = state.trace[-max_trace_steps:] if max_trace_steps > 0 else state.trace
    lines = []
    for item in selected:
        lines.extend([
            f"Step {item['step']}",
            'Obs: ' + compact(item.get('obs', ''), obs_chars),
            'Action: ' + str(item.get('projected_action') or item.get('action') or ''),
            'Valid: ' + str(bool(item.get('valid'))),
            'Reward: ' + str(item.get('reward')),
            'Done: ' + str(bool(item.get('done'))),
            'Feedback: ' + compact(item.get('feedback', ''), feedback_chars),
            '',
        ])
    return '\n'.join(lines).strip() or '(empty)'


def build_summary_prompt(
    state: EpisodeState,
    max_trace_steps: int,
    obs_chars: int,
    feedback_chars: int,
    context_mode: str = 'structured',
) -> str:
    evidence = build_raw_trace_evidence(state, max_trace_steps, obs_chars, feedback_chars)
    if context_mode == 'structured':
        evidence = build_structured_evidence(state, max_trace_steps, obs_chars, feedback_chars)
    elif context_mode == 'adaptive':
        evidence = build_adaptive_evidence(state, max_trace_steps, obs_chars, feedback_chars)
    elif context_mode == 'hybrid':
        evidence = (
            '[Structured evidence]\n'
            + build_structured_evidence(state, max_trace_steps, obs_chars, feedback_chars)
            + '\n\n[Recent raw trace]\n'
            + build_raw_trace_evidence(state, max_trace_steps, obs_chars, feedback_chars)
        )
        evidence = compact(evidence, 6500)
    status = 'SUCCEEDED' if state.won else 'FAILED'
    return SUMMARY_TEMPLATE.format(
        task=state.task,
        status=status,
        reward=f'{state.reward:.1f}',
        steps=state.steps,
        invalid=state.invalid_actions,
        trace=evidence,
    )


def append_retry_memory(prompt: str, memory: str) -> str:
    memory = compact(memory, 900)
    return (
        f'{prompt}\n\n'
        '[Retry memory from previous attempt]\n'
        f'{memory}\n\n'
        'Treat the retry memory as route hints, not as the current state. Choose exactly one admissible action now. '
        'If the memory says take/move but that action is not admissible, first navigate/open until it is admissible. '
        'Never take an object back out of the final target after placing it unless the task explicitly requires moving it again.'
    )


def run_phase(
    *,
    phase: str,
    env_manager: AlfWorldEnvironmentManager,
    client: LLMClient,
    obs: dict[str, Any],
    states: list[EpisodeState],
    max_steps: int,
    max_workers: int,
    action_max_tokens: int,
    retry_memories: list[str] | None = None,
) -> tuple[dict[str, Any], list[EpisodeState]]:
    env_num = len(states)
    for step_idx in range(max_steps):
        active = [i for i, st in enumerate(states) if not st.done]
        if not active:
            break
        prompts = list(obs.get('text') or [''] * env_num)
        if retry_memories is not None:
            prompts = [append_retry_memory(prompts[i], retry_memories[i]) for i in range(env_num)]
        llm_outputs = call_actions_parallel(
            client=client,
            prompts=prompts,
            active_indices=active,
            max_workers=max_workers,
            max_tokens=action_max_tokens,
        )
        raw_actions = []
        llm_meta_by_idx: dict[int, dict[str, Any]] = {}
        for i in range(env_num):
            if states[i].done:
                raw_actions.append('<think>done</think>\n<action>look</action>')
            else:
                text, meta = llm_outputs.get(i, ('<think>missing output</think>\n<action>look</action>', {'error': 'missing'}))
                raw_actions.append(text)
                llm_meta_by_idx[i] = meta

        prev_obs_text = list(obs.get('anchor') or obs.get('text') or [''] * env_num)
        raw_actions_for_trace = list(raw_actions)
        next_obs, rewards, dones, infos = env_manager.step(raw_actions)
        projected_actions = list(raw_actions)
        next_anchor = list(next_obs.get('anchor') or next_obs.get('text') or [''] * env_num)
        rewards_np = np.asarray(rewards).reshape(-1)
        dones_np = np.asarray(dones).reshape(-1)

        for i in active:
            info = infos[i] if i < len(infos) and isinstance(infos[i], dict) else {}
            valid = bool(info.get('is_action_valid', False))
            if not valid:
                states[i].invalid_actions += 1
            reward = float(rewards_np[i])
            states[i].reward += reward
            states[i].steps += 1
            states[i].final_info = dict(info)
            projected_action = projected_actions[i] if i < len(projected_actions) else get_action_text(raw_actions_for_trace[i])
            states[i].trace.append({
                'phase': phase,
                'step': step_idx + 1,
                'obs': prev_obs_text[i],
                'prompt_excerpt': compact(prompts[i], 1800),
                'raw_action': raw_actions_for_trace[i],
                'projected_action': projected_action,
                'valid': valid,
                'reward': reward,
                'done': bool(dones_np[i]),
                'won': infer_won(info, states[i].reward),
                'feedback': next_anchor[i],
                'llm_meta': llm_meta_by_idx.get(i, {}),
            })
            if bool(dones_np[i]):
                states[i].done = True
                states[i].won = infer_won(info, states[i].reward)
        obs = next_obs
        print(
            f"[PHASE {phase}] step={step_idx + 1}/{max_steps} "
            f"done={sum(st.done for st in states)}/{env_num} "
            f"success={sum(st.won for st in states)}/{env_num}",
            flush=True,
        )
    for st in states:
        if not st.done:
            st.won = infer_won(st.final_info, st.reward)
    return obs, states


def make_env(env_num: int, seed: int, ray_num_cpus: int, worker_cpus: float) -> AlfWorldEnvironmentManager:
    os.environ.setdefault('ALFWORLD_DATA', str(DEFAULT_ALFWORLD_DATA))
    import ray  # noqa: PLC0415
    if ray.is_initialized():
        ray.shutdown()
    ray.init(num_cpus=ray_num_cpus, ignore_reinit_error=True, include_dashboard=False, logging_level='ERROR')
    cfg = SimpleNamespace(env=SimpleNamespace(history_length=2))
    envs = build_alfworld_envs(
        str(ALF_CONFIG),
        seed=seed,
        env_num=env_num,
        group_n=1,
        resources_per_worker={'num_cpus': worker_cpus, 'num_gpus': 0.0},
        is_train=True,
        env_kwargs={},
    )
    return AlfWorldEnvironmentManager(envs, alfworld_projection, cfg)


def write_report(out_dir: Path, records: list[dict[str, Any]], config: dict[str, Any]) -> None:
    n = len(records)
    traj1_sr = sum(1 for r in records if r['traj1']['won']) / max(1, n)
    traj2_sr = sum(1 for r in records if r['traj2']['won']) / max(1, n)
    match_rate = sum(1 for r in records if r['task_match']) / max(1, n)
    improved = sum(1 for r in records if (not r['traj1']['won']) and r['traj2']['won'])
    degraded = sum(1 for r in records if r['traj1']['won'] and (not r['traj2']['won']))
    lines = [
        '# ALFWorld Traj-Refine 10-Task Probe',
        '',
        '## Config',
        '',
        '```json',
        json.dumps(config, ensure_ascii=False, indent=2),
        '```',
        '',
        '## Metrics',
        '',
        f'- tasks: {n}',
        f'- traj1_success_rate: {traj1_sr:.4f} ({sum(1 for r in records if r["traj1"]["won"])}/{n})',
        f'- traj2_success_rate: {traj2_sr:.4f} ({sum(1 for r in records if r["traj2"]["won"])}/{n})',
        f'- task_match_rate_after_retry_reset: {match_rate:.4f}',
        f'- improved_failed_to_success: {improved}',
        f'- degraded_success_to_failed: {degraded}',
        '',
        '## Per Task',
        '',
        '| id | task | match | traj1 | traj2 | r1 | r2 | len1 | len2 | summary |',
        '|---:|---|---:|---:|---:|---:|---:|---:|---:|---|',
    ]
    for r in records:
        summary = str(r.get('summary', '')).replace('\n', '<br>')
        if len(summary) > 360:
            summary = summary[:360] + '...'
        lines.append(
            f"| {r['task_id']} | `{r['task']}` | {int(r['task_match'])} | {int(r['traj1']['won'])} | "
            f"{int(r['traj2']['won'])} | {r['traj1']['reward']:.1f} | {r['traj2']['reward']:.1f} | "
            f"{r['traj1']['steps']} | {r['traj2']['steps']} | {summary} |"
        )
    lines.extend([
        '',
        '## Trace Pointers',
        '',
        '- Full JSONL records: `records.jsonl`',
        '- Compact metrics: `metrics.json`',
    ])
    (out_dir / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    metrics = {
        'tasks': n,
        'traj1_success_rate': traj1_sr,
        'traj2_success_rate': traj2_sr,
        'task_match_rate_after_retry_reset': match_rate,
        'improved_failed_to_success': improved,
        'degraded_success_to_failed': degraded,
    }
    (out_dir / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description='Closed-LLM ALFWorld traj1-summary-traj2 refinement probe on train tasks.')
    parser.add_argument('--num-tasks', type=int, default=int(os.environ.get('NUM_TASKS', '10')))
    parser.add_argument('--seed', type=int, default=int(os.environ.get('SEED', '2026')))
    parser.add_argument('--max-steps', type=int, default=int(os.environ.get('MAX_STEPS', '50')))
    parser.add_argument('--model', default=os.environ.get('MODEL', 'azure::gpt-5.4-mini'))
    parser.add_argument('--base-url', default=os.environ.get('BASE_URL') or os.environ.get('QGENIE_API_ENDPOINT') or 'https://qgenie-api.qualcomm.com/v1')
    parser.add_argument('--api-key-env', default=os.environ.get('API_KEY_ENV', 'QGENIE_API_KEY'))
    parser.add_argument('--env-file', default='/prj/corp/crd/morpheus/lasvegas/china-scratch/ziyuhuan/Projects/DataFactory_SML/SML/sml/.env')
    parser.add_argument('--out-dir', default='')
    parser.add_argument('--max-workers', type=int, default=int(os.environ.get('MAX_WORKERS', '2')))
    parser.add_argument('--action-max-tokens', type=int, default=int(os.environ.get('ACTION_MAX_TOKENS', '180')))
    parser.add_argument('--summary-max-tokens', type=int, default=int(os.environ.get('SUMMARY_MAX_TOKENS', '180')))
    parser.add_argument('--summary-trace-steps', type=int, default=int(os.environ.get('SUMMARY_TRACE_STEPS', '8')))
    parser.add_argument('--summary-obs-chars', type=int, default=int(os.environ.get('SUMMARY_OBS_CHARS', '220')))
    parser.add_argument('--summary-feedback-chars', type=int, default=int(os.environ.get('SUMMARY_FEEDBACK_CHARS', '260')))
    parser.add_argument('--summary-context-mode', choices=['structured', 'adaptive', 'raw_tail', 'hybrid'], default=os.environ.get('SUMMARY_CONTEXT_MODE', 'adaptive'))
    parser.add_argument('--ray-num-cpus', type=int, default=int(os.environ.get('RAY_NUM_CPUS', '16')))
    parser.add_argument('--worker-cpus', type=float, default=float(os.environ.get('ENV_WORKER_CPUS', '0.2')))
    args = parser.parse_args()

    load_env_file(Path(args.env_file))
    api_key = os.environ.get(args.api_key_env) or os.environ.get('OPENAI_API_KEY') or ''
    if not api_key:
        raise SystemExit(f'Missing API key: set {args.api_key_env} or use --env-file')
    if not Path(os.environ.get('ALFWORLD_DATA', str(DEFAULT_ALFWORLD_DATA)), 'json_2.1.1').exists():
        os.environ['ALFWORLD_DATA'] = str(DEFAULT_ALFWORLD_DATA)

    run_tag = time.strftime('%Y%m%d_%H%M%S')
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / 'EXPS/alfworld_effective_summary/outputs' / f'traj_refine_train10_{args.model.replace("::", "_").replace("/", "_")}_{run_tag}'
    out_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config['out_dir'] = str(out_dir)
    config['alfworld_data'] = os.environ.get('ALFWORLD_DATA')
    (out_dir / 'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')

    print(f'[INFO] out_dir={out_dir}', flush=True)
    print(f'[INFO] model={args.model} num_tasks={args.num_tasks} max_steps={args.max_steps}', flush=True)
    env_manager = make_env(args.num_tasks, args.seed, args.ray_num_cpus, args.worker_cpus)
    client = LLMClient(model=args.model, api_key=api_key, base_url=args.base_url, timeout=180, verify_ssl=False)

    try:
        obs1, infos1 = env_manager.reset({})
        states1 = []
        for i in range(args.num_tasks):
            raw = (obs1.get('anchor') or obs1.get('text'))[i]
            info = infos1[i] if i < len(infos1) else {}
            states1.append(EpisodeState(
                task_id=i,
                task=extract_task(str(raw)),
                gamefile=str(info.get('extra.gamefile') or ''),
            ))
        _, states1 = run_phase(
            phase='traj1',
            env_manager=env_manager,
            client=client,
            obs=obs1,
            states=states1,
            max_steps=args.max_steps,
            max_workers=args.max_workers,
            action_max_tokens=args.action_max_tokens,
            retry_memories=None,
        )

        summaries = []
        summary_prompts = []
        for st in states1:
            prompt = build_summary_prompt(st, args.summary_trace_steps, args.summary_obs_chars, args.summary_feedback_chars, args.summary_context_mode)
            summary_prompts.append(prompt)
        with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            futures = []
            for prompt in summary_prompts:
                futures.append(pool.submit(
                    call_llm,
                    client,
                    [{'role': 'system', 'content': SUMMARY_SYSTEM}, {'role': 'user', 'content': prompt}],
                    args.summary_max_tokens,
                    3,
                ))
            for fut in futures:
                text, meta = fut.result()
                summaries.append({'text': text, 'meta': meta})
        (out_dir / 'summary_prompts.jsonl').write_text(
            ''.join(json.dumps({'task_id': i, 'prompt': p, 'summary': summaries[i]['text'], 'meta': summaries[i]['meta']}, ensure_ascii=False) + '\n' for i, p in enumerate(summary_prompts)),
            encoding='utf-8',
        )

        obs2, infos2 = env_manager.reset({'retry_same_seed': True})
        states2 = []
        task_matches = []
        for i in range(args.num_tasks):
            raw = (obs2.get('anchor') or obs2.get('text'))[i]
            task2 = extract_task(str(raw))
            info = infos2[i] if i < len(infos2) else {}
            states2.append(EpisodeState(
                task_id=i,
                task=task2,
                gamefile=str(info.get('extra.gamefile') or ''),
            ))
            task_matches.append(task2 == states1[i].task)
        _, states2 = run_phase(
            phase='traj2',
            env_manager=env_manager,
            client=client,
            obs=obs2,
            states=states2,
            max_steps=args.max_steps,
            max_workers=args.max_workers,
            action_max_tokens=args.action_max_tokens,
            retry_memories=[s['text'] for s in summaries],
        )

        records = []
        for i in range(args.num_tasks):
            records.append({
                'task_id': i,
                'task': states1[i].task,
                'traj2_task': states2[i].task,
                'task_match': bool(task_matches[i]),
                'summary': summaries[i]['text'],
                'summary_prompt': summary_prompts[i],
                'traj1': {
                    'won': bool(states1[i].won),
                    'reward': float(states1[i].reward),
                    'steps': int(states1[i].steps),
                    'invalid_actions': int(states1[i].invalid_actions),
                    'gamefile': states1[i].gamefile,
                    'trace': states1[i].trace,
                },
                'traj2': {
                    'won': bool(states2[i].won),
                    'reward': float(states2[i].reward),
                    'steps': int(states2[i].steps),
                    'invalid_actions': int(states2[i].invalid_actions),
                    'gamefile': states2[i].gamefile,
                    'trace': states2[i].trace,
                },
            })
        with (out_dir / 'records.jsonl').open('w', encoding='utf-8') as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        write_report(out_dir, records, config)
        print('[DONE]', out_dir, flush=True)
        print((out_dir / 'metrics.json').read_text(encoding='utf-8'), flush=True)
    finally:
        try:
            env_manager.envs.close()
        except Exception:
            pass
        try:
            import ray
            ray.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
