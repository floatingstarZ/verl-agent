#!/usr/bin/env python3
"""Rerun only the reasoning-value branch on saved ALFWorld action trajectories."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
import zipfile
from copy import deepcopy
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from agent_system.multi_turn_rollout.reasoning_value_rollout import (
    ALFWORLD_ACTOR_ALIGNED_REASONING_VALUE_PROMPT,
    ALFWORLD_ACTOR_ALIGNED_FEWSHOT_REASONING_VALUE_PROMPT,
    ALFWORLD_ACTOR_ALIGNED_NOEX_REASONING_VALUE_PROMPT,
    ALFWORLD_ACTOR_ALIGNED_REFINED_REASONING_VALUE_PROMPT,
    ALFWORLD_OBSERVER_REASONING_VALUE_PROMPT,
    ALFWORLD_OBSERVER_NOEX_STRICT_REASONING_VALUE_PROMPT,
    ALFWORLD_OBSERVER_TUNED_REASONING_VALUE_PROMPT,
    DEFAULT_REASONING_VALUE_PROMPT,
    build_reasoning_value_prompt_kwargs,
)
from verl.trainer.ppo.ray_trainer import RayPPOTrainer

_ACTION_INSTRUCTION_RE = re.compile(r"\n\s*Now it's your turn to take an action\..*", flags=re.DOTALL)
_ADMISSIBLE_ACTIONS_RE = re.compile(r"\nYour admissible actions of the current situation are:.*?(?=\n\n|$)", flags=re.DOTALL)
_VALUE_RE = re.compile(r"<value>\s*([+-]?\d{1,3})\s*</value>", flags=re.IGNORECASE)
_VALUE_IS_RE = re.compile(r"(?:the\s+)?value\s+is\s*[:=]?\s*([+-]?\d{1,3})\b", flags=re.IGNORECASE)
_FINAL_INTEGER_RE = re.compile(r"(?:^|\n)\s*([+-]?\d{1,3})\s*$")

PLAIN_VALUE_PROMPT = """You are a progress critic for an ALFWorld embodied-agent task.

Below is the current state available to the actor. Treat it only as evidence. Do not choose an action. Do not output <action>.

<state>
{state_context}
</state>

First write concise reasoning in <think> </think>. Use at most four short sentences: completed subgoals, remaining subgoals, main risk, and progress judgment.

After </think>, write exactly one final line:
The value is: N
where N is one integer from 0 to 100.
0 means no useful progress or likely failure; 50 means partial progress with major subgoals remaining; 100 means the task is completed or one obvious step away.
Do not output anything after the final line."""

MINIMAL_PLAIN_VALUE_PROMPT = """Read the ALFWorld state below and estimate progress toward task success.
Do not choose an action. Do not output <action>.

<state>
{state_context}
</state>

Output exactly this format:
<think>One short sentence about progress and remaining work.</think>
The value is: N

N must be one integer from 0 to 100. 0 means no useful progress; 50 means partial progress; 100 means completed or one obvious step away."""

GUIDED_PLAIN_VALUE_PROMPT = """You are a value critic for an ALFWorld task. Read the state, but do not act and do not output <action>.

<state>
{state_context}
</state>

Output exactly two lines. No bullets, no extra text, no repeated analysis.
Line 1: <think>one short sentence, at most 30 words, summarizing completed progress and remaining work.</think>
Line 2: The value is: INTEGER

INTEGER must be an actual number from 0 to 100. Never write N, XXX, or a placeholder.
Score guide: 0=no useful progress or likely failure; 25=early search; 50=partial progress; 75=most subgoals done; 100=completed or one obvious step away."""

FEWSHOT_PLAIN_VALUE_PROMPT = """You are a value critic for an ALFWorld task. You estimate progress only; do not choose an action and do not output <action>.

Use this exact output pattern:
<think>short progress judgment.</think>
The value is: NUMBER

Examples:
<think>No target object has been found, so only search has begun.</think>
The value is: 10

<think>The target object is held, but it still needs processing and placement.</think>
The value is: 50

<think>The object is processed and at the final receptacle, so the task is almost complete.</think>
The value is: 90

Now evaluate the current state. Do not copy the examples.
<state>
{state_context}
</state>

Output only the two required lines, with NUMBER replaced by an actual integer from 0 to 100."""

STRICT_PLAIN_VALUE_PROMPT = """Read this ALFWorld state and score task progress. Do not act.

<state>
{state_context}
</state>

You must finish the answer. First close the think tag, then immediately give the score.
Required format:
<think>Progress: one concise sentence.</think>
The value is: 42

Replace 42 with your own integer score from 0 to 100. Do not write any other final line. If uncertain, choose your best integer and still output the final line."""

COMPACT_PLAIN_VALUE_PROMPT = """You are a progress critic for an ALFWorld embodied-agent task.
Read the state as evidence only. Do not choose an action. Do not output <action>.

<state>
{state_context}
</state>

Your answer must start with <think> and must contain a closing </think>.
Inside <think>, write one concise sentence under 40 words about progress and remaining work.
After </think>, immediately write one line that starts with The value is: followed by your actual integer score from 0 to 100.

Use 0 for no useful progress or likely failure, 25 for early search, 50 for partial progress, 75 for most subgoals done, and 100 for completed or one obvious step away.
Never write placeholders such as N, NUMBER, INTEGER, brackets, or XXX."""

FINAL_INT_COMPACT_PROMPT = """You are a progress critic for an ALFWorld embodied-agent task.
Read the state as evidence only. Do not choose an action. Do not output <action>.

<state>
{state_context}
</state>

Your answer must start with <think> and must contain a closing </think>.
Inside <think>, write one concise sentence under 40 words about completed progress, remaining work, and whether the agent seems stuck.
After </think>, write only the integer score from 0 to 100 on the next line.
Do not write any words, labels, punctuation, or XML tags after </think>.

Use completed facts, not optimistic plans.
Use 0 for no useful progress or likely failure, 25 for early search, 50 for partial progress, 75 for most subgoals done, and 100 for completed or one obvious step away.

Example:
<think>Tomato is held but not heated yet, so progress is real but major work remains.</think>
50"""

FINAL_INT_GUARDED_PROMPT = """You are a progress critic for an ALFWorld embodied-agent task.
Read the state as evidence only. Do not choose an action. Do not output <action>.

<state>
{state_context}
</state>

The first characters of your answer must be <think>.
Your answer must contain exactly one closing </think>.
Do not put any number before </think>.
After </think>, the next non-whitespace characters must be the integer score from 0 to 100.
Do not write any label such as Value, Score, or The value is.

Inside <think>, write one short sentence about:
1. what is already completed,
2. what still remains,
3. whether the agent is making progress or is stuck.

Score by completed facts only:
0=no useful progress or likely failure
25=searching
50=partial progress
75=almost done
100=done or one obvious valid action away

Example output:
<think>The object is held, but it still needs processing and final placement.</think>
50"""

FINAL_INT_TWO_LINE_PROMPT = """You are a progress critic for an ALFWorld embodied-agent task.
Read the state as evidence only. Do not choose an action. Do not output <action>.

<state>
{state_context}
</state>

Return exactly two lines and nothing else.
Line 1 must be a complete <think>...</think> block.
Line 2 must be only one integer from 0 to 100.

Use one short sentence in <think> describing completed progress, remaining work, and whether the current trajectory appears stuck.
Base the score on completed facts, not on possible future plans.
0=no useful progress or likely failure
25=early search
50=partial progress
75=close to success
100=completed or one obvious valid action away

Example:
<think>The target object is found and held, but it still needs one major processing step.</think>
50"""

FINAL_INT_USER_CLEAN_PROMPT = """You are a progress critic for an ALFWorld task.
Read the state as evidence only. Your role is to evaluate progress rather than act.

<state>
{state_context}
</state>

Return exactly two lines and nothing else.

Line 1 must start with <think> and end with </think>.
Inside <think>, write one short sentence about:
what is already done, what still remains, and whether the agent seems making progress or stuck.

Line 2 must be only one integer from 0 to 100.

Interpret the score as overall progress toward eventual task success:
a higher score means more of the task is already completed and less important work remains;
a lower score means little progress, missing key subgoals, or likely failure.

Example:
<think>The target object is found, but one key step remains, so progress is real.</think>
58"""

FINAL_INT_VALUE_PROMPT = """You are a progress critic for an ALFWorld task.
Read the current state. Your job is not to act, but to judge how promising this state is for eventually completing the task.

<state>
{state_context}
</state>

Think briefly about only these facts:
1. What has already been completed?
2. What important subgoals are still unfinished?
3. Is the agent making useful progress, stuck, or going in the wrong direction?

Write one short <think>...</think> block.
After </think>, output only one integer from 0 to 100 on the final line.

Value meaning:
0 = no useful progress, clearly stuck, or likely failure.
25 = early search; the goal is understood but key object/location is not secured.
50 = partial progress; key object/location is found or held, but important steps remain.
75 = close to success; most required subgoals are done, only placement/finishing remains.
100 = task is completed or one obvious valid action away from completion.

Important rules:
- Do not choose an action.
- Do not output <action>.
- Do not list all visible objects or all possible actions.
- Base the score on completed facts, not on a possible future plan.
- The final line must be only the integer score, with no words, labels, XML tags, or punctuation."""

FINAL_INT_STRICT_PROMPT = """You are grading ALFWorld task progress. Do not act.

<state>
{state_context}
</state>

Return exactly two lines:
<think>One concise sentence: completed facts, remaining work, and whether the trajectory seems stuck.</think>
SCORE

SCORE must be replaced by one integer from 0 to 100.
Use only completed facts, not optimistic plans.
0=no useful progress or stuck. 25=searching. 50=object/location secured but major work remains. 75=almost done. 100=done or one obvious valid action away.
Do not write labels like Score, Value, N, or The value is. The second line must be digits only."""

FINAL_INT_FEWSHOT_PROMPT = """You are a progress critic for ALFWorld. Do not choose an action. Do not output <action>.
The last line of your answer must be only digits.

Examples:
<think>The target object has not been found, so the agent is still searching.</think>
10

<think>The object is held, but it still needs processing and final placement.</think>
50

<think>The object is processed and at the final receptacle, so the task is almost complete.</think>
90

Now score the current state using the same format.
<state>
{state_context}
</state>

Base the score on completed facts, not on an optimistic future plan."""


def clean_action_prompt(text: str, drop_admissible_actions: bool = False) -> str:
    text = text or ""
    start = text.find("You are an expert agent operating")
    if start >= 0:
        text = text[start:]
    text = _ACTION_INSTRUCTION_RE.sub("", text)
    if drop_admissible_actions:
        text = _ADMISSIBLE_ACTIONS_RE.sub("\nAdmissible actions are available to the actor but omitted here for brevity.", text)
    return text.strip()


def build_value_prompt(action_prompt: str, prompt_style: str = "xml") -> str:
    compact_styles = {
        "plain_value_guided",
        "plain_value_fewshot",
        "plain_value_strict",
        "plain_value_compact",
        "alfworld_actor_aligned",
        "alfworld_actor_aligned_noex",
        "alfworld_actor_aligned_fewshot",
        "alfworld_actor_aligned_refined",
        "alfworld_observer",
        "alfworld_observer_tuned",
        "alfworld_observer_noex_strict",
        "final_int_compact",
        "final_int_guarded",
        "final_int_two_line",
        "final_int_user_clean",
        "final_int",
        "final_int_strict",
        "final_int_fewshot",
    }
    state_context = clean_action_prompt(action_prompt, drop_admissible_actions=prompt_style in compact_styles)
    if prompt_style == "plain_value":
        return PLAIN_VALUE_PROMPT.format(actor_prompt=action_prompt.strip(), state_context=state_context)
    if prompt_style == "plain_value_minimal":
        return MINIMAL_PLAIN_VALUE_PROMPT.format(actor_prompt=action_prompt.strip(), state_context=state_context)
    if prompt_style == "plain_value_guided":
        return GUIDED_PLAIN_VALUE_PROMPT.format(actor_prompt=action_prompt.strip(), state_context=state_context)
    if prompt_style == "plain_value_fewshot":
        return FEWSHOT_PLAIN_VALUE_PROMPT.format(actor_prompt=action_prompt.strip(), state_context=state_context)
    if prompt_style == "plain_value_strict":
        return STRICT_PLAIN_VALUE_PROMPT.format(actor_prompt=action_prompt.strip(), state_context=state_context)
    if prompt_style == "plain_value_compact":
        return COMPACT_PLAIN_VALUE_PROMPT.format(actor_prompt=action_prompt.strip(), state_context=state_context)
    if prompt_style == "alfworld_actor_aligned":
        return ALFWORLD_ACTOR_ALIGNED_REASONING_VALUE_PROMPT.format(
            **build_reasoning_value_prompt_kwargs(
                action_prompt,
                strip_action_instruction=True,
                observer_include_admissible_actions=False,
                observer_max_admissible_actions=0,
            )
        )
    if prompt_style == "alfworld_actor_aligned_noex":
        return ALFWORLD_ACTOR_ALIGNED_NOEX_REASONING_VALUE_PROMPT.format(
            **build_reasoning_value_prompt_kwargs(
                action_prompt,
                strip_action_instruction=True,
                observer_include_admissible_actions=False,
                observer_max_admissible_actions=0,
            )
        )
    if prompt_style == "alfworld_actor_aligned_fewshot":
        return ALFWORLD_ACTOR_ALIGNED_FEWSHOT_REASONING_VALUE_PROMPT.format(
            **build_reasoning_value_prompt_kwargs(
                action_prompt,
                strip_action_instruction=True,
                observer_include_admissible_actions=False,
                observer_max_admissible_actions=0,
            )
        )
    if prompt_style == "alfworld_actor_aligned_refined":
        return ALFWORLD_ACTOR_ALIGNED_REFINED_REASONING_VALUE_PROMPT.format(
            **build_reasoning_value_prompt_kwargs(
                action_prompt,
                strip_action_instruction=True,
                observer_include_admissible_actions=False,
                observer_max_admissible_actions=0,
            )
        )
    if prompt_style == "alfworld_observer":
        return ALFWORLD_OBSERVER_REASONING_VALUE_PROMPT.format(
            **build_reasoning_value_prompt_kwargs(
                action_prompt,
                strip_action_instruction=True,
                observer_include_admissible_actions=False,
                observer_max_admissible_actions=0,
            )
        )
    if prompt_style == "alfworld_observer_tuned":
        return ALFWORLD_OBSERVER_TUNED_REASONING_VALUE_PROMPT.format(
            **build_reasoning_value_prompt_kwargs(
                action_prompt,
                strip_action_instruction=True,
                observer_include_admissible_actions=False,
                observer_max_admissible_actions=0,
            )
        )
    if prompt_style == "alfworld_observer_noex_strict":
        return ALFWORLD_OBSERVER_NOEX_STRICT_REASONING_VALUE_PROMPT.format(
            **build_reasoning_value_prompt_kwargs(
                action_prompt,
                strip_action_instruction=True,
                observer_include_admissible_actions=False,
                observer_max_admissible_actions=0,
            )
        )
    if prompt_style == "final_int_compact":
        return FINAL_INT_COMPACT_PROMPT.format(actor_prompt=action_prompt.strip(), state_context=state_context)
    if prompt_style == "final_int_guarded":
        return FINAL_INT_GUARDED_PROMPT.format(actor_prompt=action_prompt.strip(), state_context=state_context)
    if prompt_style == "final_int_two_line":
        return FINAL_INT_TWO_LINE_PROMPT.format(actor_prompt=action_prompt.strip(), state_context=state_context)
    if prompt_style == "final_int_user_clean":
        return FINAL_INT_USER_CLEAN_PROMPT.format(actor_prompt=action_prompt.strip(), state_context=state_context)
    if prompt_style == "final_int":
        return FINAL_INT_VALUE_PROMPT.format(actor_prompt=action_prompt.strip(), state_context=state_context)
    if prompt_style == "final_int_strict":
        return FINAL_INT_STRICT_PROMPT.format(actor_prompt=action_prompt.strip(), state_context=state_context)
    if prompt_style == "final_int_fewshot":
        return FINAL_INT_FEWSHOT_PROMPT.format(actor_prompt=action_prompt.strip(), state_context=state_context)
    return DEFAULT_REASONING_VALUE_PROMPT.format(actor_prompt=action_prompt.strip(), state_context=state_context)


def parse_value(text: str):
    text = text or ""
    match = _VALUE_RE.search(text) or _VALUE_IS_RE.search(text)
    if match:
        value = int(match.group(1))
        return max(0, min(100, value))

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    match = re.fullmatch(r"[+-]?\d{1,3}", lines[-1])
    if not match:
        return None
    value = int(match.group(0))
    return max(0, min(100, value))


def iter_steps(cases):
    for case_idx, case in enumerate(cases):
        for step_idx, step in enumerate(case.get("steps", [])):
            yield case_idx, step_idx, step


def write_aux_outputs(cases, output_dir: Path, json_name: str):
    trainer = object.__new__(RayPPOTrainer)
    html_path = output_dir / "reason_value_cases.html"
    trainer._write_reason_value_cases_html(cases, str(html_path), json_name)

    svg_path = output_dir / "overall_value_curve.svg"
    svg_path.write_text(trainer._build_reason_value_overall_curve_svg(cases), encoding="utf-8")

    buckets = {}
    for _, _, step in iter_steps(cases):
        idx = int(step.get("step_index", 0))
        bucket = buckets.setdefault(idx, {"pred": [], "target": [], "reward": [], "valid": [], "text_valid": []})
        bucket["pred"].append(float(step.get("value_pred", 0.0) or 0.0))
        bucket["target"].append(float(step.get("value_target", 0.0) or 0.0))
        bucket["reward"].append(float(step.get("reward", 0.0) or 0.0))
        bucket["valid"].append(1.0 if step.get("is_action_valid") else 0.0)
        bucket["text_valid"].append(1.0 if step.get("value_text_valid") else 0.0)
    with open(output_dir / "overall_value_curve.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step_index", "n", "mean_value_pred", "mean_value_target", "mean_reward", "valid_ratio", "value_text_valid_ratio"])
        for idx in sorted(buckets):
            bucket = buckets[idx]
            n = len(bucket["pred"])
            writer.writerow([
                idx,
                n,
                sum(bucket["pred"]) / n,
                sum(bucket["target"]) / n,
                sum(bucket["reward"]) / n,
                sum(bucket["valid"]) / n,
                sum(bucket["text_valid"]) / n,
            ])

    with open(output_dir / "case_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "case_index", "traj_uid", "success", "episode_reward", "episode_length", "task_type", "task",
            "mean_value_pred", "max_value_pred", "final_value_pred", "value_text_valid_ratio", "first_action", "last_action",
        ])
        for case in cases:
            steps = case.get("steps", [])
            preds = [float(step.get("value_pred", 0.0) or 0.0) for step in steps]
            valid = [1.0 if step.get("value_text_valid") else 0.0 for step in steps]
            writer.writerow([
                case.get("case_index"), case.get("traj_uid"), case.get("success"), case.get("episode_reward"),
                case.get("episode_length"), case.get("task_type"), case.get("task"),
                sum(preds) / len(preds) if preds else 0.0,
                max(preds) if preds else 0.0,
                preds[-1] if preds else 0.0,
                sum(valid) / len(valid) if valid else 0.0,
                steps[0].get("action", "") if steps else "",
                steps[-1].get("action", "") if steps else "",
            ])

    all_steps = [step for _, _, step in iter_steps(cases)]
    valid_count = sum(1 for step in all_steps if step.get("value_text_valid"))
    values = [int(step.get("value_text", 0)) for step in all_steps if step.get("value_text_valid")]
    summary = {
        "num_cases": len(cases),
        "success_cases": sum(1 for case in cases if case.get("success")),
        "total_steps": len(all_steps),
        "value_text_valid_steps": valid_count,
        "value_text_valid_ratio": valid_count / len(all_steps) if all_steps else 0.0,
        "mean_text_value_0_100": sum(values) / len(values) if values else None,
        "min_text_value_0_100": min(values) if values else None,
        "max_text_value_0_100": max(values) if values else None,
    }
    (output_dir / "value_case_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    with zipfile.ZipFile(output_dir / "reason_value_cases_value_only.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in [
            "reason_value_cases.html",
            "reason_value_cases.json",
            "overall_value_curve.svg",
            "overall_value_curve.csv",
            "case_summary.csv",
            "value_case_summary.json",
            "README.txt",
        ]:
            path = output_dir / name
            if path.exists():
                zf.write(path, arcname=name)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.72)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--prompt-style", choices=["xml", "plain_value", "plain_value_minimal", "plain_value_guided", "plain_value_fewshot", "plain_value_strict", "plain_value_compact", "alfworld_actor_aligned", "alfworld_actor_aligned_noex", "alfworld_actor_aligned_fewshot", "alfworld_actor_aligned_refined", "alfworld_observer", "alfworld_observer_tuned", "alfworld_observer_noex_strict", "final_int_compact", "final_int_guarded", "final_int_two_line", "final_int_user_clean", "final_int", "final_int_strict", "final_int_fewshot"], default="xml")
    parser.add_argument("--limit-cases", type=int, default=0)
    parser.add_argument("--limit-steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    input_json = Path(args.input_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(input_json.read_text(encoding="utf-8"))
    cases = deepcopy(data.get("cases", []))
    if args.limit_cases > 0:
        cases = cases[: args.limit_cases]

    step_refs = list(iter_steps(cases))
    if args.limit_steps > 0:
        keep = set((case_idx, step_idx) for case_idx, step_idx, _ in step_refs[: args.limit_steps])
        for case_idx, case in enumerate(cases):
            case["steps"] = [step for step_idx, step in enumerate(case.get("steps", [])) if (case_idx, step_idx) in keep]
        cases = [case for case in cases if case.get("steps")]
        step_refs = list(iter_steps(cases))

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=False)
    prompts = []
    for _, _, step in step_refs:
        prompt_text = build_value_prompt(step.get("action_prompt", ""), prompt_style=args.prompt_style)
        chat_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            add_generation_prompt=True,
            tokenize=False,
        )
        prompts.append(chat_prompt)
        step["old_value_prompt"] = step.get("value_prompt", "")
        step["old_value_response"] = step.get("value_response", "")
        step["old_value_pred"] = step.get("value_pred", None)
        step["value_prompt"] = prompt_text

    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=1.0,
        max_tokens=args.max_tokens,
        stop=["<|im_end|>", "<|endoftext|>"],
    )
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        trust_remote_code=False,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=False,
    )

    start = time.time()
    outputs = []
    for offset in range(0, len(prompts), args.batch_size):
        batch = prompts[offset : offset + args.batch_size]
        outputs.extend(llm.generate(batch, sampling))
        print(f"[progress] generated {len(outputs)}/{len(prompts)}", flush=True)

    valid = 0
    for (_, _, step), output in zip(step_refs, outputs):
        text = output.outputs[0].text.strip()
        parsed = parse_value(text)
        step["value_response"] = text
        step["value_text"] = "" if parsed is None else str(parsed)
        step["value_text_valid"] = parsed is not None
        step["value_pred"] = 0.0 if parsed is None else parsed / 100.0
        if parsed is not None:
            valid += 1

    json_path = output_dir / "reason_value_cases.json"
    json_path.write_text(json.dumps({"cases": cases}, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = write_aux_outputs(cases, output_dir=output_dir, json_name=json_path.name)
    readme = (
        f"Input JSON: {input_json}\n"
        f"Model path: {args.model_path}\n"
        f"Prompt style: {args.prompt_style}\n"
        f"Generated value branch only; action trajectories/rewards are reused from input.\n"
        f"Steps: {len(step_refs)}\n"
        f"Valid parsed values: {valid}/{len(step_refs)}\n"
        f"Elapsed seconds: {time.time() - start:.1f}\n"
        f"HTML: {output_dir / 'reason_value_cases.html'}\n"
        f"JSON: {json_path}\n"
    )
    (output_dir / "README.txt").write_text(readme, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(output_dir / "reason_value_cases.html", flush=True)


if __name__ == "__main__":
    main()
