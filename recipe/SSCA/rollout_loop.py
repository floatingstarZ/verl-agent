import copy
import uuid
from typing import Any, Dict, List, Tuple

import numpy as np

from agent_system.environments import EnvironmentManagerBase
from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector as BaseTrajectoryCollector
from agent_system.multi_turn_rollout.utils import to_list_of_dict, torch_to_numpy
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.utils.dataset.rl_dataset import collate_fn


SUMMARY_INSTRUCTION = """Fill a safe five-line retry memory for the same ALFWorld task.
The retry starts from the initial scene. It will see this memory plus the live observation, not the first-attempt trajectory.

Grounding: use only [Task], [Compact first attempt trace], and [Outcome feedback]. Copy the target object and target receptacle from the task. If an item was not seen, write not observed.

Output exactly these five labels, one line each, no Markdown, no bullets, no numbering, no wrapper tags, no extra text. Missing any label receives no credit; if uncertain, fill unknown instead of dropping a line:
Task: <task phrase>
Known: object=<task target object; observed/not observed/unknown; step>; receptacle=<task target receptacle; visible/not visible/unknown; step>; inventory=<known/empty/unknown>
Attempt1: tried=<locations/actions tried>; bad=<empty/wasted locations/actions>; errors=<malformed steps or none>
Plan: first=<one command-style intention>; backup=<one different command-style intention>; after_object=<take/put/examine subgoal or unknown>
Rule: one admissible command per action; no or/and-then/list actions; no invented names

Plan verbs should be ALFWorld-style: go to, open, take, put, move, examine, inventory, look, cool, heat, clean, use, or slice. Do not use synonyms like check, inspect, explore, search, order, or verify.
Limit: at most 90 words total. Stop after the Rule line.
"""


SUMMARY_LABELS = ("Task:", "Known:", "Attempt1:", "Plan:", "Rule:")
SUMMARY_PLAN_FIELDS = ("first=", "backup=", "after_object=")
SUMMARY_ALLOWED_PLAN_PREFIXES = (
    "go to ",
    "open ",
    "take ",
    "put ",
    "move ",
    "examine ",
    "inventory",
    "look",
    "cool ",
    "heat ",
    "clean ",
    "use ",
    "slice ",
    "unknown",
)
SUMMARY_DISALLOWED_PLAN_WORDS = ("check", "inspect", "explore", "search", "order", "verify", "try to")


class SSCA2TrajCollector(BaseTrajectoryCollector):
    """First-version SSCA rollout with a trainable summary action.

    Per prompt, the chain is:
    1. traj1: ALFWorld task prompt -> first-attempt actions;
    2. summary: compact observation trace + outcome feedback + summary prompt -> retry memory;
    3. traj2: reset to the initial task, then ALFWorld task prompt + retry memory -> retry actions.

    Training data is arranged as two standard GRPO groups per prompt:
    - traj1 rows use the first-attempt outcome reward;
    - summary + traj2 rows share one retry uid/traj_uid and use the retry outcome
      reward, so the retry-memory response tokens are optimized for improving
      subsequent retry policy quality.
    """

    def _ssca_cfg(self):
        return self.config.get("ssca", {})

    def _cfg(self, name: str, default: Any) -> Any:
        cfg = self._ssca_cfg()
        if hasattr(cfg, "get"):
            return cfg.get(name, default)
        return default

    def _max_history_chars(self) -> int:
        return int(self._cfg("summary_max_history_chars", 12000))

    def _max_prompt_chars(self) -> int:
        return int(self._cfg("summary_max_prompt_chars", 1200))

    def _max_summary_chars(self) -> int:
        return int(self._cfg("summary_max_chars", 1200))

    def _summary_context_mode(self) -> str:
        return str(self._cfg("summary_context_mode", "compact_obs_trace"))

    def _summary_trace_max_steps(self) -> int:
        return int(self._cfg("summary_trace_max_steps", 6))

    def _summary_step_obs_chars(self) -> int:
        return int(self._cfg("summary_step_obs_chars", 240))

    def _summary_step_feedback_chars(self) -> int:
        return int(self._cfg("summary_step_feedback_chars", 240))

    def _summary_invalid_response_chars(self) -> int:
        return int(self._cfg("summary_invalid_response_chars", 0))

    def _failure_reward_threshold(self) -> float:
        return float(self._cfg("failure_reward_threshold", 0.0))

    def _retry_same_seed(self) -> bool:
        return bool(self._cfg("retry_same_seed", True))

    def _summary_quality_reward_coef(self) -> float:
        return float(self._cfg("summary_quality_reward_coef", 0.2))

    def _summary_quality_max_words(self) -> int:
        return int(self._cfg("summary_quality_max_words", 110))

    def _clip_text(self, text: str, max_chars: int) -> str:
        text = str(text or "")
        if max_chars > 0 and len(text) > max_chars:
            return text[-max_chars:]
        return text

    def _clip_head(self, text: str, max_chars: int) -> str:
        text = str(text or "")
        if max_chars > 0 and len(text) > max_chars:
            return text[:max_chars]
        return text

    def _extract_task_text(self, text: str) -> str:
        marker = "Your task is to: "
        text = str(text or "")
        start = text.find(marker)
        if start == -1:
            return "unknown"
        start += len(marker)
        end_candidates = [pos for pos in [text.find("\n", start), text.find(".", start)] if pos != -1]
        end = min(end_candidates) if end_candidates else len(text)
        return text[start:end].strip() or "unknown"

    def _task_terms(self, task_text: str) -> set[str]:
        stop_words = {
            "a", "an", "and", "in", "into", "on", "onto", "put", "find", "two",
            "some", "the", "with", "to", "of", "them", "it", "then", "examine",
        }
        terms = set()
        for token in str(task_text or "").lower().replace("/", " ").replace("-", " ").split():
            token = token.strip(".,:;!?()[]{}'\"")
            if len(token) >= 3 and token not in stop_words:
                terms.add(token)
        return terms

    def _summary_plan_value(self, plan_line: str, field: str) -> str:
        lower = str(plan_line or "").lower()
        start = lower.find(field)
        if start == -1:
            return ""
        start += len(field)
        end = len(plan_line)
        for other_field in SUMMARY_PLAN_FIELDS:
            if other_field == field:
                continue
            pos = lower.find(other_field, start)
            if pos != -1:
                end = min(end, pos)
        return str(plan_line[start:end]).strip(" ;,.:")

    def _summary_quality(self, summary_text: str, task_text: str) -> Dict[str, float]:
        text = str(summary_text or "").strip()
        lower = text.lower()
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        label_count = sum(1 for label in SUMMARY_LABELS if label in text)
        ordered_labels = len(lines) == len(SUMMARY_LABELS) and all(
            lines[idx].startswith(label) for idx, label in enumerate(SUMMARY_LABELS)
        )
        five_labels = label_count == len(SUMMARY_LABELS)
        word_count = len(text.split())
        word_limit_ok = word_count <= self._summary_quality_max_words()
        no_wrapper = not any(token in text for token in ("```", "<think>", "</think>", "<action>", "</action>", "[Task]", "[Known]"))
        no_bullets = not any(line.startswith(("-", "*")) or line[:1].isdigit() for line in lines)

        plan_line = next((line for line in lines if line.startswith("Plan:")), "")
        plan_lower = plan_line.lower()
        plan_field_count = sum(1 for field in SUMMARY_PLAN_FIELDS if field in plan_lower)
        first_plan = self._summary_plan_value(plan_line, "first=").lower()
        first_plan_ok = bool(first_plan) and first_plan.startswith(SUMMARY_ALLOWED_PLAN_PREFIXES)
        bad_plan_word = any(word in plan_lower for word in SUMMARY_DISALLOWED_PLAN_WORDS)
        multi_action_plan = any(token in plan_lower for token in (" and then ", " then ", " or ", "->"))
        comma_or_list_plan = "," in plan_line or "\n-" in plan_line
        plan_ok = plan_field_count >= 2 and first_plan_ok and not bad_plan_word and not multi_action_plan and not comma_or_list_plan

        terms = self._task_terms(task_text)
        if terms:
            task_hits = sum(1 for term in terms if term in lower)
            task_overlap = task_hits / max(1, min(len(terms), 4))
            task_overlap = min(1.0, task_overlap)
        else:
            task_overlap = 0.0

        score = (
            0.30 * (label_count / len(SUMMARY_LABELS))
            + 0.20 * float(ordered_labels)
            + 0.15 * (plan_field_count / len(SUMMARY_PLAN_FIELDS))
            + 0.15 * float(plan_ok)
            + 0.10 * float(word_limit_ok)
            + 0.05 * float(no_wrapper and no_bullets)
            + 0.05 * task_overlap
        )
        return {
            "score": float(max(0.0, min(1.0, score))),
            "label_count": float(label_count),
            "five_label": float(five_labels),
            "ordered_five_line": float(ordered_labels),
            "plan_field_count": float(plan_field_count),
            "plan_ok": float(plan_ok),
            "word_count": float(word_count),
            "word_limit_ok": float(word_limit_ok),
            "no_wrapper": float(no_wrapper and no_bullets),
            "task_overlap": float(task_overlap),
        }

    def _extract_action_text(self, response: str) -> str:
        text = str(response or "")
        lower = text.lower()
        start_tag = "<action>"
        end_tag = "</action>"
        start_idx = lower.find(start_tag)
        end_idx = lower.find(end_tag)
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            return lower[start_idx + len(start_tag):end_idx].strip()
        return self._clip_text(text, 80).strip()

    def _malformed_reason(self, response: str, is_valid: bool) -> str:
        if is_valid:
            return ""
        text = str(response or "")
        lower = text.lower()
        reasons = []
        if "<think>" not in lower or "</think>" not in lower:
            reasons.append("missing <think>...</think>")
        if "<action>" not in lower or "</action>" not in lower:
            reasons.append("missing <action>...</action>")
        if any("\u4e00" <= char <= "\u9fff" for char in text):
            reasons.append("contains Chinese characters")
        if not reasons:
            reasons.append("projection marked action invalid")
        return "; ".join(reasons)

    def _as_text_item(self, values: Any, idx: int) -> str:
        if values is None:
            return ""
        try:
            value = values[idx]
        except Exception:
            return ""
        if isinstance(value, list):
            value = value[0] if value else ""
        return str(value or "")

    def _select_compact_steps(self, history: List[Dict[str, Any]], task_text: str) -> List[Dict[str, Any]]:
        if not history:
            return []
        max_steps = max(1, self._summary_trace_max_steps())
        terms = self._task_terms(task_text)
        last_step = max(int(step.get("step", 0)) for step in history)
        scored = []
        for idx, step in enumerate(history):
            step_num = int(step.get("step", idx + 1))
            blob = " ".join(
                str(step.get(key, "")).lower()
                for key in ["observation", "env_feedback", "parsed_action"]
            )
            task_related = bool(terms and any(term in blob for term in terms))
            invalid = not bool(step.get("valid", True))
            score = step_num
            if step_num == 1:
                score += 30
            if step_num == last_step:
                score += 40
            if invalid:
                score += 100
            if task_related:
                score += 50
            scored.append((score, idx, step))
        selected = [step for _, _, step in sorted(scored, key=lambda item: (-item[0], item[1]))[:max_steps]]
        return sorted(selected, key=lambda step: int(step.get("step", 0)))

    def _format_compact_trace(self, initial_scene: str, history: List[Dict[str, Any]], task_text: str) -> str:
        selected = self._select_compact_steps(history, task_text)
        lines = [
            "[Task]",
            task_text or "unknown",
            "",
            "[Initial scene]",
            self._clip_head(initial_scene, self._max_prompt_chars()),
            "",
            "[Compact first attempt]",
        ]
        if len(selected) < len(history):
            selected_steps = ", ".join(str(step.get("step")) for step in selected)
            lines.append(f"Selected {len(selected)} of {len(history)} steps for summary evidence: {selected_steps}.")
        for step in selected:
            lines.extend(
                [
                    f"Step {step.get('step')}",
                    "Obs: " + self._clip_head(step.get("observation", ""), self._summary_step_obs_chars()),
                    "Projected action: " + self._clip_head(step.get("parsed_action", ""), 160),
                    f"Valid: {bool(step.get('valid', False))}",
                ]
            )
            malformed = str(step.get("malformed_reason", ""))
            if malformed:
                lines.append("Malformed reason: " + malformed)
                invalid_response_chars = self._summary_invalid_response_chars()
                if invalid_response_chars > 0:
                    lines.append("Raw model response excerpt: " + self._clip_head(step.get("action_output", ""), invalid_response_chars))
            lines.extend(
                [
                    "Env reward: " + str(step.get("reward", 0.0)),
                    "Done: " + str(bool(step.get("done", False))),
                    "Env feedback: " + self._clip_head(step.get("env_feedback", ""), self._summary_step_feedback_chars()),
                    "",
                ]
            )
        return self._clip_head("\n".join(lines).strip(), self._max_history_chars())

    def _generate_from_obs(self, gen_batch: DataProto, actor_rollout_wg, obs: Dict[str, Any]) -> DataProto:
        batch = self.preprocess_batch(gen_batch=gen_batch, obs=obs)
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
        if "multi_modal_data" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("multi_modal_data")
        if "raw_prompt" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("raw_prompt")
        if "tools_kwargs" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("tools_kwargs")

        batch_input = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
        )
        batch_input.meta_info = gen_batch.meta_info
        batch_input_padded, pad_size = pad_dataproto_to_divisor(batch_input, actor_rollout_wg.world_size)
        batch_output_padded = actor_rollout_wg.generate_sequences(batch_input_padded)
        batch_output = unpad_dataproto(batch_output_padded, pad_size=pad_size)
        return batch.union(batch_output)

    def _decode_responses(self, batch: DataProto) -> List[str]:
        return [text.strip() for text in self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)]

    def _make_group_uids(self, batch_size: int, suffix: str) -> np.ndarray:
        if self.config.env.rollout.n > 0:
            uid_batch = []
            for idx in range(batch_size):
                if idx % self.config.env.rollout.n == 0:
                    uid = f"{uuid.uuid4()}:{suffix}"
                uid_batch.append(uid)
            return np.array(uid_batch, dtype=object)
        uid = f"{uuid.uuid4()}:{suffix}"
        return np.array([uid for _ in range(batch_size)], dtype=object)

    def _annotate_common(
        self,
        batch: DataProto,
        uid_batch: np.ndarray,
        traj_uid: np.ndarray,
        phase: str,
        attempt: int,
        step_idx: int,
        active_masks: np.ndarray,
        summary_texts: List[str] | None = None,
        retry_prompt_match: List[bool] | None = None,
        linked_first_traj_uid: np.ndarray | None = None,
    ) -> None:
        batch_size = len(uid_batch)
        if summary_texts is None:
            summary_texts = [""] * batch_size
        if retry_prompt_match is None:
            retry_prompt_match = [True] * batch_size
        if linked_first_traj_uid is None:
            linked_first_traj_uid = np.array([""] * batch_size, dtype=object)
        chain_stage = {
            "traj1": "first_attempt_policy",
            "summary": f"summary_action_from_{self._summary_context_mode()}",
            "traj2": "retry_policy_conditioned_on_summary",
        }.get(phase, phase)
        reward_source = "first_attempt_reward" if phase == "traj1" else "retry_attempt_reward"
        context_contract = {
            "traj1": "alfworld_task_prompt_only",
            "summary": f"{self._summary_context_mode()}_plus_outcome_feedback_to_summary_action",
            "traj2": "alfworld_task_prompt_plus_summary_no_traj1_details",
        }.get(phase, "unknown")
        batch.non_tensor_batch["uid"] = uid_batch
        batch.non_tensor_batch["traj_uid"] = traj_uid
        batch.non_tensor_batch["ssca_phase"] = np.array([phase] * batch_size, dtype=object)
        batch.non_tensor_batch["ssca_attempt"] = np.array([attempt] * batch_size, dtype=object)
        batch.non_tensor_batch["ssca_step_idx"] = np.array([step_idx] * batch_size, dtype=object)
        batch.non_tensor_batch["ssca_is_summary"] = np.array([phase == "summary"] * batch_size, dtype=bool)
        batch.non_tensor_batch["ssca_summary_is_policy_action"] = np.array([phase == "summary"] * batch_size, dtype=bool)
        batch.non_tensor_batch["ssca_chain_stage"] = np.array([chain_stage] * batch_size, dtype=object)
        batch.non_tensor_batch["ssca_reward_source"] = np.array([reward_source] * batch_size, dtype=object)
        batch.non_tensor_batch["ssca_context_contract"] = np.array([context_contract] * batch_size, dtype=object)
        batch.non_tensor_batch["ssca_summary_context_mode"] = np.array([self._summary_context_mode()] * batch_size, dtype=object)
        batch.non_tensor_batch["ssca_visible_first_trajectory"] = np.array([phase == "summary"] * batch_size, dtype=bool)
        batch.non_tensor_batch["ssca_summary_text"] = np.array(summary_texts, dtype=object)
        batch.non_tensor_batch["ssca_retry_prompt_match"] = np.array(retry_prompt_match, dtype=bool)
        batch.non_tensor_batch["ssca_linked_first_traj_uid"] = linked_first_traj_uid
        batch.non_tensor_batch["ssca_summary_quality_score"] = np.zeros(batch_size, dtype=np.float32)
        batch.non_tensor_batch["ssca_summary_label_count"] = np.zeros(batch_size, dtype=np.float32)
        batch.non_tensor_batch["ssca_summary_five_label"] = np.zeros(batch_size, dtype=np.float32)
        batch.non_tensor_batch["ssca_summary_ordered_five_line"] = np.zeros(batch_size, dtype=np.float32)
        batch.non_tensor_batch["ssca_summary_plan_ok"] = np.zeros(batch_size, dtype=np.float32)
        batch.non_tensor_batch["ssca_summary_word_count"] = np.zeros(batch_size, dtype=np.float32)
        batch.non_tensor_batch["ssca_reward_bonus"] = np.zeros(batch_size, dtype=np.float32)
        batch.non_tensor_batch["active_masks"] = torch_to_numpy(active_masks, is_object=True)

    def gather_rollout_data(
        self,
        total_batch_list: List[List[Dict]],
        episode_rewards: np.ndarray,
        episode_lengths: np.ndarray,
        success: Dict[str, np.ndarray],
        traj_uid: np.ndarray,
        tool_callings: np.ndarray,
    ) -> DataProto:
        batch_size = len(total_batch_list)
        rollout_metrics = {}
        for key, value in success.items():
            values = np.asarray(value)
            if values.size > 0:
                rollout_metrics[key] = float(np.mean(values.astype(np.float32)))

        effective_batch = []
        for bs in range(batch_size):
            for data in total_batch_list[bs]:
                assert traj_uid[bs] == data["traj_uid"], "data is not from the same trajectory"
                if data["active_masks"]:
                    env_reward = float(episode_rewards[bs])
                    reward_bonus = float(data.get("ssca_reward_bonus", 0.0) or 0.0)
                    data["ssca_env_episode_rewards"] = env_reward
                    data["episode_rewards"] = env_reward + reward_bonus
                    data["episode_lengths"] = episode_lengths[bs]
                    data["tool_callings"] = tool_callings[bs]
                    for key, value in rollout_metrics.items():
                        data[key] = value
                    effective_batch.append(data)

        gen_batch_output = DataProto.from_single_dict(data=collate_fn(effective_batch))
        return gen_batch_output

    def _feedback_text(
        self,
        reward: float,
        success: bool,
        length: float,
        invalid_count: int,
        final_info: Dict[str, Any] | None,
    ) -> str:
        status = "FAILED" if reward <= self._failure_reward_threshold() else "SUCCEEDED"
        if success:
            status = "SUCCEEDED"
        lines = [
            f"Attempt status: {status}.",
            f"Final environment reward: {reward:.4f}.",
            f"Episode length: {int(length)} environment actions.",
            f"Invalid action count: {invalid_count}.",
        ]
        if final_info:
            if "won" in final_info:
                lines.append(f"Environment won flag: {final_info.get('won')}.")
            if "goal_condition_success_rate" in final_info:
                lines.append(f"Goal-condition success rate: {final_info.get('goal_condition_success_rate')}.")
            obs = final_info.get("observation_text")
            if obs:
                if isinstance(obs, list):
                    obs = obs[0] if obs else ""
                lines.append("Final observation: " + self._clip_text(str(obs), 1200))
        if status == "FAILED":
            lines.append("Use this outcome only as evidence for failure diagnosis; do not invent unseen objects or locations.")
        else:
            lines.append("Use this outcome only as evidence for preserving useful actions and improving retry efficiency.")
        return "\n".join(lines)

    def _build_summary_obs(
        self,
        initial_obs_texts: List[str],
        initial_anchor_texts: List[str],
        histories: List[List[Dict[str, Any]]],
        episode_rewards: np.ndarray,
        episode_lengths: np.ndarray,
        invalid_counts: np.ndarray,
        final_infos: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        prompts = []
        for idx, history in enumerate(histories):
            final_info = final_infos[idx] if idx < len(final_infos) else {}
            reward = float(episode_rewards[idx])
            success = bool(final_info.get("won", reward > self._failure_reward_threshold())) if isinstance(final_info, dict) else reward > self._failure_reward_threshold()
            initial_anchor = initial_anchor_texts[idx] if idx < len(initial_anchor_texts) else ""
            initial_scene = initial_anchor or initial_obs_texts[idx]
            task_text = self._extract_task_text(initial_scene)
            if self._summary_context_mode() == "full_prompt_trace":
                trajectory_text = self._clip_text(
                    "\n".join(str(step.get("legacy_text", step)) for step in history),
                    self._max_history_chars(),
                )
                context_header = "[First trajectory full prompt trace]"
            else:
                trajectory_text = self._format_compact_trace(
                    initial_scene=initial_scene,
                    history=history,
                    task_text=task_text,
                )
                context_header = "[Compact first attempt trace]"
            feedback = self._feedback_text(
                reward=reward,
                success=success,
                length=float(episode_lengths[idx]),
                invalid_count=int(invalid_counts[idx]),
                final_info=final_info,
            )
            prompts.append(
                f"{SUMMARY_INSTRUCTION}\n"
                f"{context_header}\n{trajectory_text}\n\n"
                f"[Outcome feedback]\n{feedback}\n\n"
                "Write only the five-line retry memory. Stop after the Rule line."
            )
        return {"text": prompts, "image": None, "anchor": np.array(prompts, dtype=object)}

    def _prepend_summary_to_obs(self, obs: Dict[str, Any], summary_texts: List[str], initial_obs_texts: List[str]) -> Tuple[Dict[str, Any], List[bool]]:
        obs_texts = list(obs.get("text") or [""] * len(summary_texts))
        updated_texts = []
        prompt_match = []
        for idx, obs_text in enumerate(obs_texts):
            summary = self._clip_text(summary_texts[idx], self._max_summary_chars())
            initial_prompt = self._clip_text(initial_obs_texts[idx], self._max_prompt_chars())
            current_prompt = str(obs_text or "")
            prompt_match.append(initial_prompt[:200] in current_prompt if initial_prompt else True)
            updated_texts.append(
                f"{current_prompt}\n\n"
                "[Retry memory generated from the first attempt]\n"
                f"{summary}\n\n"
                "Retry context contract: use the current ALFWorld observation plus this retry memory only. "
                "Treat the retry as starting from the initial scene; choose exactly one admissible action."
            )
        return {"text": updated_texts, "image": obs.get("image"), "anchor": obs.get("anchor")}, prompt_match

    def _run_env_episode(
        self,
        gen_batch: DataProto,
        actor_rollout_wg,
        envs: EnvironmentManagerBase,
        obs: Dict[str, Any],
        uid_batch: np.ndarray,
        traj_uid: np.ndarray,
        phase: str,
        attempt: int,
        summary_texts: List[str] | None = None,
        retry_prompt_match: List[bool] | None = None,
        linked_first_traj_uid: np.ndarray | None = None,
    ):
        batch_size = len(gen_batch.batch)
        is_done = np.zeros(batch_size, dtype=bool)
        batch_lists = [[] for _ in range(batch_size)]
        infos_lists = [[] for _ in range(batch_size)]
        histories: List[List[Dict[str, Any]]] = [[] for _ in range(batch_size)]
        episode_rewards = np.zeros(batch_size, dtype=np.float32)
        episode_lengths = np.zeros(batch_size, dtype=np.float32)
        tool_callings = np.zeros(batch_size, dtype=np.float32)
        invalid_counts = np.zeros(batch_size, dtype=np.int32)
        final_infos: List[Dict[str, Any]] = [{} for _ in range(batch_size)]

        for step_idx in range(int(self.config.env.max_steps)):
            active_masks = np.logical_not(is_done)
            batch = self._generate_from_obs(gen_batch=gen_batch, actor_rollout_wg=actor_rollout_wg, obs=obs)
            self._annotate_common(
                batch=batch,
                uid_batch=uid_batch,
                traj_uid=traj_uid,
                phase=phase,
                attempt=attempt,
                step_idx=step_idx,
                active_masks=active_masks,
                summary_texts=summary_texts,
                retry_prompt_match=retry_prompt_match,
                linked_first_traj_uid=linked_first_traj_uid,
            )

            text_actions = self._decode_responses(batch)
            next_obs, rewards, dones, infos = envs.step(text_actions)

            if len(rewards.shape) == 2:
                rewards = rewards.squeeze(1)
            if len(dones.shape) == 2:
                dones = dones.squeeze(1)

            if "is_action_valid" in infos[0]:
                valids = np.array([info["is_action_valid"] for info in infos], dtype=bool)
            else:
                valids = np.ones(batch_size, dtype=bool)
            batch.non_tensor_batch["is_action_valid"] = valids

            if "tool_calling" in infos[0]:
                tool_callings[active_masks] += np.array([info["tool_calling"] for info in infos], dtype=np.float32)[active_masks]

            rewards_np = torch_to_numpy(rewards)
            episode_rewards[active_masks] += rewards_np[active_masks]
            episode_lengths[active_masks] += 1
            invalid_counts[active_masks] += (~valids[active_masks]).astype(np.int32)

            batch.non_tensor_batch["rewards"] = torch_to_numpy(rewards, is_object=True)
            batch_items = to_list_of_dict(batch)

            obs_texts = obs.get("text") or [""] * batch_size
            obs_anchors = obs.get("anchor", None)
            next_texts = next_obs.get("text") or [""] * batch_size
            next_anchors = next_obs.get("anchor", None)
            for idx in range(batch_size):
                if bool(active_masks[idx]):
                    current_observation = self._as_text_item(obs_anchors, idx) or self._as_text_item(obs_texts, idx)
                    env_feedback = self._as_text_item(next_anchors, idx) or self._as_text_item(next_texts, idx)
                    action_output = text_actions[idx]
                    histories[idx].append(
                        {
                            "step": step_idx + 1,
                            "observation": current_observation,
                            "action_output": action_output,
                            "parsed_action": self._extract_action_text(action_output),
                            "reward": float(rewards_np[idx]),
                            "done": bool(dones[idx]),
                            "valid": bool(valids[idx]),
                            "malformed_reason": self._malformed_reason(action_output, bool(valids[idx])),
                            "env_feedback": env_feedback,
                            "legacy_text": "\n".join(
                                [
                                    f"Step {step_idx + 1}",
                                    f"Observation: {self._clip_text(self._as_text_item(obs_texts, idx), 1200)}",
                                    f"Action: {action_output}",
                                    f"Reward: {float(rewards_np[idx])}",
                                    f"Done: {bool(dones[idx])}",
                                    f"Valid action: {bool(valids[idx])}",
                                ]
                            ),
                        }
                    )
                    final_infos[idx] = copy.deepcopy(infos[idx])
                batch_lists[idx].append(batch_items[idx])
                infos_lists[idx].append(infos[idx])

            is_done = np.logical_or(is_done, dones)
            obs = next_obs
            if summary_texts is not None:
                obs, _ = self._prepend_summary_to_obs(obs, summary_texts, initial_obs_texts=[""] * batch_size)

            if is_done.all():
                break

        return batch_lists, infos_lists, histories, episode_rewards, episode_lengths, tool_callings, invalid_counts, final_infos

    def vanilla_multi_turn_loop(
        self,
        gen_batch: DataProto,
        actor_rollout_wg,
        envs: EnvironmentManagerBase,
    ):
        batch_size = len(gen_batch.batch)
        env_kwargs = gen_batch.non_tensor_batch.pop("env_kwargs", None)

        first_obs, _ = envs.reset(kwargs=env_kwargs)
        length_obs = len(first_obs["text"]) if first_obs["text"] is not None else len(first_obs["image"])
        assert len(gen_batch.batch) == length_obs, f"gen_batch size {len(gen_batch.batch)} does not match obs size {length_obs}"
        initial_obs_texts = list(first_obs.get("text") or [""] * batch_size)
        initial_anchor_values = first_obs.get("anchor", None)
        initial_anchor_texts = [
            self._as_text_item(initial_anchor_values, idx) or initial_obs_texts[idx]
            for idx in range(batch_size)
        ]

        uid_first = self._make_group_uids(batch_size=batch_size, suffix="traj1")
        uid_retry = self._make_group_uids(batch_size=batch_size, suffix="retry")
        traj_uid_first = np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object)
        traj_uid_retry = np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object)

        first_lists, first_infos, first_histories, reward1, len1, tools1, invalid1, final_infos1 = self._run_env_episode(
            gen_batch=gen_batch,
            actor_rollout_wg=actor_rollout_wg,
            envs=envs,
            obs=first_obs,
            uid_batch=uid_first,
            traj_uid=traj_uid_first,
            phase="traj1",
            attempt=1,
            linked_first_traj_uid=traj_uid_first,
        )

        summary_obs = self._build_summary_obs(
            initial_obs_texts=initial_obs_texts,
            initial_anchor_texts=initial_anchor_texts,
            histories=first_histories,
            episode_rewards=reward1,
            episode_lengths=len1,
            invalid_counts=invalid1,
            final_infos=final_infos1,
        )
        summary_batch = self._generate_from_obs(gen_batch=gen_batch, actor_rollout_wg=actor_rollout_wg, obs=summary_obs)
        summary_texts = self._decode_responses(summary_batch)
        summary_task_texts = [
            self._extract_task_text(initial_anchor_texts[idx] or initial_obs_texts[idx])
            for idx in range(batch_size)
        ]
        summary_quality = [
            self._summary_quality(summary_texts[idx], summary_task_texts[idx])
            for idx in range(batch_size)
        ]
        summary_quality_score = np.array([item["score"] for item in summary_quality], dtype=np.float32)
        summary_label_count = np.array([item["label_count"] for item in summary_quality], dtype=np.float32)
        summary_five_label = np.array([item["five_label"] for item in summary_quality], dtype=np.float32)
        summary_ordered_five_line = np.array([item["ordered_five_line"] for item in summary_quality], dtype=np.float32)
        summary_plan_ok = np.array([item["plan_ok"] for item in summary_quality], dtype=np.float32)
        summary_word_count = np.array([item["word_count"] for item in summary_quality], dtype=np.float32)
        is_validate = bool(gen_batch.meta_info.get("validate", False)) if hasattr(gen_batch, "meta_info") else False
        summary_bonus_coef = 0.0 if is_validate else self._summary_quality_reward_coef()
        summary_bonus = summary_bonus_coef * summary_quality_score

        active_summary = np.ones(batch_size, dtype=bool)
        self._annotate_common(
            batch=summary_batch,
            uid_batch=uid_retry,
            traj_uid=traj_uid_retry,
            phase="summary",
            attempt=2,
            step_idx=-1,
            active_masks=active_summary,
            summary_texts=summary_texts,
            linked_first_traj_uid=traj_uid_first,
        )
        summary_batch.non_tensor_batch["ssca_summary_quality_score"] = summary_quality_score
        summary_batch.non_tensor_batch["ssca_summary_label_count"] = summary_label_count
        summary_batch.non_tensor_batch["ssca_summary_five_label"] = summary_five_label
        summary_batch.non_tensor_batch["ssca_summary_ordered_five_line"] = summary_ordered_five_line
        summary_batch.non_tensor_batch["ssca_summary_plan_ok"] = summary_plan_ok
        summary_batch.non_tensor_batch["ssca_summary_word_count"] = summary_word_count
        summary_batch.non_tensor_batch["ssca_reward_bonus"] = summary_bonus.astype(np.float32)
        summary_batch.non_tensor_batch["rewards"] = np.zeros(batch_size, dtype=object)
        summary_batch.non_tensor_batch["is_action_valid"] = np.ones(batch_size, dtype=bool)
        summary_items = to_list_of_dict(summary_batch)

        retry_kwargs = {"retry_same_seed": self._retry_same_seed()}
        retry_obs, _ = envs.reset(kwargs=retry_kwargs)
        retry_obs, retry_prompt_match = self._prepend_summary_to_obs(retry_obs, summary_texts, initial_obs_texts)
        retry_lists, retry_infos, _, reward2, len2, tools2, invalid2, final_infos2 = self._run_env_episode(
            gen_batch=gen_batch,
            actor_rollout_wg=actor_rollout_wg,
            envs=envs,
            obs=retry_obs,
            uid_batch=uid_retry,
            traj_uid=traj_uid_retry,
            phase="traj2",
            attempt=2,
            summary_texts=summary_texts,
            retry_prompt_match=retry_prompt_match,
            linked_first_traj_uid=traj_uid_first,
        )

        total_batch_list = []
        total_infos = []
        total_episode_rewards = []
        total_episode_lengths = []
        total_traj_uid = []
        total_tool_callings = []

        success1 = np.array([
            float(final_infos1[idx].get("won", reward1[idx] > self._failure_reward_threshold()))
            if isinstance(final_infos1[idx], dict)
            else float(reward1[idx] > self._failure_reward_threshold())
            for idx in range(batch_size)
        ], dtype=np.float32)
        success2 = np.array([
            float(final_infos2[idx].get("won", reward2[idx] > self._failure_reward_threshold()))
            if isinstance(final_infos2[idx], dict)
            else float(reward2[idx] > self._failure_reward_threshold())
            for idx in range(batch_size)
        ], dtype=np.float32)
        reward_delta = reward2.astype(np.float32) - reward1.astype(np.float32)
        valid1 = 1.0 - invalid1.astype(np.float32) / np.maximum(len1.astype(np.float32), 1.0)
        valid2 = 1.0 - invalid2.astype(np.float32) / np.maximum(len2.astype(np.float32), 1.0)
        total_metric_count = batch_size * 2

        def repeated_metric(value: float) -> np.ndarray:
            return np.full(total_metric_count, float(value), dtype=np.float32)

        def safe_corr(left: np.ndarray, right: np.ndarray) -> float:
            if left.size < 2 or float(np.std(left)) < 1e-6 or float(np.std(right)) < 1e-6:
                return 0.0
            corr = float(np.corrcoef(left.astype(np.float32), right.astype(np.float32))[0, 1])
            return 0.0 if np.isnan(corr) else corr

        for idx in range(batch_size):
            total_batch_list.append(first_lists[idx])
            total_infos.append(first_infos[idx])
            total_episode_rewards.append(reward1[idx])
            total_episode_lengths.append(len1[idx])
            total_traj_uid.append(traj_uid_first[idx])
            total_tool_callings.append(tools1[idx])

        for idx in range(batch_size):
            retry_items = [summary_items[idx]] + retry_lists[idx]
            retry_info_items = [final_infos1[idx] if isinstance(final_infos1[idx], dict) else {}] + retry_infos[idx]
            total_batch_list.append(retry_items)
            total_infos.append(retry_info_items)
            total_episode_rewards.append(reward2[idx])
            total_episode_lengths.append(len2[idx])
            total_traj_uid.append(traj_uid_retry[idx])
            total_tool_callings.append(tools2[idx])

        success = {
            "success_rate": np.concatenate([success1, success2], axis=0),
            "ssca_traj1_success_rate": repeated_metric(np.mean(success1)),
            "ssca_traj2_success_rate": repeated_metric(np.mean(success2)),
            "ssca_retry_improved_success_rate": repeated_metric(np.mean(success2 > success1)),
            "ssca_retry_degraded_success_rate": repeated_metric(np.mean(success2 < success1)),
            "ssca_retry_prompt_match_rate": repeated_metric(np.mean(np.array(retry_prompt_match, dtype=np.float32))),
            "ssca_metric/traj1/reward_mean": repeated_metric(np.mean(reward1)),
            "ssca_metric/traj2/reward_mean": repeated_metric(np.mean(reward2)),
            "ssca_metric/retry/reward_delta_mean": repeated_metric(np.mean(reward_delta)),
            "ssca_metric/retry/reward_delta_std": repeated_metric(np.std(reward_delta)),
            "ssca_metric/retry/reward_improved_rate": repeated_metric(np.mean(reward_delta > 0)),
            "ssca_metric/retry/reward_degraded_rate": repeated_metric(np.mean(reward_delta < 0)),
            "ssca_metric/traj1/length_mean": repeated_metric(np.mean(len1)),
            "ssca_metric/traj2/length_mean": repeated_metric(np.mean(len2)),
            "ssca_metric/traj1/valid_action_ratio": repeated_metric(np.mean(valid1)),
            "ssca_metric/traj2/valid_action_ratio": repeated_metric(np.mean(valid2)),
            "ssca_metric/traj1/invalid_count_mean": repeated_metric(np.mean(invalid1)),
            "ssca_metric/traj2/invalid_count_mean": repeated_metric(np.mean(invalid2)),
            "ssca_metric/summary/quality_mean": repeated_metric(np.mean(summary_quality_score)),
            "ssca_metric/summary/quality_std": repeated_metric(np.std(summary_quality_score)),
            "ssca_metric/summary/label_count_mean": repeated_metric(np.mean(summary_label_count)),
            "ssca_metric/summary/five_label_rate": repeated_metric(np.mean(summary_five_label)),
            "ssca_metric/summary/ordered_five_line_rate": repeated_metric(np.mean(summary_ordered_five_line)),
            "ssca_metric/summary/plan_ok_rate": repeated_metric(np.mean(summary_plan_ok)),
            "ssca_metric/summary/word_count_mean": repeated_metric(np.mean(summary_word_count)),
            "ssca_metric/summary/bonus_mean": repeated_metric(np.mean(summary_bonus)),
            "ssca_metric/summary/bonus_nonzero_rate": repeated_metric(np.mean(summary_bonus > 0)),
            "ssca_metric/summary/quality_reward_corr": repeated_metric(safe_corr(summary_quality_score, reward2.astype(np.float32))),
        }
        return (
            total_batch_list,
            np.array(total_episode_rewards, dtype=np.float32),
            np.array(total_episode_lengths, dtype=np.float32),
            success,
            np.array(total_traj_uid, dtype=object),
            np.array(total_tool_callings, dtype=np.float32),
        )
