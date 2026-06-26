import copy
import uuid
from typing import Any, Dict, List, Tuple

import numpy as np

from agent_system.environments import EnvironmentManagerBase
from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector as BaseTrajectoryCollector
from agent_system.multi_turn_rollout.utils import to_list_of_dict, torch_to_numpy
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto


SUMMARY_INSTRUCTION = """You are producing a self-summary action for a retry on the same ALFWorld task.
This response is the summary action in the chain:
first trajectory + summary prompt -> self-summary -> ALFWorld task prompt + self-summary -> retry actions.

The retry starts from the initial environment state. The retry policy will see only the current ALFWorld task prompt/observation plus your summary; it will not see the detailed first trajectory.
Your objective is to write a summary that improves the quality of subsequent retry actions.

Write a concise policy-improvement note with this structure:
1. State belief: what task, objects, receptacles, and constraints matter?
2. Failure / success causes: which decisions helped or hurt in the first trajectory?
3. Policy improvement: what should be done differently or preserved on retry?
4. Retry plan: what action strategy should the next attempt follow?
"""


class SSCA2TrajCollector(BaseTrajectoryCollector):
    """First-version SSCA rollout with a trainable summary action.

    Per prompt, the chain is:
    1. traj1: ALFWorld task prompt -> first-attempt actions;
    2. summary: full traj1 + outcome feedback + summary prompt -> self-summary;
    3. traj2: reset to the initial task, then ALFWorld task prompt + self-summary -> retry actions.

    Training data is arranged as two standard GRPO groups per prompt:
    - traj1 rows use the first-attempt outcome reward;
    - summary + traj2 rows share one retry uid/traj_uid and use the retry outcome
      reward, so the summary response tokens are optimized for improving the
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
        return int(self._cfg("summary_max_prompt_chars", 4000))

    def _max_summary_chars(self) -> int:
        return int(self._cfg("summary_max_chars", 2000))

    def _failure_reward_threshold(self) -> float:
        return float(self._cfg("failure_reward_threshold", 0.0))

    def _retry_same_seed(self) -> bool:
        return bool(self._cfg("retry_same_seed", True))

    def _clip_text(self, text: str, max_chars: int) -> str:
        text = str(text or "")
        if max_chars > 0 and len(text) > max_chars:
            return text[-max_chars:]
        return text

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
            "summary": "summary_action_from_full_traj1",
            "traj2": "retry_policy_conditioned_on_summary",
        }.get(phase, phase)
        reward_source = "first_attempt_reward" if phase == "traj1" else "retry_attempt_reward"
        context_contract = {
            "traj1": "alfworld_task_prompt_only",
            "summary": "full_traj1_plus_outcome_feedback_to_summary_action",
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
        batch.non_tensor_batch["ssca_visible_first_trajectory"] = np.array([phase == "summary"] * batch_size, dtype=bool)
        batch.non_tensor_batch["ssca_summary_text"] = np.array(summary_texts, dtype=object)
        batch.non_tensor_batch["ssca_retry_prompt_match"] = np.array(retry_prompt_match, dtype=bool)
        batch.non_tensor_batch["ssca_linked_first_traj_uid"] = linked_first_traj_uid
        batch.non_tensor_batch["active_masks"] = torch_to_numpy(active_masks, is_object=True)

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
            lines.append("Generate a failure-analysis summary that helps the retry avoid the mistakes above.")
        else:
            lines.append("Generate an optimization summary that preserves the useful strategy and makes the retry more efficient.")
        return "\n".join(lines)

    def _build_summary_obs(
        self,
        initial_obs_texts: List[str],
        histories: List[List[str]],
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
            prompt_text = self._clip_text(initial_obs_texts[idx], self._max_prompt_chars())
            trajectory_text = self._clip_text("\n".join(history), self._max_history_chars())
            feedback = self._feedback_text(
                reward=reward,
                success=success,
                length=float(episode_lengths[idx]),
                invalid_count=int(invalid_counts[idx]),
                final_info=final_info,
            )
            prompts.append(
                f"{SUMMARY_INSTRUCTION}\n"
                f"[Original task prompt]\n{prompt_text}\n\n"
                f"[First trajectory]\n{trajectory_text}\n\n"
                f"[Outcome feedback]\n{feedback}\n\n"
                "Write only the self-summary note that will be shown to the retry policy."
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
                "[Self-summary action generated from the first attempt]\n"
                f"{summary}\n\n"
                "Retry context contract: use the current ALFWorld observation plus this self-summary only. "
                "The detailed first trajectory is not available; choose exactly one admissible action."
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
        histories: List[List[str]] = [[] for _ in range(batch_size)]
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
            for idx in range(batch_size):
                if bool(active_masks[idx]):
                    histories[idx].append(
                        "\n".join(
                            [
                                f"Step {step_idx + 1}",
                                f"Observation: {self._clip_text(obs_texts[idx], 1200)}",
                                f"Action: {text_actions[idx]}",
                                f"Reward: {float(rewards_np[idx])}",
                                f"Done: {bool(dones[idx])}",
                                f"Valid action: {bool(valids[idx])}",
                            ]
                        )
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
            histories=first_histories,
            episode_rewards=reward1,
            episode_lengths=len1,
            invalid_counts=invalid1,
            final_infos=final_infos1,
        )
        summary_batch = self._generate_from_obs(gen_batch=gen_batch, actor_rollout_wg=actor_rollout_wg, obs=summary_obs)
        summary_texts = self._decode_responses(summary_batch)
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
        summary_batch.non_tensor_batch["rewards"] = np.zeros(batch_size, dtype=object)
        summary_batch.non_tensor_batch["is_action_valid"] = np.ones(batch_size, dtype=bool)
        summary_items = to_list_of_dict(summary_batch)

        retry_kwargs = {"retry_same_seed": self._retry_same_seed()}
        retry_obs, _ = envs.reset(kwargs=retry_kwargs)
        retry_obs, retry_prompt_match = self._prepend_summary_to_obs(retry_obs, summary_texts, initial_obs_texts)
        retry_lists, retry_infos, _, reward2, len2, tools2, _, final_infos2 = self._run_env_episode(
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
        success_rate = []
        prompt_match_values = []

        for idx in range(batch_size):
            total_batch_list.append(first_lists[idx])
            total_infos.append(first_infos[idx])
            total_episode_rewards.append(reward1[idx])
            total_episode_lengths.append(len1[idx])
            total_traj_uid.append(traj_uid_first[idx])
            total_tool_callings.append(tools1[idx])
            success_rate.append(float(final_infos1[idx].get("won", reward1[idx] > self._failure_reward_threshold())) if isinstance(final_infos1[idx], dict) else float(reward1[idx] > self._failure_reward_threshold()))
            prompt_match_values.append(True)

        for idx in range(batch_size):
            retry_items = [summary_items[idx]] + retry_lists[idx]
            retry_info_items = [final_infos1[idx] if isinstance(final_infos1[idx], dict) else {}] + retry_infos[idx]
            total_batch_list.append(retry_items)
            total_infos.append(retry_info_items)
            total_episode_rewards.append(reward2[idx])
            total_episode_lengths.append(len2[idx])
            total_traj_uid.append(traj_uid_retry[idx])
            total_tool_callings.append(tools2[idx])
            success_rate.append(float(final_infos2[idx].get("won", reward2[idx] > self._failure_reward_threshold())) if isinstance(final_infos2[idx], dict) else float(reward2[idx] > self._failure_reward_threshold()))
            prompt_match_values.append(bool(retry_prompt_match[idx]))

        success = {
            "success_rate": np.array(success_rate, dtype=np.float32),
            "ssca_retry_prompt_match_rate": np.array(prompt_match_values, dtype=np.float32),
        }
        return (
            total_batch_list,
            np.array(total_episode_rewards, dtype=np.float32),
            np.array(total_episode_lengths, dtype=np.float32),
            success,
            np.array(total_traj_uid, dtype=object),
            np.array(total_tool_callings, dtype=np.float32),
        )
