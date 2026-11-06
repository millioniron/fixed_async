import inspect
import os
import threading
import time
from typing import Dict, Optional, Union, List

import ray
import torch
from codetiming import Timer
from tqdm import tqdm

from roll.configs.worker_config import WorkerConfig
from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.decorator import Dispatch, register
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.strategy.factory import create_strategy
from roll.distributed.strategy.strategy import InferenceStrategy, TrainStrategy
from roll.models.model_providers import (
    default_actor_model_provider,
    default_diffusion_module_provider,
    default_reward_model_provider,
    default_value_model_provider,
)
from roll.platforms import current_platform
from roll.utils.checkpoint_manager import download_model
from roll.utils.context_managers import state_offload_manger, log_gpu_memory_usage
from roll.utils.dynamic_batching import make_mini_batch_iter_for_dynamic_batching
from roll.utils.functionals import agg_loss, append_to_dict, compute_approx_kl, flatten_sum, masked_mean, postprocess_generate, reduce_metrics
from roll.utils.offload_nccl import reload_process_groups
from roll.utils.offload_states import OffloadStateType


class ActorWorker(Worker):
    def __init__(self, worker_config: WorkerConfig):
        super().__init__(worker_config=worker_config)
        self.tokenizer = None
        self.strategy: TrainStrategy = None
        self._logprobs_cache = {}

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def initialize(self, pipeline_config):
        super().initialize(pipeline_config)

        self.strategy = create_strategy(worker=self)

        if self.worker_config.model_args.model_type == "diffusion_module":
            self.strategy.initialize(model_provider=default_diffusion_module_provider)
        else:
            self.strategy.initialize(model_provider=default_actor_model_provider)

        self.tokenizer = self.strategy.tokenizer
        if self.pipeline_config.resume_from_checkpoint:
            load_dir = download_model(self.pipeline_config.resume_from_checkpoint)
            self.strategy.load_checkpoint(load_dir=load_dir, tag="checkpoint")
        self.logger.info(f"{self.worker_name} initialized")

        self.strategy.offload_states()

    def _log_local_train_batch_summary(self, prefix: str, data: DataProto):
        if data is None or data.batch is None or data.batch.batch_size[0] == 0:
            self.logger.warning("%s empty local batch", prefix)
            return

        attention_lengths = data.batch["attention_mask"].view(data.batch.batch_size[0], -1).sum(-1).detach().cpu()
        response_lengths = data.batch["response_mask"][:, 1:].sum(-1).detach().cpu() if "response_mask" in data.batch else None

        msg_parts = [
            f"{prefix} rank={self.rank_info.rank}",
            f"dp={self.rank_info.dp_rank}/{self.rank_info.dp_size}",
            f"tp={self.rank_info.tp_rank}/{self.rank_info.tp_size}",
            f"pp={self.rank_info.pp_rank}/{self.rank_info.pp_size}",
            f"cp={self.rank_info.cp_rank}/{self.rank_info.cp_size}",
            f"local_batch={data.batch.batch_size[0]}",
            f"attn_sum[min/mean/max]={int(attention_lengths.min().item())}/{attention_lengths.float().mean().item():.1f}/{int(attention_lengths.max().item())}",
            f"zero_attn={int((attention_lengths == 0).sum().item())}",
        ]

        if response_lengths is not None:
            msg_parts.extend([
                f"resp_sum[min/mean/max]={int(response_lengths.min().item())}/{response_lengths.float().mean().item():.1f}/{int(response_lengths.max().item())}",
                f"zero_resp={int((response_lengths == 0).sum().item())}",
            ])

        if "original_input_length" in data.non_tensor_batch:
            original_lengths = [int(x) for x in data.non_tensor_batch["original_input_length"]]
            msg_parts.append(
                f"orig_len[min/max]={min(original_lengths)}/{max(original_lengths)}"
            )
        if "is_overlength" in data.non_tensor_batch:
            overlength_count = sum(bool(x) for x in data.non_tensor_batch["is_overlength"])
            msg_parts.append(f"overlength={overlength_count}")

        self.logger.info(" ".join(msg_parts))

    @register(dispatch_mode=Dispatch.DP_MP_DISPATCH_FIRST)
    def train_step(self, data: DataProto):
        """
        return DataProto(meta_info={'metrics': metrics})
        """
        global_step = data.meta_info.get("global_step", 0)
        is_offload_states = data.meta_info.get("is_offload_states", True)
        data.meta_info.setdefault(
            "is_offload_optimizer_states_in_train_step",
            self.pipeline_config.is_offload_optimizer_states_in_train_step,
        )
        metrics = {}
        self.logger.info(f"{self.worker_name} generate global step {global_step}")

        with state_offload_manger(
            strategy=self.strategy,
            metrics=metrics,
            metric_infix=f"{self.cluster_name}/train_step",
            is_offload_states=is_offload_states,
            load_kwargs={"include": [OffloadStateType.model_params, OffloadStateType.other_params]},
        ):
            data = data.to(current_platform.device_type)
            data = self.strategy.get_data_input(data)
            self._log_local_train_batch_summary(f"{self.worker_name} local_train_batch", data)
            per_device_train_batch_size = self.worker_config.training_args.per_device_train_batch_size
            backward_batch_size = (
                    per_device_train_batch_size * self.worker_config.training_args.gradient_accumulation_steps
            )
            if self.worker_config.use_dynamic_batching_in_train:
                # TODO: support `keep_mini_batch`, The number of mini_batch may be smaller than original size
                dataloader = make_mini_batch_iter_for_dynamic_batching(
                    data=data,
                    epochs=self.pipeline_config.ppo_epochs,
                    ga_steps=self.worker_config.training_args.gradient_accumulation_steps,
                )
            else:
                dataloader = data.make_iterator(
                    mini_batch_size=backward_batch_size,
                    epochs=self.pipeline_config.ppo_epochs,
                    seed=self.pipeline_config.seed,
                    dataloader_kwargs={"shuffle": True},
                )

            for batch_idx, backward_batch in tqdm(enumerate(dataloader),
                                                  desc=f"{self.worker_name} train global step {global_step}",
                                                  total=data.batch.batch_size[0] * self.pipeline_config.ppo_epochs // backward_batch_size):
                if batch_idx < 2:
                    self._log_local_train_batch_summary(
                        f"{self.worker_name} backward_batch[{batch_idx}]",
                        backward_batch,
                    )
                pg_metrics = self.strategy.train_step(batch=backward_batch, loss_func=self.loss_func)
                if self.worker_config.use_dynamic_batching_in_train or self.worker_config.use_sequence_packing:
                    pg_metrics = reduce_metrics(pg_metrics)
                append_to_dict(metrics, pg_metrics)

            metrics["actor/lr"] = self.strategy.scheduler.get_last_lr()[0]
            backward_steps = data.batch.batch_size[0] * self.pipeline_config.ppo_epochs // backward_batch_size
            metrics["actor/backward_steps"] = backward_steps

            # Divide @sum metrics by backward_steps to get average
            for key in list(metrics.keys()):
                if key.endswith("@sum"):
                    if isinstance(metrics[key], list):
                        total = flatten_sum(metrics[key])
                        metrics[key] = total / backward_steps if backward_steps > 0 else total
                    elif isinstance(metrics[key], (int, float)):
                        metrics[key] = metrics[key] / backward_steps if backward_steps > 0 else metrics[key]

            data.to("cpu")

        self._logprobs_cache.clear()
        output = DataProto(meta_info={"metrics": metrics})
        return output

    @register(dispatch_mode=Dispatch.DP_MP_DISPATCH_FIRST)
    def compute_log_probs(self, data: DataProto, snapshot_step: Optional[int] = None, use_proximal: bool = False):
        """
        return DataProto.from_dict(tensors={'log_probs': output})
        """
        global_step = data.meta_info.get("global_step", 0)
        is_offload_states = data.meta_info.get("is_offload_states", True)
        metrics = {}
        snapshot_metrics_enabled = bool(getattr(self.pipeline_config, "enable_snapshot_overhead_metrics", False))
        snapshot_load_time_ms = 0.0
        snapshot_restore_time_ms = 0.0
        snapshot_forward_time_ms = 0.0
        with state_offload_manger(
            strategy=self.strategy,
            metrics=metrics,
            metric_infix=f"{self.cluster_name}/compute_log_probs",
            is_offload_states=is_offload_states,
            load_kwargs={"include": [OffloadStateType.model_params]},
        ):
            data = self.strategy.get_data_input(data)
            data = data.to(current_platform.device_type)
            data.meta_info["micro_batch_size"] = self.worker_config.infer_batch_size

            if snapshot_step is not None and use_proximal:
                raise ValueError("snapshot_step and use_proximal cannot be set at the same time")
            if snapshot_step is not None:
                snapshot_load_start = time.perf_counter()
                self.strategy.load_snapshot(step=snapshot_step)
                snapshot_load_time_ms = (time.perf_counter() - snapshot_load_start) * 1000.0
            elif use_proximal:
                self.strategy.load_proximal()
            try:
                snapshot_forward_start = time.perf_counter()
                with torch.no_grad():
                    results: Dict[str, torch.Tensor] = self.strategy.forward_step(
                        batch=data, forward_func=self.forward_func_log_probs
                    )
                snapshot_forward_time_ms = (time.perf_counter() - snapshot_forward_start) * 1000.0
            finally:
                if snapshot_step is not None or use_proximal:
                    snapshot_restore_start = time.perf_counter()
                    self.strategy.restore_current()
                    if snapshot_step is not None:
                        snapshot_restore_time_ms = (time.perf_counter() - snapshot_restore_start) * 1000.0
            if results is None:
                return DataProto(batch=None, meta_info={"metrics": metrics})
            output = DataProto.from_dict(tensors={"log_probs": results["log_probs"], "entropy": results["entropy"]})
            output = output.to("cpu")
            data.to("cpu")
        if snapshot_step is not None and snapshot_metrics_enabled:
            prefix = f"{self.cluster_name}/snapshot"
            metrics.update(
                {
                    f"{prefix}/load_time_ms@sum": snapshot_load_time_ms,
                    f"{prefix}/load_time_ms@mean": snapshot_load_time_ms,
                    f"{prefix}/load_count@sum": 1.0,
                    f"{prefix}/restore_time_ms@sum": snapshot_restore_time_ms,
                    f"{prefix}/restore_time_ms@mean": snapshot_restore_time_ms,
                    f"{prefix}/restore_count@sum": 1.0,
                    f"{prefix}/forward_time_ms@sum": snapshot_forward_time_ms,
                    f"{prefix}/forward_time_ms@mean": snapshot_forward_time_ms,
                    f"{prefix}/forward_count@sum": 1.0,
                    f"{prefix}/requested_step": float(snapshot_step),
                }
            )
        output.meta_info = {"metrics": metrics}
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_snapshot(self, step: int):
        metrics = {}
        with state_offload_manger(
            strategy=self.strategy,
            metrics=metrics,
            metric_infix=f"{self.cluster_name}/save_snapshot",
            load_kwargs={"include": [OffloadStateType.model_params]},
        ):
            if getattr(self.pipeline_config, "enable_snapshot_overhead_metrics", False):
                snapshot_metrics = self.strategy.save_snapshot(step=step)
                metrics.update(snapshot_metrics)
            else:
                self.strategy.save_snapshot(step=step)
        return DataProto(meta_info={"metrics": metrics})

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def evict_snapshot(self, step: int):
        self.strategy.evict_snapshot(step=step)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def update_proximal_ewma(self, beta: float):
        metrics = {}
        with state_offload_manger(
            strategy=self.strategy,
            metrics=metrics,
            metric_infix=f"{self.cluster_name}/update_proximal_ewma",
            load_kwargs={"include": [OffloadStateType.model_params]},
        ):
            self.strategy.update_proximal_ewma(beta=beta)
        return DataProto(meta_info={"metrics": metrics})

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def reset_proximal_to_current(self):
        metrics = {}
        with state_offload_manger(
            strategy=self.strategy,
            metrics=metrics,
            metric_infix=f"{self.cluster_name}/reset_proximal_to_current",
            load_kwargs={"include": [OffloadStateType.model_params]},
        ):
            self.strategy.reset_proximal_to_current()
        return DataProto(meta_info={"metrics": metrics})

    def forward_func_log_probs(self, data: DataProto, output_tensor: torch.Tensor):
        """
        forward func 接口定义:
            data: DataProto, 由forward_step透传
            output_tensor: torch.Tensor, model.forward()的输出Tensor
        """
        log_probs = self.strategy.op_compute_log_probs(
            logits=output_tensor, input_ids=data.batch["input_ids"], attention_mask=data.batch["response_mask"]
        )
        entropy = self.strategy.op_compute_entropy(logits=output_tensor, attention_mask=data.batch["response_mask"])
        return torch.tensor(0., device=output_tensor.device), {"log_probs": log_probs.clone().detach(), "entropy": entropy.clone().detach()}

    def get_old_log_probs_with_cache(self, data: DataProto, log_probs: torch.Tensor) -> torch.Tensor:
        """
        Get old_log_probs with intra-step caching when enable_old_logprobs_recompute == False.
        When caching is enabled, the first forward pass log_probs can be reused as old_log_probs
        since they are mathematically equivalent in on-policy settings.
        This method can be overridden by subclasses for custom caching behavior.

        Args:
            data: DataProto containing input data and sample_uuids
            log_probs: Current forward pass log_probs tensor

        Returns:
            old_log_probs tensor (detached, no gradients)
        """
        # Original computation path when caching is disabled
        if self.pipeline_config.enable_old_logprobs_recompute or "sample_uuid" not in data.non_tensor_batch:
            # When enable_old_logprobs_recompute=True, use the pre-computed old_log_probs from batch
            return data.batch["old_log_probs"]

        sample_uuids = data.non_tensor_batch["sample_uuid"]

        # Check first sample_uuid for efficiency - if it exists, all likely exist
        first_uuid = sample_uuids[0]
        if first_uuid in self._logprobs_cache:
            # All samples likely cached, retrieve all from cache
            cached_old_log_probs = []

            for sample_uuid in sample_uuids:
                cached_old_log_probs.append(self._logprobs_cache[sample_uuid])

            old_log_probs = torch.cat(cached_old_log_probs, dim=0).to(current_platform.device_type)
        else:
            # Cache miss - use current log_probs as old_log_probs (mathematically equivalent in on-policy)
            old_log_probs = log_probs.detach()
            if self.pipeline_config.ppo_epochs > 1:
                for i, sample_uuid in enumerate(sample_uuids):
                    self._logprobs_cache[sample_uuid] = old_log_probs[i : i + 1].cpu()

        return old_log_probs

    @staticmethod
    def _to_int_list(values: object) -> List[int]:
        if values is None:
            return []
        if isinstance(values, torch.Tensor):
            return [int(v) for v in values.detach().cpu().tolist()]
        if hasattr(values, "tolist"):
            return [int(v) for v in values.tolist()]
        if isinstance(values, (list, tuple)):
            return [int(v) for v in values]
        return [int(values)]

    def _build_token_versions(self, data: DataProto) -> torch.Tensor:
        response_mask = data.batch["response_mask"][:, 1:]
        versions = torch.full_like(response_mask, fill_value=-1, dtype=torch.long)
        create_steps = data.non_tensor_batch.get("create_step")
        segment_steps_batch = data.non_tensor_batch.get("segment_infer_steps")
        segment_lengths_batch = data.non_tensor_batch.get("segment_lengths")

        for sample_idx in range(response_mask.shape[0]):
            response_positions = torch.nonzero(response_mask[sample_idx], as_tuple=False).flatten()
            if response_positions.numel() == 0:
                continue

            fallback_step = -1
            if create_steps is not None:
                fallback_step = int(self._to_int_list(create_steps[sample_idx])[0])
            elif "global_step" in data.meta_info:
                fallback_step = int(data.meta_info["global_step"])

            use_segment_plan = False
            if segment_steps_batch is not None and segment_lengths_batch is not None:
                segment_steps = self._to_int_list(segment_steps_batch[sample_idx])
                segment_lengths = self._to_int_list(segment_lengths_batch[sample_idx])
                if len(segment_steps) == len(segment_lengths) and len(segment_lengths) > 0:
                    if sum(segment_lengths) == int(response_positions.numel()):
                        cursor = 0
                        for segment_step, segment_length in zip(segment_steps, segment_lengths):
                            if segment_length <= 0:
                                continue
                            segment_positions = response_positions[cursor: cursor + segment_length]
                            cursor += segment_length
                            if segment_positions.numel() > 0:
                                versions[sample_idx, segment_positions] = int(segment_step)
                        use_segment_plan = True

            if not use_segment_plan and fallback_step >= 0:
                versions[sample_idx, response_positions] = fallback_step

        return versions

    def get_behavior_log_probs(self, data: DataProto, old_log_probs: torch.Tensor) -> torch.Tensor:
        behavior_log_probs = data.batch.get("infer_logprobs")
        if behavior_log_probs is None or len(behavior_log_probs) == 0:
            return old_log_probs
        return behavior_log_probs

    def use_interpolated_prox_ref(self) -> bool:
        return self._resolved_proximal_approx_mode() != "disabled"

    def use_behavior_for_ppo_ratio_ref(self) -> bool:
        return getattr(self.pipeline_config, "old_log_probs_source", "current_recompute") == "current_recompute_ppo_infer"

    def _resolved_proximal_approx_mode(self) -> str:
        source = getattr(self.pipeline_config, "old_log_probs_source", "current_recompute")
        if source == "interp_prox_loglinear":
            return "log_linear"
        if source == "interp_prox_linear":
            return "linear"
        return getattr(self.pipeline_config, "proximal_logprob_approx", "disabled")

    def get_proximal_log_probs(
        self,
        data: DataProto,
        log_probs: torch.Tensor,
        behavior_log_probs: torch.Tensor,
    ) -> torch.Tensor:
        approx_mode = self._resolved_proximal_approx_mode()
        if approx_mode == "disabled":
            return behavior_log_probs

        current_version = int(data.meta_info.get("global_step", 0))
        proximal_version = current_version - 1
        versions = self._build_token_versions(data).to(device=log_probs.device, dtype=log_probs.dtype)

        generated_tokens_mask = versions >= 0
        version_diff = float(current_version) - versions
        version_gap = float(proximal_version) - versions
        alpha = torch.where(
            (version_diff > 0) & generated_tokens_mask,
            version_gap / version_diff,
            torch.zeros_like(versions),
        )
        alpha = torch.clamp(alpha, 0.0, 1.0)

        if approx_mode == "log_linear":
            proximal_log_probs = behavior_log_probs + alpha * (log_probs - behavior_log_probs)
        elif approx_mode == "linear":
            p_behavior = behavior_log_probs.exp()
            p_theta = log_probs.exp()
            proximal_log_probs = torch.log((1.0 - alpha) * p_behavior + alpha * p_theta + 1e-10)
        elif approx_mode == "behavior":
            proximal_log_probs = behavior_log_probs
        else:
            raise ValueError(f"Unsupported proximal_logprob_approx: {approx_mode}")

        return torch.where(generated_tokens_mask, proximal_log_probs, behavior_log_probs)

    def get_ppo_ratio_ref_log_probs(
        self,
        policy_ref_log_probs: torch.Tensor,
        behavior_log_probs: torch.Tensor,
    ) -> torch.Tensor:
        if self.use_behavior_for_ppo_ratio_ref():
            return behavior_log_probs
        return policy_ref_log_probs

    def resolve_train_infer_correction(
        self,
        data: DataProto,
        response_mask: torch.Tensor,
        old_log_probs: torch.Tensor,
        infer_log_probs: torch.Tensor,
        global_valid_samples: int,
        global_valid_tokens: int,
    ):
        """Reuse pipeline-computed correction when available; otherwise compute locally."""
        precomputed_is_weight = data.batch.get("train_infer_is_weight")
        if precomputed_is_weight is not None:
            if precomputed_is_weight.shape != response_mask.shape:
                raise ValueError(
                    "train_infer_is_weight shape mismatch: "
                    f"expected {response_mask.shape}, got {precomputed_is_weight.shape}"
                )
            return precomputed_is_weight, torch.ones_like(response_mask), {}

        from roll.utils.train_infer_corrections import compute_train_infer_correction

        return compute_train_infer_correction(
            cfg=self.pipeline_config.train_infer_correction,
            response_mask=response_mask,
            old_log_probs=old_log_probs,
            infer_log_probs=infer_log_probs,
            global_valid_samples=global_valid_samples,
            global_valid_tokens=global_valid_tokens,
        )

    def compute_joint_remaining_metrics(
        self,
        base_mask: torch.Tensor,
        filtered_mask: torch.Tensor,
        clipped_mask: torch.Tensor,
        prefix: str = "actor/train_infer_ppo_remaining",
    ) -> Dict[str, float]:
        """Measure how much data remains after train/infer filtering and PPO clipping.

        `base_mask` should be the original token mask before train/infer filtering.
        `filtered_mask` should be the mask actually used by PPO after filtering.
        `clipped_mask` marks tokens whose PPO ratio falls outside the clip range.
        """
        base_mask = base_mask.long()
        filtered_mask = filtered_mask.long()
        clipped_mask = clipped_mask.float()

        effective_mask = filtered_mask.float() * (1.0 - clipped_mask)
        metrics = {
            f"{prefix}_mask_mean": masked_mean(effective_mask, base_mask).detach().item(),
        }

        base_sample_mask = (base_mask.sum(dim=-1) > 0).float()
        effective_sample_mask = (effective_mask.sum(dim=-1) > 0).float()
        metrics[f"{prefix}_sample_mean"] = masked_mean(
            effective_sample_mask, base_sample_mask
        ).detach().item()
        return metrics

    def loss_func(self, data: DataProto, output_tensor: torch.Tensor):
        """
        loss func接口定义:
            data: DataProto, 由train_step透传
            output_tensor: torch.Tensor, model.forward()的输出Tensor
        """

        base_response_mask = data.batch["response_mask"][:, 1:].long()
        response_mask = base_response_mask
        ref_log_probs = data.batch["ref_log_probs"]
        advantages = data.batch["advantages"]

        batch_num_tokens = data.meta_info['batch_num_tokens']
        global_valid_samples = data.meta_info['global_valid_samples']

        log_probs = self.strategy.op_compute_log_probs(
            logits=output_tensor, input_ids=data.batch["input_ids"], attention_mask=data.batch["response_mask"]
        )
        old_log_probs = self.get_old_log_probs_with_cache(data, log_probs)
        behavior_log_probs = self.get_behavior_log_probs(data, old_log_probs)
        policy_ref_log_probs = old_log_probs
        if self.use_interpolated_prox_ref():
            policy_ref_log_probs = self.get_proximal_log_probs(data, log_probs, behavior_log_probs)

        ratio = (log_probs - policy_ref_log_probs).exp()

        pg_clip_low = (
            self.pipeline_config.pg_clip_low
            if self.pipeline_config.use_pg_clip_range
            else self.pipeline_config.pg_clip
        )
        pg_clip_high = (
            self.pipeline_config.pg_clip_high
            if self.pipeline_config.use_pg_clip_range
            else self.pipeline_config.pg_clip
        )
        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - pg_clip_low, 1 + pg_clip_high) * advantages
        pg_loss = -torch.min(surr1, surr2)
        if self.pipeline_config.dual_clip_loss:
            dual_clip_loss = -torch.max(-pg_loss, (1 + self.pipeline_config.pg_clip * 2) * advantages)
            pg_loss = torch.where(advantages < 0, dual_clip_loss, pg_loss)

        pg_loss = agg_loss(loss_mat=pg_loss, loss_mask=response_mask, loss_agg_mode=self.pipeline_config.loss_agg_mode,
                           batch_num_tokens=batch_num_tokens['response_mask'],
                           global_valid_samples=global_valid_samples['response_mask'])

        kl_loss = compute_approx_kl(
            log_probs=log_probs, log_probs_base=ref_log_probs, action_mask=response_mask, kl_penalty="k3"
        )
        kl_loss = agg_loss(loss_mat=kl_loss, loss_mask=response_mask, loss_agg_mode=self.pipeline_config.loss_agg_mode,
                           batch_num_tokens=batch_num_tokens['response_mask'],
                           global_valid_samples=global_valid_samples['response_mask'])

        approxkl = compute_approx_kl(
            log_probs=log_probs, log_probs_base=policy_ref_log_probs, action_mask=response_mask, kl_penalty="mse"
        )
        policykl = compute_approx_kl(
            log_probs=log_probs, log_probs_base=policy_ref_log_probs, action_mask=response_mask, kl_penalty="kl"
        )
        clipped_low = (ratio < 1 - pg_clip_low).float()
        clipped_high = (ratio > 1 + pg_clip_high).float()
        clipped = (clipped_low + clipped_high).float()
        active_clipped = torch.lt(surr2, surr1).float()
        joint_remaining_metrics = self.compute_joint_remaining_metrics(
            base_mask=base_response_mask,
            filtered_mask=response_mask,
            clipped_mask=active_clipped,
        )

        if self.pipeline_config.use_kl_loss:
            total_loss = pg_loss + kl_loss * self.pipeline_config.kl_loss_coef
        else:
            total_loss = pg_loss
        if self.pipeline_config.entropy_loss_coef > 0:
            entropy = self.strategy.op_compute_entropy(
                logits=output_tensor, attention_mask=data.batch["response_mask"]
            )
            entropy_loss = agg_loss(
                loss_mat=entropy,
                loss_mask=response_mask,
                loss_agg_mode=self.pipeline_config.loss_agg_mode,
                batch_num_tokens=batch_num_tokens['response_mask'],
                global_valid_samples=global_valid_samples['response_mask'],
            )
            total_loss = total_loss - entropy_loss * self.pipeline_config.entropy_loss_coef

        pg_metrics = {
            "actor/ppo_ratio_high_clipfrac": clipped_high.mean().detach().item(),
            "actor/ppo_ratio_low_clipfrac": clipped_low.mean().detach().item(),
            "actor/ppo_ratio_clipfrac": clipped.mean().detach().item(),
            "actor/ratio_mean": masked_mean(ratio, response_mask, dim=-1).mean().detach().item(),
            "actor/ratio_max": torch.max(ratio * response_mask).detach().item(),
            "actor/ratio_min": torch.min(ratio * response_mask + (1 - response_mask) * 1e10).detach().item(),
            "actor/clipfrac": agg_loss(
                loss_mat=active_clipped,
                loss_mask=response_mask,
                loss_agg_mode=self.pipeline_config.loss_agg_mode,
                batch_num_tokens=batch_num_tokens['response_mask'],
                global_valid_samples=global_valid_samples['response_mask'],
            )
            .detach()
            .item(),
            **joint_remaining_metrics,
            "actor/pg_loss": pg_loss.detach().item(),
            "actor/kl_loss": kl_loss.detach().item(),
            "actor/total_loss": total_loss.detach().item(),
            "actor/approxkl": agg_loss(
                loss_mat=approxkl, loss_mask=response_mask, loss_agg_mode=self.pipeline_config.loss_agg_mode,
                batch_num_tokens=batch_num_tokens['response_mask'],
                global_valid_samples=global_valid_samples['response_mask'],
            )
            .detach()
            .item(),
            "actor/policykl": agg_loss(
                loss_mat=policykl, loss_mask=response_mask, loss_agg_mode=self.pipeline_config.loss_agg_mode,
                batch_num_tokens=batch_num_tokens['response_mask'],
                global_valid_samples=global_valid_samples['response_mask'],
            )
            .detach()
            .item(),
        }

        return total_loss, pg_metrics

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def do_checkpoint(self, global_step, is_last_step=None):
        if self.worker_config.offload_nccl:
            reload_process_groups()
        with Timer("do_checkpoint") as total_timer:
            ckpt_id = f"checkpoint-{global_step}"

            # actor train是直接存在save dir目录下的，其他role是存在save_dir/cluster_name下的
            save_dir = os.path.join(self.pipeline_config.output_dir, self.worker_name, ckpt_id)
            self.logger.info(f"save checkpoint-{global_step} to {save_dir}")

            # could be passed for other strategy with kwargs
            exec_metrics: Dict = self.strategy.save_checkpoint(
                save_dir, global_step, ckpt_id, is_last_step=is_last_step
            )

        metrics = {
            f"time/{self.cluster_name}/do_checkpoint/total": total_timer.last,
        }
        metric_prefix = f"time/{self.cluster_name}/do_checkpoint"
        metrics.update({f"{metric_prefix}/{k}": v for k, v in exec_metrics.items()})
        output = DataProto(meta_info={"metrics": metrics})
        return output


class InferWorker(Worker):
    def __init__(self, worker_config: WorkerConfig):
        super().__init__(worker_config=worker_config)
        self.tokenizer = None
        self.strategy = None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    async def initialize(self, pipeline_config):
        super().initialize(pipeline_config)

        self.strategy = create_strategy(worker=self)

        await self.strategy.initialize(model_provider=default_actor_model_provider)
        self.tokenizer = getattr(self.strategy, "tokenizer")
        self.logger.info(f"{self.worker_name} initialized")

        await self.strategy.offload_states()

        # Platform must have been initialized when calling current_platform.reset_max_memory_allocated
        # with arguments (inside state_offload_manager). We explicitly init platform here because
        # current process is used as engine client when using vllm v1 engine, and
        # there is no chance to init platform context.
        current_platform.init()

    def get_url(self):
        return self.strategy.get_url()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    async def load_states(self, *args, **kwargs):
        await self.strategy.load_states(*args, **kwargs)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    async def offload_states(self, *args, **kwargs):
        await self.strategy.offload_states(*args, **kwargs)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    async def load_states_partial(self, target_dp_ranks: List[int]):
        """Load states for workers whose dp_rank is in target_dp_ranks."""

        # Log entry memory (only for TP rank 0 to reduce log spam)
        if self.rank_info.tp_rank == 0:
            log_gpu_memory_usage(
                head=f"Worker {self.rank} (DP {self.rank_info.dp_rank}) load_states_partial_entry",
                logger=self.logger,
                rank=None
            )

        assert getattr(self, "strategy", None) is not None, "worker has no strategy to load"
        if self.rank_info.dp_rank in target_dp_ranks:
            # AST: AST_PRECONDITION(is_model_in_gpu is False) - verify strategy offloaded before load
            is_loaded = self.strategy.is_model_in_gpu()

            assert is_loaded is False, (
                    f"Pre-condition: strategy must be offloaded before load_states_partial, "
                    f"got Worker {self.rank} (DP {self.rank_info.dp_rank}) is_model_in_gpu={is_loaded}"
                )

            await self.strategy.load_states()
            self.logger.info(f"Worker {self.rank} (DP {self.rank_info.dp_rank}) loaded states")
        else:
            self.logger.debug(f"Worker {self.rank} (DP {self.rank_info.dp_rank}) skipped load")


        # Log exit memory (only for TP rank 0 to reduce log spam)
        if self.rank_info.tp_rank == 0:
            log_gpu_memory_usage(
                head=f"Worker {self.rank} (DP {self.rank_info.dp_rank}) load_states_partial_exit",
                logger=self.logger,
                rank=None
            )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    async def offload_states_partial(self, target_dp_ranks: List[int]):
        """Offload states for workers whose dp_rank is in target_dp_ranks."""

        # Log entry memory (only for TP rank 0 to reduce log spam)
        if self.rank_info.tp_rank == 0:
            log_gpu_memory_usage(
                head=f"Worker {self.rank} (DP {self.rank_info.dp_rank}) offload_states_partial_entry",
                logger=self.logger,
                rank=None
            )

        assert getattr(self, "strategy", None) is not None, "worker has no strategy to offload"
        if self.rank_info.dp_rank in target_dp_ranks:
            # AST: AST_PRECONDITION(is_model_in_gpu is True) - verify strategy loaded before offload
            is_loaded = self.strategy.is_model_in_gpu

            assert is_loaded is True, (
                    f"Pre-condition: strategy must be loaded before offload_states_partial, "
                    f"got Worker {self.rank} (DP {self.rank_info.dp_rank}) is_model_in_gpu={is_loaded}"
                )

            await self.strategy.offload_states()
            self.logger.info(f"Worker {self.rank} (DP {self.rank_info.dp_rank}) offloaded states")
        else:
            self.logger.debug(f"Worker {self.rank} (DP {self.rank_info.dp_rank}) skipped offload")


        # Log exit memory and verify offload success (only for TP rank 0 to reduce log spam)
        if self.rank_info.tp_rank == 0:
            log_gpu_memory_usage(
                head=f"Worker {self.rank} (DP {self.rank_info.dp_rank}) offload_states_partial_exit",
                logger=self.logger,
                rank=None
            )

            # Verify offloaded workers have near-zero GPU memory usage
            if self.rank_info.dp_rank in target_dp_ranks:
                import torch
                gpu_memory_gb = torch.cuda.memory_allocated() / 1024**3
                if gpu_memory_gb > 1.0:
                    raise RuntimeError(
                        f"GPU memory not properly offloaded for Worker {self.rank} (DP {self.rank_info.dp_rank}): "
                        f"{gpu_memory_gb:.2f} GB still allocated (expected < 1 GB after offload)"
                    )


    async def broadcast_parameter(self, *args, **kwargs):
        await self.strategy.broadcast_parameter(*args, **kwargs)

    async def setup_collective_group(self, *args, **kwargs):
        await self.strategy.setup_collective_group(*args, **kwargs)

    async def start_model_update(self, *args, **kwargs):
        raise NotImplementedError

    async def update_parameter_in_bucket(self, *args, **kwargs):
        await self.strategy.update_parameter_in_bucket(*args, **kwargs)

    async def add_lora(self, *args, **kwargs):
        await self.strategy.add_lora(*args, **kwargs)

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE)
    async def generate(self, data: DataProto):
        """
        batch = TensorDict(
            {
                'prompts': idx,
                'responses': response,
                'input_ids': seq,  # here input_ids become the whole sentences
                'attention_mask': attention_mask,
                'position_ids': position_ids,
                'old_log_probs': log_probs,
            },
            batch_size=batch_size)
        return DataProto(batch=batch)
        """
        if "generation_config" not in data.meta_info:
            generation_config = self.worker_config.generating_args.to_dict()
        else:
            generation_config = data.meta_info["generation_config"]

        generation_config["eos_token_id"] = [self.tokenizer.eos_token_id, self.tokenizer.pad_token_id]
        generation_config["pad_token_id"] = self.tokenizer.pad_token_id

        global_step = data.meta_info.get("global_step", 0)
        self.logger.info(f"{self.worker_name} generate global step {global_step}")

        data = data.to("cuda")
        data.meta_info["micro_batch_size"] = self.worker_config.infer_batch_size

        output = await self.strategy.generate(batch=data, generation_config=generation_config)
        output = postprocess_generate(
            prompts=data,
            output=output,
            num_return_sequences=generation_config["num_return_sequences"],
            sequence_length=self.pipeline_config.sequence_length,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        data.to("cpu")
        output = output.to("cpu")
        return output

    async def generate_request(self, payload: Dict) -> Dict:
        """
        payload: {
            input_ids": list[int],
            Optinal(multi_modal_data): dict[prompt_token_ids: list[int], multi_modal_data: dict[iamge, ...]],
            rid: str,
            sampling_params: dict,
            Optional(**strategy_specific_fields), # e.g. return_logprob for sglang
        }
        """
        return await self.strategy.generate_request(payload=payload)

    async def abort_requests(self, request_ids):
        await self.strategy.abort_requests(request_ids)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    async def process_weights_after_loading(self):
        if getattr(self, "strategy", None) is not None:
            await self.strategy.process_weights_after_loading()


class CriticWorker(Worker):

    def __init__(self, worker_config: WorkerConfig):
        super().__init__(worker_config=worker_config)
        self.tokenizer = None
        self.strategy: Optional[Union[InferenceStrategy, TrainStrategy]] = None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def initialize(self, pipeline_config):
        super().initialize(pipeline_config)

        self.strategy = create_strategy(worker=self)

        self.strategy.initialize(model_provider=default_value_model_provider)
        self.tokenizer = self.strategy.tokenizer

        if self.pipeline_config.resume_from_checkpoint:
            load_dir = os.path.join(download_model(self.pipeline_config.resume_from_checkpoint), self.cluster_name)
            self.strategy.load_checkpoint(load_dir=load_dir, tag="checkpoint")

        self.logger.info(f"{self.worker_name} initialized")

        self.strategy.offload_states()

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE)
    def compute_values(self, data: DataProto):
        """
        return DataProto.from_dict(tensors={'values': values})
        """
        global_step = data.meta_info.get("global_step", 0)
        is_offload_states = data.meta_info.get("is_offload_states", True)
        metrics = {}
        with state_offload_manger(
            strategy=self.strategy,
            metrics=metrics,
            metric_infix=f"{self.cluster_name}/compute_values",
            is_offload_states=is_offload_states,
            load_kwargs={"include": [OffloadStateType.model_params]},
        ):
            data = data.to(current_platform.device_type)
            data.meta_info["micro_batch_size"] = self.worker_config.infer_batch_size
            with torch.no_grad():
                results: Dict[str, torch.Tensor] = self.strategy.forward_step(
                    batch=data, forward_func=self.forward_func_values
                )

            output = DataProto.from_dict(tensors={"values": results["values"]})
            data.to("cpu")
            output = output.to("cpu")

        output.meta_info = {"metrics": metrics}
        return output

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE)
    def train_step(self, data: DataProto):
        """
        return DataProto(meta_info={'metrics': metrics})
        """
        global_step = data.meta_info.get("global_step", 0)
        is_offload_states = data.meta_info.get("is_offload_states", True)
        metrics = {}
        with state_offload_manger(
            strategy=self.strategy,
            metrics=metrics,
            metric_infix=f"{self.cluster_name}/train_step",
            is_offload_states=is_offload_states,
            load_kwargs={"include": [OffloadStateType.model_params, OffloadStateType.other_params]},
        ):
            data = data.to(current_platform.device_type)
            per_device_train_batch_size = self.worker_config.training_args.per_device_train_batch_size
            backward_batch_size = (
                per_device_train_batch_size * self.worker_config.training_args.gradient_accumulation_steps
            )

            dataloader = data.make_iterator(
                mini_batch_size=backward_batch_size,
                epochs=1,
                seed=self.pipeline_config.seed,
                dataloader_kwargs={"shuffle": True},
            )

            for batch_idx, data in tqdm(
                enumerate(dataloader),
                desc=f"{self.worker_name} train global step {global_step}",
                total=data.batch.batch_size[0] * self.pipeline_config.ppo_epochs // backward_batch_size,
            ):
                vf_metrics = self.strategy.train_step(batch=data, loss_func=self.loss_func)
                append_to_dict(metrics, vf_metrics)

            data.to("cpu")
            metrics["critic/lr"] = self.strategy.scheduler.get_last_lr()[0]

        output = DataProto(meta_info={"metrics": metrics}).to("cpu")

        return output

    def loss_func(self, data: DataProto, output_tensor: torch.Tensor):
        """
        loss func接口定义:
            data: DataProto, 由train_step透传
            output_tensor: torch.Tensor, model.forward()的输出Tensor
        """
        response_mask = data.batch["response_mask"][:, 1:]
        old_values = data.batch["values"]
        returns = data.batch["returns"]

        values, _ = self.forward_func_values(data=data, output_tensor=output_tensor)

        if self.pipeline_config.value_clip is not None:
            values_clipped = torch.clip(
                values,
                old_values - self.pipeline_config.value_clip,
                old_values + self.pipeline_config.value_clip,
            )
            surr1 = (values - returns) ** 2
            surr2 = (values_clipped - returns) ** 2
            vf_clipfrac = masked_mean(torch.gt(surr2, surr1).float(), response_mask, dim=-1).mean()
            loss = torch.max(surr1, surr2)
        else:
            loss = (values - returns) ** 2
            vf_clipfrac = masked_mean(loss, response_mask, dim=-1).mean()

        vf_loss = 0.5 * masked_mean(loss, response_mask, dim=-1).mean()

        vf_metrics = {
            "critic/loss": vf_loss.detach().item(),
            "critic/value": (masked_mean(old_values, response_mask, dim=-1)).mean().detach().item(),
            "critic/vpred": (masked_mean(values, response_mask, dim=-1)).mean().detach().item(),
            "critic/clipfrac": vf_clipfrac.detach().item(),
            "critic/error": masked_mean((values - returns) ** 2, response_mask, dim=-1).mean().detach().item(),
        }

        return vf_loss, vf_metrics

    def forward_func_values(self, data: DataProto, output_tensor: torch.Tensor):
        values = output_tensor[:, :-1]
        values = values.squeeze(dim=-1)
        return values, {"values": values.clone().detach()}

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def do_checkpoint(self, global_step, is_last_step=None):
        with Timer("do_checkpoint") as total_timer:
            ckpt_id = f"checkpoint-{global_step}"
            save_dir = os.path.join(self.pipeline_config.output_dir, self.worker_name, ckpt_id, self.cluster_name)
            critic_save_dir = os.path.join(self.pipeline_config.output_dir, self.worker_name, ckpt_id)
            self.logger.info(f"save checkpoint-{global_step} to {save_dir}")
            exec_metrics: Dict = self.strategy.save_checkpoint(
                save_dir, global_step, ckpt_id, local_state_path=critic_save_dir, is_last_step=is_last_step
            )

        metrics = {
            f"time/{self.cluster_name}/do_checkpoint/total": total_timer.last,
        }
        metric_prefix = f"time/{self.cluster_name}/do_checkpoint"
        metrics.update({f"{metric_prefix}/{k}": v for k, v in exec_metrics.items()})
        output = DataProto(meta_info={"metrics": metrics})
        return output


class RewardWorker(Worker):
    """
    Reward Model 使用 AutoModelForSequenceClassification 协议
    """

    def __init__(self, worker_config: WorkerConfig):
        super().__init__(worker_config=worker_config)
        self.tokenizer = None
        self.strategy: Optional[Union[InferenceStrategy, TrainStrategy]] = None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def initialize(self, pipeline_config):
        super().initialize(pipeline_config)

        self.strategy = create_strategy(worker=self)

        self.strategy.initialize(model_provider=default_reward_model_provider)
        self.tokenizer = self.strategy.tokenizer

        self.logger.info(f"{self.worker_name} initialized")
        self.strategy.offload_states()

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE, clear_cache=False)
    def compute_rewards(self, data: DataProto):
        """
        return DataProto.from_dict(tensors={'rewards': rewards})
        """
        global_step = data.meta_info.get("global_step", 0)
        is_offload_states = data.meta_info.get("is_offload_states", True)
        metrics = {}
        with state_offload_manger(
            strategy=self.strategy,
            metrics=metrics,
            metric_infix=f"{self.cluster_name}/compute_rewards",
            is_offload_states=is_offload_states,
        ):
            data = data.to(current_platform.device_type)

            # TODO: _switch_chat_template, 异构reward model

            data.meta_info["micro_batch_size"] = self.worker_config.infer_batch_size
            with torch.no_grad():
                results: Dict[str, torch.Tensor] = self.strategy.forward_step(
                    batch=data, forward_func=self.forward_func_values
                )
            token_level_rewards = results["values"]  # (bsz, input_ids.shape[1]-1)
            input_ids = data.batch["input_ids"][:, 1:]
            seq_lengths = torch.eq(input_ids, self.tokenizer.pad_token_id).int().argmax(-1) - 1
            seq_lengths = (seq_lengths % input_ids.shape[-1]).to(token_level_rewards.device)
            response_level_rewards = token_level_rewards[
                torch.arange(seq_lengths.shape[0], device=token_level_rewards.device), seq_lengths
            ]

            output = DataProto.from_dict(
                tensors={"token_level_rewards": token_level_rewards, "response_level_rewards": response_level_rewards}
            )

            data.to("cpu")
            output = output.to("cpu")

        output.meta_info = {"metrics": metrics}
        return output

    def forward_func_values(self, data: DataProto, output_tensor: torch.Tensor):
        values = output_tensor[:, 1:]
        values = values.squeeze(dim=-1)
        return values, {"values": values.clone().detach()}
