import os
import re
import shutil
import time
from collections import defaultdict
from concurrent import futures
from typing import Any, Dict, List

import numpy as np
import ray
import torch
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from transformers import set_seed

from roll.distributed.executor.cluster import Cluster
from roll.distributed.executor.model_update_group import ModelUpdateGroup
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.scheduler.resource_manager import ResourceManager
from roll.utils.checkpoint_manager import CheckpointManager, download_model
from roll.utils.dynamic_batching import dynamic_batching_shard
from roll.utils.functionals import append_to_dict, reduce_metrics
from roll.utils.logging import get_logger
from roll.utils.tracking import create_tracker
from roll.utils.worker_state import WorkerState

logger = get_logger()


class BasePipeline:
    model_update_groups: List[ModelUpdateGroup] = []
    checkpoint_clusters: List = []
    EWMA_PROX_RESET_PATIENCE_STEPS = 10

    def __init__(self, pipeline_config):
        set_seed(seed=pipeline_config.seed)
        self.pipeline_config = pipeline_config
        self.resource_manager = ResourceManager(
            num_nodes=self.pipeline_config.num_nodes, num_gpus_per_node=self.pipeline_config.num_gpus_per_node
        )
        self.state = WorkerState()
        self.checkpoint_manager = CheckpointManager(checkpoint_config=self.pipeline_config.checkpoint_config)
        self.tracker = create_tracker(
            tracker_name=self.pipeline_config.track_with,
            config=self.pipeline_config.to_dict(),
            **self.pipeline_config.tracker_kwargs,
        )
        self.resume_from_checkpoint = False
        self.executor: futures.ThreadPoolExecutor = futures.ThreadPoolExecutor(max_workers=5)
        self.resume_futures = []
        self._last_evicted_snapshot_step = -1

        if self.pipeline_config.resume_from_checkpoint:
            self.resume_from_checkpoint = download_model(self.pipeline_config.resume_from_checkpoint)

            logger.info(f"resume_from_checkpoint: {self.resume_from_checkpoint}")
            load_dir = os.path.join(self.resume_from_checkpoint, "pipeline")
            self.state = WorkerState.load_from_json(load_dir=load_dir, tag="pipeline")

            def resume_metrics():
                for metrics in self.state.log_history:
                    self.tracker.log(values=metrics, step=metrics["system/step"])

            self.resume_futures.append(self.executor.submit(resume_metrics))

    def run(self):
        pass

    def set_model_update_pair(self, src_cluster, tgt_cluster, frequency=1):
        self.model_update_groups.append(
            ModelUpdateGroup(src_cluster=src_cluster, tgt_cluster=tgt_cluster, frequency=frequency, pipeline_config=self.pipeline_config)
        )

    def set_checkpoint_clusters(self, *clusters):
        self.checkpoint_clusters.extend(clusters)

    def model_update(self, global_step):
        metrics = {}
        for model_update_group in self.model_update_groups:
            metrics.update(model_update_group.model_update(global_step))
            model_update_group.tgt_cluster.process_weights_after_loading()
        return metrics

    def uses_ewma_old_log_probs(self) -> bool:
        return getattr(self.pipeline_config, "old_log_probs_source", "current_recompute") in ("ewma_prox", "ppo_ewma_async")

    def snapshot_overhead_metrics_enabled(self) -> bool:
        return bool(getattr(self.pipeline_config, "enable_snapshot_overhead_metrics", False))

    def should_update_ewma_on_model_update(self) -> bool:
        return getattr(self.pipeline_config, "old_log_probs_source", "current_recompute") == "ppo_ewma_async"

    def should_update_ewma_after_train(self) -> bool:
        return getattr(self.pipeline_config, "old_log_probs_source", "current_recompute") == "ewma_prox"

    def save_async_snapshot(self, cluster: Cluster, global_step: int):
        if not (
            self.pipeline_config.async_pipeline
            and self.pipeline_config.enable_async_old_logprobs_snapshot
            and getattr(self.pipeline_config, "old_log_probs_source", "current_recompute") == "snapshot_old"
        ):
            return {}
        snapshot_metrics = cluster.save_snapshot(step=global_step, blocking=True)
        if isinstance(snapshot_metrics, list):
            metrics_list = [item.meta_info.get("metrics", {}) for item in snapshot_metrics if item is not None]
            merged_metrics = {}
            for item_metrics in metrics_list:
                append_to_dict(merged_metrics, item_metrics)
            return reduce_metrics(merged_metrics) if merged_metrics else {}
        return snapshot_metrics.meta_info.get("metrics", {})

    def evict_async_snapshots(self, cluster: Cluster, current_step: int):
        if not (
            self.pipeline_config.async_pipeline
            and self.pipeline_config.enable_async_old_logprobs_snapshot
            and getattr(self.pipeline_config, "old_log_probs_source", "current_recompute") == "snapshot_old"
        ):
            return
        min_snapshot_step = current_step - int(np.ceil(self.pipeline_config.async_generation_ratio))
        for step in range(self._last_evicted_snapshot_step + 1, max(0, min_snapshot_step)):
            cluster.evict_snapshot(step=step, blocking=True)
            self._last_evicted_snapshot_step = step

    @staticmethod
    def _to_int_list(values: object) -> List[int]:
        if values is None:
            return []
        if isinstance(values, np.ndarray):
            return [int(v) for v in values.tolist()]
        if isinstance(values, (list, tuple)):
            return [int(v) for v in values]
        return [int(values)]

    def _build_snapshot_token_plan(self, batch: DataProto, create_steps: np.ndarray, current_step: int):
        segment_steps_batch = batch.non_tensor_batch.get("segment_infer_steps")
        segment_lengths_batch = batch.non_tensor_batch.get("segment_lengths")
        step_to_sample_positions = defaultdict(lambda: defaultdict(list))

        for sample_idx in range(len(batch)):
            response_positions = torch.nonzero(batch.batch["response_mask"][sample_idx, 1:], as_tuple=False).flatten()
            if response_positions.numel() == 0:
                continue

            fallback_step = int(create_steps[sample_idx])
            if fallback_step > current_step:
                raise ValueError(f"Found future create_step={fallback_step} while current_step={current_step}")

            use_segment_plan = False
            if segment_steps_batch is not None and segment_lengths_batch is not None:
                segment_steps = self._to_int_list(segment_steps_batch[sample_idx])
                segment_lengths = self._to_int_list(segment_lengths_batch[sample_idx])
                if len(segment_steps) == len(segment_lengths) and len(segment_lengths) > 0:
                    if sum(segment_lengths) == response_positions.numel():
                        cursor = 0
                        for segment_step, segment_length in zip(segment_steps, segment_lengths):
                            if segment_length <= 0:
                                continue
                            if segment_step > current_step:
                                raise ValueError(
                                    f"Found future segment_infer_step={segment_step} while current_step={current_step}"
                                )
                            segment_positions = response_positions[cursor: cursor + segment_length]
                            cursor += segment_length
                            if segment_positions.numel() > 0:
                                step_to_sample_positions[int(segment_step)][sample_idx].append(segment_positions)
                        use_segment_plan = True
                    else:
                        logger.warning(
                            "segment metadata mismatch for sample %s: response_tokens=%s, segment_lengths_sum=%s; "
                            "falling back to create_step=%s",
                            sample_idx,
                            int(response_positions.numel()),
                            int(sum(segment_lengths)),
                            fallback_step,
                        )

            if not use_segment_plan:
                step_to_sample_positions[fallback_step][sample_idx].append(response_positions)

        return step_to_sample_positions

    def _truncate_snapshot_shard(
        self,
        shard: DataProto,
        indices: List[int],
        sample_positions: Dict[int, List[torch.Tensor]],
    ) -> DataProto:
        if shard.batch is None or "attention_mask" not in shard.batch or "response_mask" not in shard.batch:
            return shard

        full_seq_len = shard.batch["attention_mask"].shape[-1]
        if full_seq_len <= 1:
            return shard

        max_target_pos = max(
            int(position.max().item())
            for sample_idx in indices
            for position in sample_positions[sample_idx]
            if position.numel() > 0
        )
        # old_log_probs is aligned to input_ids[:, 1:]:
        # old_log_probs[j] corresponds to the log-prob of input_ids[j + 1].
        # To keep target position j, the truncated input must therefore retain tokens up to j + 1,
        # i.e. required_input_len = j + 2 in the original input_ids space.
        required_input_len = min(full_seq_len, max_target_pos + 2)
        required_output_len = max(required_input_len - 1, 0)

        for key, tensor in list(shard.batch.items()):
            if not isinstance(tensor, torch.Tensor) or tensor.ndim == 0 or tensor.shape[0] != len(shard):
                continue
            if tensor.shape[-1] == full_seq_len:
                shard.batch[key] = tensor[..., :required_input_len]
            elif tensor.shape[-1] == full_seq_len - 1:
                shard.batch[key] = tensor[..., :required_output_len]

        response_mask = torch.zeros_like(shard.batch["response_mask"])
        for local_idx, sample_idx in enumerate(indices):
            token_positions = torch.cat(sample_positions[sample_idx]).to(dtype=torch.long)
            response_mask[local_idx, token_positions + 1] = True
        shard.batch["response_mask"] = response_mask
        return shard

    @staticmethod
    def _clear_dynamic_batching_meta(batch: DataProto):
        for key in (
            "dynamic_batching_origin_local_indices",
            "dynamic_batching_rank_batch_sizes",
            "dynamic_batching_rank_origin_local_indices",
            "dynamic_batching_rank_micro_batch_indices",
            "dynamic_batching_rank_micro_batch_lengths",
            "dynamic_batching_rank_num_micro_batchs",
            "global_micro_batch_indices",
            "global_micro_batch_lengths",
            "micro_batch_indices",
            "micro_batch_lengths",
            "num_micro_batchs",
        ):
            batch.meta_info.pop(key, None)

    def _prepare_snapshot_shard_for_log_probs(self, shard: DataProto, metrics: Dict[str, List[float]]) -> DataProto:
        self._clear_dynamic_batching_meta(shard)

        actor_cfg = self.pipeline_config.actor_train
        if not actor_cfg.use_dynamic_batching_in_infer:
            shard.meta_info["force_disable_dynamic_batching_in_infer"] = True
            logger.info(
                "snapshot_old shard dynamic batching disabled: shard_size=%s, actor_train_dp_size=%s",
                len(shard),
                self.actor_train.dp_size,
            )
            return shard

        if shard.batch is not None and "attention_mask" in shard.batch:
            input_len = int(shard.batch["attention_mask"].shape[-1])
            round_to = int(actor_cfg.sequence_length_round_in_infer)
            padded_input_len = ((input_len + round_to - 1) // round_to) * round_to
            if padded_input_len > input_len:
                padded_output_len = max(padded_input_len - 1, 0)
                for key, tensor in list(shard.batch.items()):
                    if not isinstance(tensor, torch.Tensor) or tensor.ndim == 0 or tensor.shape[0] != len(shard):
                        continue
                    if tensor.shape[-1] == input_len:
                        pad_width = padded_input_len - input_len
                    elif tensor.shape[-1] == input_len - 1:
                        pad_width = padded_output_len - (input_len - 1)
                    else:
                        continue
                    if pad_width > 0:
                        shard.batch[key] = torch.nn.functional.pad(tensor, (0, pad_width), value=0)

        effective_dp_size = self.actor_train.dp_size
        if len(shard) < effective_dp_size:
            logger.info(
                "snapshot_old shard fallback to plain forward: shard_size=%s < actor_train_dp_size=%s",
                len(shard),
                effective_dp_size,
            )
            pad_count = effective_dp_size - len(shard)
            pad_indices = list(range(len(shard))) + [len(shard) - 1] * pad_count
            padded_shard = shard.select_idxs(pad_indices)
            padded_shard.meta_info = dict(shard.meta_info)
            padded_shard.meta_info["force_disable_dynamic_batching_in_infer"] = True
            padded_shard.meta_info["snapshot_shard_original_size"] = len(shard)
            padded_shard.meta_info["snapshot_shard_pad_count"] = pad_count
            return padded_shard
        shard.meta_info["force_disable_dynamic_batching_in_infer"] = False
        shard, dynamic_batching_metrics = dynamic_batching_shard(
            shard,
            effective_dp_size,
            actor_cfg.max_tokens_per_microbatch_in_infer,
            actor_cfg.sequence_length_round_in_infer,
            actor_cfg.strategy_args.strategy_config.get("pipeline_model_parallel_size", 1),
            actor_cfg.strategy_args.strategy_config.get("virtual_pipeline_model_parallel_size", None),
            "actor_train/compute_log_probs_snapshot_shard",
        )
        logger.info(
            "snapshot_old shard dynamic batching applied: shard_size=%s, actor_train_dp_size=%s, rank_batch_sizes=%s, rank_num_micro_batches=%s",
            len(shard),
            effective_dp_size,
            shard.meta_info.get("dynamic_batching_rank_batch_sizes"),
            shard.meta_info.get("dynamic_batching_rank_num_micro_batchs"),
        )
        append_to_dict(metrics, dynamic_batching_metrics)
        return shard

    def compute_old_log_probs_with_snapshots(self, batch: DataProto, actor_train: Cluster, current_step: int) -> DataProto:
        snapshot_total_start = time.perf_counter()
        create_steps = batch.non_tensor_batch.get("create_step")
        if not self.pipeline_config.enable_async_old_logprobs_snapshot:
            return actor_train.compute_log_probs(batch, blocking=True)
        if create_steps is None:
            raise ValueError("snapshot_old requires create_step metadata, but none was found in batch.non_tensor_batch")

        create_steps = np.asarray(create_steps, dtype=np.int64)
        step_to_sample_positions = self._build_snapshot_token_plan(batch=batch, create_steps=create_steps, current_step=current_step)
        old_log_probs = None
        entropies = None
        metrics: Dict[str, List[float]] = {}
        version_steps = sorted(step_to_sample_positions)

        for step in version_steps:
            sample_positions = step_to_sample_positions[step]
            indices = sorted(sample_positions.keys())
            logger.info(
                "snapshot_old processing version shard: step=%s, shard_size=%s, current_step=%s",
                step,
                len(indices),
                current_step,
            )
            shard = batch.select_idxs(indices)
            shard = self._truncate_snapshot_shard(shard=shard, indices=indices, sample_positions=sample_positions)
            shard = self._prepare_snapshot_shard_for_log_probs(shard=shard, metrics=metrics)
            reorder_local_indices = shard.meta_info.get("dynamic_batching_origin_local_indices", None)
            original_shard_size = int(shard.meta_info.get("snapshot_shard_original_size", len(indices)))
            snapshot_step = None if step == current_step else step
            if snapshot_step is None:
                shard_output: DataProto = actor_train.compute_log_probs(shard, blocking=True)
            else:
                shard_output = actor_train.compute_log_probs(shard, snapshot_step=snapshot_step, blocking=True)
            if shard_output.batch is None:
                continue
            if old_log_probs is None:
                old_log_probs = torch.zeros_like(batch.batch["attention_mask"][:, 1:], dtype=shard_output.batch["log_probs"].dtype)
                entropies = torch.zeros_like(batch.batch["attention_mask"][:, 1:], dtype=shard_output.batch["entropy"].dtype)
            if reorder_local_indices is None:
                reorder_local_indices = list(range(original_shard_size))
            else:
                reorder_local_indices = list(reorder_local_indices[:original_shard_size])
            for local_idx, shard_local_idx in enumerate(reorder_local_indices):
                sample_idx = indices[shard_local_idx]
                token_positions = torch.cat(sample_positions[sample_idx]).to(dtype=torch.long)
                old_log_probs[sample_idx, token_positions] = shard_output.batch["log_probs"][local_idx, token_positions]
                entropies[sample_idx, token_positions] = shard_output.batch["entropy"][local_idx, token_positions]
            append_to_dict(metrics, shard_output.meta_info.pop("metrics", {}))

        if old_log_probs is None:
            old_log_probs = torch.zeros_like(batch.batch["attention_mask"][:, 1:], dtype=torch.float32)
            entropies = torch.zeros_like(old_log_probs)
        self.evict_async_snapshots(cluster=actor_train, current_step=current_step)
        if self.snapshot_overhead_metrics_enabled():
            historical_steps = [step for step in version_steps if step != current_step]
            max_staleness_gap = max([current_step - step for step in version_steps], default=0)
            metrics.update(
                {
                    "actor/snapshot_old/total_time_ms": [(time.perf_counter() - snapshot_total_start) * 1000.0],
                    "actor/snapshot_old/version_shard_count": [len(version_steps)],
                    "actor/snapshot_old/historical_version_shard_count": [len(historical_steps)],
                    "actor/snapshot_old/max_staleness_gap": [max_staleness_gap],
                }
            )
        return DataProto.from_dict(
            tensors={"log_probs": old_log_probs, "entropy": entropies},
            meta_info={"metrics": reduce_metrics(metrics)},
        )

    def compute_old_log_probs_with_ewma_prox(self, batch: DataProto, actor_train: Cluster) -> DataProto:
        return actor_train.compute_log_probs(batch, use_proximal=True, blocking=True)

    def compute_old_log_probs_by_source(self, batch: DataProto, actor_train: Cluster, current_step: int) -> DataProto:
        source = getattr(self.pipeline_config, "old_log_probs_source", "current_recompute")
        if source in ("ewma_prox", "ppo_ewma_async"):
            return self.compute_old_log_probs_with_ewma_prox(batch=batch, actor_train=actor_train)
        if source == "snapshot_old":
            return self.compute_old_log_probs_with_snapshots(
                batch=batch,
                actor_train=actor_train,
                current_step=current_step,
            )
        if source in ("current_recompute", "current_recompute_ppo_infer", "interp_prox_loglinear", "interp_prox_linear"):
            return actor_train.compute_log_probs(batch, blocking=True)
        raise ValueError(f"Unsupported old_log_probs_source: {source}")

    def update_ewma_proximal(self, cluster: Cluster, beta: float):
        outputs = cluster.update_proximal_ewma(beta, blocking=True)
        if isinstance(outputs, list):
            metrics_list = [item.meta_info.get("metrics", {}) for item in outputs if item is not None]
            merged_metrics = {}
            for item_metrics in metrics_list:
                append_to_dict(merged_metrics, item_metrics)
            return reduce_metrics(merged_metrics) if merged_metrics else {}
        return outputs.meta_info.get("metrics", {})

    def reset_ewma_proximal_to_current(self, cluster: Cluster):
        outputs = cluster.reset_proximal_to_current(blocking=True)
        if isinstance(outputs, list):
            metrics_list = [item.meta_info.get("metrics", {}) for item in outputs if item is not None]
            merged_metrics = {}
            for item_metrics in metrics_list:
                append_to_dict(merged_metrics, item_metrics)
            return reduce_metrics(merged_metrics) if merged_metrics else {}
        return outputs.meta_info.get("metrics", {})

    def maybe_reset_ewma_proximal(self, cluster: Cluster, metrics: Dict[str, Any]):
        if not self.uses_ewma_old_log_probs():
            return {}
        if not getattr(self.pipeline_config, "ewma_prox_reset_on_low_remaining_ratio", False):
            return {}

        remaining_ratio = metrics.get("actor/train_infer_ppo_remaining_mask_mean")
        if remaining_ratio is None:
            remaining_ratio = metrics.get("actor/train_infer_final_mask_mean")
        if remaining_ratio is None:
            return {}

        threshold = float(getattr(self.pipeline_config, "ewma_prox_reset_remaining_ratio_threshold", 0.8))
        streak_key = "ewma_prox_low_remaining_ratio_streak"
        current_streak = int(self.state.kv.get(streak_key, 0))
        if float(remaining_ratio) < threshold:
            current_streak += 1
        else:
            current_streak = 0

        reset_triggered = current_streak >= self.EWMA_PROX_RESET_PATIENCE_STEPS
        out = {
            "actor/ewma_prox_reset_remaining_ratio": float(remaining_ratio),
            "actor/ewma_prox_reset_remaining_ratio_threshold": threshold,
            "actor/ewma_prox_low_remaining_ratio_streak": current_streak,
            "actor/ewma_prox_reset_triggered": 0.0,
        }
        if reset_triggered:
            out.update(self.reset_ewma_proximal_to_current(cluster))
            out["actor/ewma_prox_reset_triggered"] = 1.0
            current_streak = 0
            out["actor/ewma_prox_low_remaining_ratio_streak"] = 0

        self.state.kv[streak_key] = current_streak
        return out

    def do_checkpoint(self, global_step, is_last_step=None):
        if is_last_step is None:
            is_last_step = global_step == self.pipeline_config.max_steps - 1

        metrics = self.state.log_history[-1]
        metrics["system/step"] = global_step
        if global_step > 0 and (
            global_step % self.pipeline_config.save_steps == 0 or global_step == self.pipeline_config.max_steps - 1
        ):
            ckpt_metrics_refss = []
            for cluster in self.checkpoint_clusters:
                ckpt_metrics_refss.append(
                    cluster.do_checkpoint(global_step=global_step, is_last_step=is_last_step, blocking=False)
                )

            for ckpt_metrics_refs in ckpt_metrics_refss:
                ckpt_metrics = DataProto.materialize_concat(data_refs=ckpt_metrics_refs)
                metrics.update(reduce_metrics(ckpt_metrics.meta_info.pop("metrics", {})))

            ckpt_id = f"checkpoint-{global_step}"
            pipeline_save_dir = os.path.join(self.pipeline_config.output_dir, "pipeline", ckpt_id)
            save_dir = os.path.join(self.pipeline_config.output_dir, "pipeline", ckpt_id, "pipeline")
            self.state.save_to_json(save_dir=save_dir, tag="pipeline")
            self.state.save_rng_state(save_dir=save_dir, tag="pipeline")
            self.checkpoint_manager.upload(ckpt_id=ckpt_id, local_state_path=pipeline_save_dir)

            # Clean up old checkpoints if max_ckpt_to_keep is set
            self._cleanup_old_checkpoints()

        futures.wait(self.resume_futures)
        self.resume_futures.clear()

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints if max_ckpt_to_keep is set."""
        max_ckpt = getattr(self.pipeline_config, 'max_ckpt_to_keep', 0)
        if max_ckpt <= 0:
            return

        output_dir = self.pipeline_config.output_dir
        if not os.path.exists(output_dir):
            return

        # Pattern to match checkpoint directories: checkpoint-{step}
        ckpt_pattern = re.compile(r'^checkpoint-(\d+)$')

        # Collect all checkpoint steps across all subdirectories
        all_ckpt_steps = set()
        for subdir in os.listdir(output_dir):
            subdir_path = os.path.join(output_dir, subdir)
            if not os.path.isdir(subdir_path):
                continue
            for item in os.listdir(subdir_path):
                match = ckpt_pattern.match(item)
                if match:
                    all_ckpt_steps.add(int(match.group(1)))

        # Sort steps and determine which to delete
        sorted_steps = sorted(all_ckpt_steps, reverse=True)
        steps_to_delete = sorted_steps[max_ckpt:]

        if not steps_to_delete:
            return

        logger.info(f"Cleaning up old checkpoints. Keeping {max_ckpt}, deleting steps: {steps_to_delete}")

        # Delete old checkpoints from all subdirectories
        for subdir in os.listdir(output_dir):
            subdir_path = os.path.join(output_dir, subdir)
            if not os.path.isdir(subdir_path):
                continue
            for step in steps_to_delete:
                ckpt_dir = os.path.join(subdir_path, f"checkpoint-{step}")
                if os.path.exists(ckpt_dir):
                    try:
                        shutil.rmtree(ckpt_dir)
                        logger.info(f"Deleted old checkpoint: {ckpt_dir}")
                    except Exception as e:
                        logger.warning(f"Failed to delete checkpoint {ckpt_dir}: {e}")

    def download_models(self, *clusters: Cluster):
        node2pg: Dict[str, PlacementGroup] = {}
        node2model_names: Dict[str, set[str]] = defaultdict(set)
        for cluster in clusters:
            assert cluster.placement_groups is not None
            for pg_list in cluster.placement_groups:
                assert len(pg_list) > 0
                worker_nodes = set()
                for pg in pg_list:
                    node_rank = pg["node_rank"]
                    if node_rank not in worker_nodes:
                        worker_nodes.add(node_rank)
                        node2pg[node_rank] = pg["placement_group"]
                        if cluster.worker_config.model_args.model_name_or_path:
                            node2model_names[node_rank].add(cluster.worker_config.model_args.model_name_or_path)
                        if self.pipeline_config.resume_from_checkpoint:
                            node2model_names[node_rank].add(self.pipeline_config.resume_from_checkpoint)
        ray.get(
            [
                download_models.options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=node2pg[node_rank])
                ).remote(model_name_or_paths=model_names)
                for node_rank, model_names in node2model_names.items()
            ]
        )

@ray.remote
def download_models(model_name_or_paths: set[str]):
    with futures.ThreadPoolExecutor(max_workers=5) as thread_executor:
        futures.wait([thread_executor.submit(download_model, model_name_or_path)
                      for model_name_or_path in model_name_or_paths])
