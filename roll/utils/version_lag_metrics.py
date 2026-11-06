from __future__ import annotations

from typing import Dict, List

import numpy as np


def _to_int_list(values: object) -> List[int]:
    if values is None:
        return []
    if isinstance(values, np.ndarray):
        return [int(v) for v in values.tolist()]
    if isinstance(values, (list, tuple)):
        return [int(v) for v in values]
    return [int(values)]


def _weighted_percentile(values: np.ndarray, weights: np.ndarray, percentile: float) -> float:
    if values.size == 0:
        return 0.0

    order = np.argsort(values, kind="stable")
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    threshold = percentile / 100.0 * weights.sum()
    idx = int(np.searchsorted(cumulative, threshold, side="left"))
    idx = min(idx, len(values) - 1)
    return float(values[idx])


def compute_version_lag_metrics(batch, global_step: int) -> Dict[str, float]:
    segment_steps_batch = batch.non_tensor_batch.get("segment_infer_steps")
    segment_lengths_batch = batch.non_tensor_batch.get("segment_lengths")
    if segment_steps_batch is None or segment_lengths_batch is None:
        return {}

    segment_lags: List[float] = []
    token_lags: List[float] = []
    token_weights: List[float] = []
    mixed_version_flags: List[float] = []
    unique_versions_per_traj: List[float] = []
    version_spans_per_traj: List[float] = []

    for step_values, length_values in zip(segment_steps_batch, segment_lengths_batch):
        infer_steps = _to_int_list(step_values)
        segment_lengths = _to_int_list(length_values)
        valid_infer_steps = []
        valid_lengths = []
        for infer_step, length in zip(infer_steps, segment_lengths):
            if infer_step >= 0 and length > 0:
                valid_infer_steps.append(infer_step)
                valid_lengths.append(length)
        if not valid_infer_steps:
            continue

        valid_pairs = [
            (global_step - infer_step, length)
            for infer_step, length in zip(valid_infer_steps, valid_lengths)
        ]

        traj_lags = np.asarray([lag for lag, _ in valid_pairs], dtype=np.float64)
        traj_lengths = np.asarray([length for _, length in valid_pairs], dtype=np.float64)
        segment_lags.extend(traj_lags.tolist())
        token_lags.extend(traj_lags.tolist())
        token_weights.extend(traj_lengths.tolist())

        unique_versions = sorted(set(valid_infer_steps))
        mixed_version_flags.append(float(len(unique_versions) > 1))
        unique_versions_per_traj.append(float(len(unique_versions)))
        version_spans_per_traj.append(float(max(unique_versions) - min(unique_versions)))

    if not segment_lags:
        return {}

    segment_lags_np = np.asarray(segment_lags, dtype=np.float64)
    token_lags_np = np.asarray(token_lags, dtype=np.float64)
    token_weights_np = np.asarray(token_weights, dtype=np.float64)

    metrics = {
        "actor/version_lag_segment_mean": float(segment_lags_np.mean()),
        "actor/version_lag_segment_p95": float(np.percentile(segment_lags_np, 95)),
        "actor/version_lag_segment_max": float(segment_lags_np.max()),
        "actor/version_lag_token_mean": float(np.average(token_lags_np, weights=token_weights_np)),
        "actor/version_lag_token_p95": _weighted_percentile(token_lags_np, token_weights_np, 95),
        "actor/mixed_version_traj_ratio": float(np.mean(mixed_version_flags)) if mixed_version_flags else 0.0,
        "actor/unique_versions_per_traj_mean": (
            float(np.mean(unique_versions_per_traj)) if unique_versions_per_traj else 0.0
        ),
        "actor/version_span_per_traj_mean": float(np.mean(version_spans_per_traj)) if version_spans_per_traj else 0.0,
    }
    return metrics
