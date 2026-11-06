from __future__ import annotations
from typing import Dict, Tuple, Optional

import torch

from roll.utils.functionals import masked_mean, masked_sum, agg_loss
from roll.pipeline.agentic.utils import compute_segment_masked_mean
from roll.configs.base_config import TrainInferCorrectionConfig
from roll.utils.logging import get_logger

logger = get_logger()


def _masked_pearson_correlation(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> float:
    """Compute Pearson correlation on valid entries only."""
    valid = mask.bool()
    if valid.dim() < x.dim():
        valid = valid.unsqueeze(-1)

    x_valid = x[valid].float()
    y_valid = y[valid].float()

    if x_valid.numel() < 2:
        return 0.0

    x_centered = x_valid - x_valid.mean()
    y_centered = y_valid - y_valid.mean()
    denom = torch.sqrt((x_centered.square().sum()) * (y_centered.square().sum()))

    if denom.item() == 0.0:
        return 1.0 if torch.allclose(x_valid, y_valid) else 0.0

    corr = torch.clamp((x_centered * y_centered).sum() / denom, min=-1.0, max=1.0)
    return corr.detach().item()


def _compute_all_granularity(old_log_probs, infer_log_probs, response_mask) -> dict:
    """Compute importance ratios and probability differences at multiple granularities."""
    response_mask = response_mask.long()
    log_ratio = old_log_probs - infer_log_probs

    ratio_token = log_ratio.exp()
    diff_token = old_log_probs.exp() - infer_log_probs.exp()

    # Geometric mean (per sequence, then broadcast to token level)
    log_ratio_geo = masked_mean(log_ratio, response_mask, dim=-1)  # [B]
    ratio_geometric = log_ratio_geo.exp().unsqueeze(-1).expand_as(ratio_token)
    diff_geometric = masked_mean(diff_token, response_mask, dim=-1).unsqueeze(-1).expand_as(diff_token)

    # Sequence-level sum (then broadcast to token level)
    log_ratio_seq = masked_sum(log_ratio, response_mask, dim=-1)   # [B]
    ratio_sequence = log_ratio_seq.exp().unsqueeze(-1).expand_as(ratio_token)
    diff_sequence = masked_sum(diff_token, response_mask, dim=-1).unsqueeze(-1).expand_as(diff_token)

    # Segment-level mean (computed per segment within each sequence)
    log_ratio_segment = compute_segment_masked_mean(log_ratio, response_mask)  # [B, T]
    ratio_segment = log_ratio_segment.exp()
    diff_segment = compute_segment_masked_mean(diff_token, response_mask)

    return {
        "ratio": {
            "token": ratio_token,
            "geometric": ratio_geometric,
            "sequence": ratio_sequence,
            "segment": ratio_segment,
        },
        "diff": {
            "token": diff_token,
            "geometric": diff_geometric,
            "sequence": diff_sequence,
            "segment": diff_segment,
        },
    }


def _compute_logprob_views(old_log_probs, infer_log_probs, response_mask) -> dict:
    """Build log-prob views at each granularity for diagnostics."""
    response_mask = response_mask.long()
    seq_valid = response_mask.sum(dim=-1) > 0

    return {
        "token": {
            "old": old_log_probs,
            "infer": infer_log_probs,
            "mask": response_mask,
        },
        "segment": {
            "old": compute_segment_masked_mean(old_log_probs, response_mask),
            "infer": compute_segment_masked_mean(infer_log_probs, response_mask),
            "mask": response_mask,
        },
        "geometric": {
            "old": masked_mean(old_log_probs, response_mask, dim=-1),
            "infer": masked_mean(infer_log_probs, response_mask, dim=-1),
            "mask": seq_valid,
        },
        "sequence": {
            "old": masked_sum(old_log_probs, response_mask, dim=-1),
            "infer": masked_sum(infer_log_probs, response_mask, dim=-1),
            "mask": seq_valid,
        },
    }


def _infer_global_valid_samples_from_mask(mask: torch.Tensor) -> float:
    """Count the number of samples that contain at least one valid token."""
    valid_samples = (mask.sum(dim=-1) > 0).float().sum().detach().item()
    return max(float(valid_samples), 1.0)


def _infer_global_valid_tokens_from_mask(mask: torch.Tensor) -> float:
    """Count the total number of valid tokens across all samples."""
    valid_tokens = mask.float().sum().detach().item()
    return max(float(valid_tokens), 1.0)


def _record_train_infer_divergence_metrics(
    metrics: Dict[str, float],
    prefix: str,
    old_log_probs: torch.Tensor,
    infer_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    global_valid_samples: Optional[int] = None,
) -> None:
    """Record train-infer divergence metrics under a metric prefix."""
    if global_valid_samples is None:
        global_valid_samples = _infer_global_valid_samples_from_mask(response_mask)

    stats = _compute_all_granularity(
        old_log_probs=old_log_probs,
        infer_log_probs=infer_log_probs,
        response_mask=response_mask,
    )
    diff = stats["diff"]
    ratio = stats["ratio"]
    logprob_views = _compute_logprob_views(
        old_log_probs=old_log_probs,
        infer_log_probs=infer_log_probs,
        response_mask=response_mask,
    )

    for agg in ("token", "segment", "geometric", "sequence"):
        metrics[f"{prefix}_logprob_corr_{agg}"] = _masked_pearson_correlation(
            logprob_views[agg]["old"],
            logprob_views[agg]["infer"],
            logprob_views[agg]["mask"],
        )
        metrics[f"{prefix}_abs_diff_{agg}_mean@sum"] = agg_loss(
            loss_mat=diff[agg].abs(),
            loss_mask=response_mask,
            loss_agg_mode="seq-mean-token-mean",
            global_valid_samples=global_valid_samples,
        ).detach().item()
        metrics[f"{prefix}_ratio_{agg}_mean@sum"] = agg_loss(
            loss_mat=ratio[agg],
            loss_mask=response_mask,
            loss_agg_mode="seq-mean-token-mean",
            global_valid_samples=global_valid_samples,
        ).detach().item()
        metrics[f"{prefix}_diff_{agg}_mean@sum"] = agg_loss(
            loss_mat=diff[agg],
            loss_mask=response_mask,
            loss_agg_mode="seq-mean-token-mean",
            global_valid_samples=global_valid_samples,
        ).detach().item()


def compute_train_infer_correction(
    cfg: TrainInferCorrectionConfig,
    response_mask: torch.Tensor,          # [B, T]
    old_log_probs: torch.Tensor,          # [B, T]
    infer_log_probs: torch.Tensor,        # [B, T]
    global_valid_samples: Optional[int] = None,  # Number of valid sequences
    global_valid_tokens: Optional[int] = None,   # Total number of valid tokens
    apply_filters: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Compute importance sampling weights and apply optional filters based on train-infer divergence."""
    metrics: Dict[str, float] = {}

    base_mask = response_mask.long()
    if global_valid_samples is None:
        global_valid_samples = _infer_global_valid_samples_from_mask(base_mask)
    if global_valid_tokens is None:
        global_valid_tokens = _infer_global_valid_tokens_from_mask(base_mask)

    _record_train_infer_divergence_metrics(
        metrics=metrics,
        prefix="actor/train_infer",
        old_log_probs=old_log_probs,
        infer_log_probs=infer_log_probs,
        response_mask=base_mask,
        global_valid_samples=global_valid_samples,
    )

    stats = _compute_all_granularity(
        old_log_probs=old_log_probs,
        infer_log_probs=infer_log_probs,
        response_mask=base_mask,
    )
    ratio = stats["ratio"]
    diff = stats["diff"]

    # 1) Importance Sampling (IS) Weight Handling
    if cfg.is_weight.enabled:
        is_weight = ratio[cfg.is_weight.weight_type]
        ub = cfg.is_weight.upper_bound
        if ub is not None:
            # Log the fraction of weights clipped due to upper bound
            metrics["actor/is_weight_clipfrac@sum"] = agg_loss(
                loss_mat=(is_weight > ub).float(),
                loss_mask=base_mask,
                loss_agg_mode='token-mean',
                batch_num_tokens=global_valid_tokens,
                global_valid_samples=global_valid_samples
            ).detach().item()
            is_weight = is_weight.clamp(min=0.0, max=ub)
        if cfg.is_weight.detach:
            is_weight = is_weight.detach()
    else:
        is_weight = torch.ones_like(ratio["token"]).detach()

    # 2) Apply Filters (if enabled)
    filter_mask = torch.ones_like(base_mask)
    recorded_val_metrics = set()  # Avoid duplicate metric logging for the same granularity

    if apply_filters:
        for i, f in enumerate(cfg.filters):
            if not f.enabled:
                continue

            agg = f.agg_type

            # --- Ratio-based Filter ---
            if f.ratio_enabled:
                m_ratio = (ratio[agg] >= f.ratio_low).float() * (ratio[agg] <= f.ratio_high).float()

                # Log pass rate of this filter over currently active tokens
                metrics[f"actor/train_infer_{agg}_ratio_mask_mean@sum"] = agg_loss(
                    loss_mat=m_ratio,
                    loss_mask=base_mask,
                    loss_agg_mode='token-mean',
                    batch_num_tokens=global_valid_tokens,
                ).detach().item()

                # Log mean value of the ratio at this granularity (for monitoring)
                val_key = f"actor/train_infer_ratio_{agg}_mean@sum"
                if val_key not in recorded_val_metrics:
                    metrics[val_key] = agg_loss(
                        loss_mat=ratio[agg],
                        loss_mask=base_mask,
                        loss_agg_mode="seq-mean-token-mean",
                        global_valid_samples=global_valid_samples,
                    ).detach().item()
                    recorded_val_metrics.add(val_key)

                filter_mask = filter_mask * m_ratio

            # --- Difference-based Filter ---
            if f.diff_enabled:
                m_diff = (diff[agg] >= f.diff_low).float() * (diff[agg] <= f.diff_high).float()

                # Log pass rate of this filter
                metrics[f"actor/train_infer_{agg}_diff_mask_mean"] = agg_loss(
                    loss_mat=m_diff,
                    loss_mask=base_mask,
                    loss_agg_mode='token-mean',
                    batch_num_tokens=global_valid_tokens,
                ).detach().item()

                # Log mean value of the difference at this granularity
                val_key = f"actor/train_infer_diff_{agg}_mean@sum"
                if val_key not in recorded_val_metrics:
                    metrics[val_key] = agg_loss(
                        loss_mat=diff[agg],
                        loss_mask=base_mask,
                        loss_agg_mode="seq-mean-token-mean",
                        global_valid_samples=global_valid_samples,
                    ).detach().item()
                    recorded_val_metrics.add(val_key)

                filter_mask = filter_mask * m_diff

    # 3) Final overall pass rate after all filters
    if apply_filters and cfg.filters:
        metrics["actor/train_infer_final_mask_mean"] = masked_mean(
            base_mask*filter_mask.float(), base_mask
        ).detach().item()
        final_sample_mask = (filter_mask.sum(dim=-1) > 0).float()
        base_sample_mask = (base_mask.sum(dim=-1) > 0).float()
        metrics["actor/train_infer_final_sample_mean"] = masked_mean(
            final_sample_mask, base_sample_mask
        ).detach().item()

        postfilter_mask = base_mask * filter_mask.long()
        _record_train_infer_divergence_metrics(
            metrics=metrics,
            prefix="actor/train_infer_postfilter",
            old_log_probs=old_log_probs,
            infer_log_probs=infer_log_probs,
            response_mask=postfilter_mask,
            # Reuse the original global sample denominator so post-filter metrics stay
            # numerically comparable to the pre-filter metrics under microbatching.
            global_valid_samples=global_valid_samples,
        )
        for agg in ("token", "segment", "geometric", "sequence"):
            metrics[f"actor/train_infer_postfilter_abs_diff_{agg}_delta@sum"] = (
                metrics[f"actor/train_infer_abs_diff_{agg}_mean@sum"]
                - metrics[f"actor/train_infer_postfilter_abs_diff_{agg}_mean@sum"]
            )

    return is_weight, filter_mask, metrics


def apply_train_infer_correction_to_batch(
    pipeline_config,
    batch,
    stat_mask_key='response_mask',
    update_mask_keys: Optional[list] = None,
):
    """Apply train-infer correction to a batch at the pipeline level.

    This function is designed for pipeline-level usage where masks are in their
    original shape [B, T]. It handles slicing internally and updates the original
    masks with the computed filter mask.

    Args:
        pipeline_config: Pipeline configuration containing train_infer_correction config
        batch: DataProto batch to modify
        stat_mask_key: Key of mask used for computing train-infer statistics (diff, ratio)
        update_mask_keys: List of mask keys to update with computed filter mask.
                          If None, defaults to ['response_mask'].

    Note:
        For worker-level usage, use compute_train_infer_correction() directly,
        as it works with already-sliced tensors [B, T-1] and provides more flexibility.
    """
    # Check if required fields are present
    if "old_log_probs" not in batch.batch or "infer_logprobs" not in batch.batch:
        missing_fields = []
        if "old_log_probs" not in batch.batch:
            missing_fields.append("'old_log_probs'")
        if "infer_logprobs" not in batch.batch:
            missing_fields.append("'infer_logprobs'")
        logger.warning(f"Skipping train-infer correction: {', '.join(missing_fields)} not found in batch.")
        stat_mask = batch.batch[stat_mask_key][:, 1:].long()  # [B, T-1]
        batch.batch["train_infer_is_weight"] = torch.ones_like(stat_mask, dtype=torch.float32)
        return batch, {}

    # Default: update response_mask if not specified
    if update_mask_keys is None:
        update_mask_keys = [stat_mask_key]

    # Get the mask for computing train-infer statistics (always sliced to [B, T-1])
    stat_mask = batch.batch[stat_mask_key][:, 1:].long()  # [B, T-1]
    old_lp = batch.batch["old_log_probs"]                # [B, T-1]
    infer_lp = batch.batch["infer_logprobs"]             # [B, T-1]

    cfg = pipeline_config.train_infer_correction

    # Compute IS weights and filter mask
    is_w, filter_mask, corr_metrics = compute_train_infer_correction(
        cfg=cfg,
        response_mask=stat_mask,
        old_log_probs=old_lp,
        infer_log_probs=infer_lp,
        global_valid_samples=None,   # Will be inferred from stat_mask
        global_valid_tokens=None,    # Will be inferred from stat_mask
        apply_filters=True,
    )

    # Set train_infer_is_weight
    batch.batch["train_infer_is_weight"] = is_w

    # Apply filter mask to all specified masks
    for key in update_mask_keys:
        if key in batch.batch:
            mask_tensor = batch.batch[key]
            # Check if mask is already sliced (shape [B, T-1]) or full (shape [B, T])
            # final_response_mask is already [:, 1:] sliced in get_sample_level_mask
            if mask_tensor.shape[-1] == filter_mask.shape[-1]:
                # Mask is already sliced (e.g., final_response_mask)
                batch.batch[key] = mask_tensor.long() * filter_mask.long()
            else:
                # Mask is full shape (e.g., response_mask), apply to [:, 1:] part
                batch.batch[key][:, 1:] = mask_tensor[:, 1:].long() * filter_mask.long()
        else:
            logger.warning(f"Mask key '{key}' not found in batch, skipping update.")

    return batch, corr_metrics
