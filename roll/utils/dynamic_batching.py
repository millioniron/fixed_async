import bisect
from typing import Iterator

import torch

from roll.distributed.scheduler.protocol import DataProto
from roll.utils.logging import get_logger


logger = get_logger()


def _build_rank_local_micro_batch_meta(
    global_micro_batch_indices: list[list[int]],
    global_micro_batch_lengths: list[int],
    local_batch_sizes: list[int],
):
    rank_micro_batch_indices = []
    rank_micro_batch_lengths = []
    rank_num_micro_batches = []

    for local_batch_size in local_batch_sizes:
        local_indices = []
        local_lengths = []
        for (start, end), length in zip(global_micro_batch_indices, global_micro_batch_lengths):
            if start >= local_batch_size:
                continue
            local_end = min(end, local_batch_size)
            if local_end <= start:
                continue
            local_indices.append([start, local_end])
            local_lengths.append(length)
        rank_micro_batch_indices.append(local_indices)
        rank_micro_batch_lengths.append(local_lengths)
        rank_num_micro_batches.append(len(local_indices))

    return rank_micro_batch_indices, rank_micro_batch_lengths, rank_num_micro_batches


def dynamic_batching_shard(
    origin_batch: DataProto,
    dp_size: int,
    max_tokens_per_microbatch: int,
    sequence_length_round: int,
    pipeline_model_parallel_size: int = 1,
    virtual_pipeline_model_parallel_size: int = None,
    log_prefix: str = None,
) -> tuple[DataProto, dict]:
    #TODO use Karmarkar–Karp algorithm to replace the greedy implementation
    attention_mask = origin_batch.batch["attention_mask"]
    batch_size = attention_mask.shape[0]
    if batch_size == 0:
        raise ValueError("dynamic_batching_shard received an empty batch")
    dp_size = max(1, min(int(dp_size), batch_size))
    seq_lens = attention_mask.view(batch_size, -1).sum(-1).tolist()

    if 0 in seq_lens:
        logger.warning(f"The attention_mask is all zero in the {log_prefix} stage. Please verify the rollout stage.")

    seq_index_sort_by_len = [i for i, _ in sorted(enumerate(seq_lens), key=lambda x: x[1])]
    seq_lens_sort = [seq_lens[i] for i in seq_index_sort_by_len]
    local_indices = torch.arange(batch_size, dtype=torch.long)[seq_index_sort_by_len]

    batch = origin_batch.slice()
    batch.reorder(torch.tensor(seq_index_sort_by_len))

    seq_len_of_shard = [seq_lens_sort[i::dp_size] for i in range(dp_size)]
    aggregated_shards = [batch[i::dp_size] for i in range(dp_size)]
    aggregated_local_indices = [local_indices[i::dp_size] for i in range(dp_size)]
    aggregated_batch_sizes = [len(shard) for shard in aggregated_shards]

    global_micro_batch_indices = [[0, 0]]
    global_micro_batch_lengths = [0]
    max_seqlen_this_mb = sequence_length_round # at least `sequence_length_round`
    shard_size = max(len(shard) for shard in aggregated_shards)

    for shard_indice in range(shard_size):
        max_seqlen_this_shard_indice = 0
        for shard, seq_lens in zip(aggregated_shards, seq_len_of_shard):
            if shard_indice >= len(seq_lens):
                continue
            seq_len = seq_lens[shard_indice]
            max_seqlen_this_shard_indice = max(max_seqlen_this_shard_indice, seq_len)
        if max_seqlen_this_shard_indice == 0:
            continue
        max_seqlen_this_shard_indice = (
            (max_seqlen_this_shard_indice + sequence_length_round - 1) // sequence_length_round
        ) * sequence_length_round
        assert max_seqlen_this_shard_indice <= max_tokens_per_microbatch, (
            f"got an input of padded ({sequence_length_round}) sequence length of {max_seqlen_this_shard_indice}, "
            f"however max microbatch size is {max_tokens_per_microbatch} tokens"
        )
        curr_mbs_size = global_micro_batch_indices[-1][1] - global_micro_batch_indices[-1][0] + 1
        max_seqlen_this_mb = max(max_seqlen_this_mb, max_seqlen_this_shard_indice)
        total_tokens_in_mbs = curr_mbs_size * max_seqlen_this_mb
        if total_tokens_in_mbs <= max_tokens_per_microbatch:
            global_micro_batch_indices[-1][-1] += 1
            global_micro_batch_lengths[-1] = max_seqlen_this_mb
        else:
            global_micro_batch_indices.append([shard_indice, shard_indice + 1])
            max_seqlen_this_mb = max_seqlen_this_shard_indice
            global_micro_batch_lengths.append(max_seqlen_this_mb)

    total_tokens = sum(
        (end - start) * length
        for (start, end), length in zip(global_micro_batch_indices, global_micro_batch_lengths)
    )
    if pipeline_model_parallel_size > 1 and virtual_pipeline_model_parallel_size:
        # pad to multiple of `microbatch_group_size_per_vp_stage`
        num_micro_batches = len(global_micro_batch_indices)
        padded_num_micro_batches = (
            (num_micro_batches + pipeline_model_parallel_size - 1) // pipeline_model_parallel_size
        ) * pipeline_model_parallel_size
        assert pipeline_model_parallel_size <= shard_size, f"The pipeline_model_size: {pipeline_model_parallel_size} should not be greater than num_seqs in one dp_rank"
        assert padded_num_micro_batches <= shard_size
        num_micro_batches_needed = padded_num_micro_batches - num_micro_batches

        splittable_mbs = [i for i in range(num_micro_batches) if (global_micro_batch_indices[i][1] - global_micro_batch_indices[i][0]) > 1]
        # sort by tokens
        splittable_mbs.sort(key=lambda x: (global_micro_batch_indices[x][1] - global_micro_batch_indices[x][0]) * global_micro_batch_lengths[x], reverse=True)

        assert len(splittable_mbs) >= num_micro_batches_needed
        dropped_mbs = []
        added_micro_batch_indices = []
        added_micro_batch_lengths = []
        while num_micro_batches_needed:
            mb_to_split = splittable_mbs.pop(0)

            # compute split point
            split_start, split_end = global_micro_batch_indices[mb_to_split]
            split_length = global_micro_batch_lengths[mb_to_split]
            split_seqs = split_end - split_start
            split_point = split_start + (split_seqs // 2)

            # generate new mb
            new_mb1 = [split_start, split_point]
            new_mb2 = [split_point, split_end]

            # record dropped and added mbs
            dropped_mbs.append(mb_to_split)
            added_micro_batch_indices += [new_mb1, new_mb2]
            added_micro_batch_lengths += [split_length, split_length]

            num_micro_batches_needed -= 1

        global_micro_batch_indices = [global_micro_batch_indices[i] for i in range(num_micro_batches) if i not in dropped_mbs]
        global_micro_batch_lengths = [global_micro_batch_lengths[i] for i in range(num_micro_batches) if i not in dropped_mbs]

        # insert added_mbs, ensure sorted
        for added_mbs_indices, added_mbs_length in zip(added_micro_batch_indices, added_micro_batch_lengths):
            insert_indice = bisect.bisect_right(global_micro_batch_indices, added_mbs_indices)
            global_micro_batch_indices.insert(insert_indice, added_mbs_indices)
            global_micro_batch_lengths.insert(insert_indice, added_mbs_length)

    rank_micro_batch_indices, rank_micro_batch_lengths, rank_num_micro_batches = _build_rank_local_micro_batch_meta(
        global_micro_batch_indices=global_micro_batch_indices,
        global_micro_batch_lengths=global_micro_batch_lengths,
        local_batch_sizes=aggregated_batch_sizes,
    )

    actual_tokens = 0
    for local_indices, local_lengths in zip(rank_micro_batch_indices, rank_micro_batch_lengths):
        actual_tokens += sum((end - start) * length for (start, end), length in zip(local_indices, local_lengths))

    batch = DataProto.concat(aggregated_shards)
    batch.meta_info["dynamic_batching_origin_local_indices"] = torch.cat(aggregated_local_indices).tolist()
    batch.meta_info["dynamic_batching_rank_batch_sizes"] = aggregated_batch_sizes
    batch.meta_info["dynamic_batching_rank_origin_local_indices"] = [indices.tolist() for indices in aggregated_local_indices]
    batch.meta_info["dynamic_batching_rank_micro_batch_indices"] = rank_micro_batch_indices
    batch.meta_info["dynamic_batching_rank_micro_batch_lengths"] = rank_micro_batch_lengths
    batch.meta_info["dynamic_batching_rank_num_micro_batchs"] = rank_num_micro_batches
    batch.meta_info["global_micro_batch_indices"] = global_micro_batch_indices
    batch.meta_info["global_micro_batch_lengths"] = global_micro_batch_lengths
    batch.meta_info["micro_batch_indices"] = global_micro_batch_indices
    batch.meta_info["micro_batch_lengths"] = global_micro_batch_lengths
    batch.meta_info["num_micro_batchs"] = len(global_micro_batch_indices)

    valid_tokens = sum(seq_lens_sort)  # unmasked tokens
    actual_tokens_origin = batch_size * attention_mask.shape[-1]  # tokens with padding
    removed_padding_tokens = actual_tokens_origin - actual_tokens
    removed_padding_ratio = removed_padding_tokens / actual_tokens_origin
    prefix = f"dynamic_batching/{log_prefix}" if log_prefix else "dynamic_batching"
    metrics = {
        f"{prefix}/valid_tokens": valid_tokens,
        f"{prefix}/actual_tokens_origin": actual_tokens_origin,
        f"{prefix}/actual_tokens": actual_tokens,
        f"{prefix}/removed_padding_tokens": removed_padding_tokens,
        f"{prefix}/removed_padding_ratio": removed_padding_ratio,
    }
    return batch, metrics


def make_mini_batch_iter_for_dynamic_batching(
    data: DataProto,
    epochs: int,
    ga_steps: int = 1,
) -> Iterator[DataProto]:
    """
        Iterator that groups previously created global micro batches into mini batches
        based on gradient accumulation steps (ga_steps).

        Terminology:
        - Micro batch: The smallest training unit that can fit into GPU memory
          for one forward/backward pass.
          These are already determined in `dynamic_batching_shard` based on
          `max_tokens_per_microbatch`.
        - Mini batch: A group of several micro batches.
          During training, you iterate over each micro batch inside a mini batch,
          perform forward/backward passes, accumulate gradients, and after `ga_steps`
          micro batches, perform a parameter update (`optimizer.step()`).

        This function:
        1. Reads the global micro batch indices/lengths from `data.meta_info`.
        2. Groups `ga_steps` consecutive micro batches into a single mini batch.
        3. Adjusts indices so micro batches are relative to the mini batch.
        4. Yields each mini batch for training.
        """
    global_micro_batch_indices = data.meta_info["global_micro_batch_indices"]
    global_micro_batch_lengths = data.meta_info["global_micro_batch_lengths"]
    for _ in range(epochs):
        for i in range(0, len(global_micro_batch_indices), ga_steps):
            indices_chunk = global_micro_batch_indices[i : i + ga_steps]
            start = indices_chunk[0][0]
            end = indices_chunk[-1][-1]
            mini_batch = data.slice(start, end)

            data.meta_info["micro_batch_indices"] = [[x - start for x in row] for row in indices_chunk]
            data.meta_info["micro_batch_lengths"] = global_micro_batch_lengths[i : i + ga_steps]
            mini_batch.meta_info["mini_batch_size"] = mini_batch.batch.batch_size[0]
            mini_batch.meta_info["num_micro_batchs"] = len(indices_chunk)

            yield (mini_batch)


def make_micro_batch_iter_for_dynamic_batching(mini_batch: DataProto):
    micro_batch_indices = mini_batch.meta_info["micro_batch_indices"]
    micro_batch_lengths = mini_batch.meta_info["micro_batch_lengths"]
    for seqlen, (start_idx, end_idx) in zip(micro_batch_lengths, micro_batch_indices):
        micro_batch = mini_batch.slice(start_idx, end_idx)
        input_ids_shape = micro_batch.batch["input_ids"].shape
        for k in mini_batch.batch.keys():
            if (len(micro_batch.batch[k].shape) == len(input_ids_shape) or k == "position_ids") and micro_batch.batch[k].shape[-1] in (
                input_ids_shape[-1],
                input_ids_shape[-1] - 1,
            ):
                micro_batch.batch[k] = torch.narrow(
                    micro_batch.batch[k],
                    dim=-1,
                    start=0,
                    length=seqlen if micro_batch.batch[k].shape[-1] == input_ids_shape[-1] else seqlen - 1,
                )
        yield micro_batch
