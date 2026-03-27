from __future__ import annotations

from functools import lru_cache

import torch
import torch.distributed as dist

from steptronoss.core.parallel_state import PM

INT32_MAX = torch.iinfo(torch.int32).max


def _all_gather_splits(send_sizes: torch.Tensor, group, world_size: int) -> torch.Tensor:
    send_sizes = send_sizes.to(dtype=torch.int64)
    if world_size == 1:
        return send_sizes.unsqueeze(0)

    gathered = send_sizes.new_empty(world_size * send_sizes.numel())
    dist.all_gather_into_tensor(gathered, send_sizes.contiguous(), group=group)
    return gathered.view(world_size, send_sizes.numel())


class _AllToAllSingle(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor, output_splits: list[int], input_splits: list[int], group):
        ctx.output_splits = [int(x) for x in output_splits]
        ctx.input_splits = [int(x) for x in input_splits]
        ctx.group = group

        total_output = sum(ctx.output_splits)
        output = input_tensor.new_empty((total_output,) + input_tensor.shape[1:])
        dist.all_to_all_single(
            output,
            input_tensor.contiguous(),
            output_split_sizes=ctx.output_splits,
            input_split_sizes=ctx.input_splits,
            group=group,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        total_input = sum(ctx.input_splits)
        grad_input = grad_output.new_empty((total_input,) + grad_output.shape[1:])
        dist.all_to_all_single(
            grad_input,
            grad_output.contiguous(),
            output_split_sizes=ctx.input_splits,
            input_split_sizes=ctx.output_splits,
            group=ctx.group,
        )
        return grad_input, None, None, None


def _all_to_all_single_tensor(input_tensor: torch.Tensor, output_splits: list[int], input_splits: list[int], group):
    return _AllToAllSingle.apply(input_tensor, output_splits, input_splits, group)


def _all_to_all_single_no_grad(
    input_tensor: torch.Tensor,
    output_splits: list[int],
    input_splits: list[int],
    group,
) -> torch.Tensor:
    total_output = sum(int(x) for x in output_splits)
    output = input_tensor.new_empty((total_output,) + input_tensor.shape[1:])
    dist.all_to_all_single(
        output,
        input_tensor.contiguous(),
        output_split_sizes=[int(x) for x in output_splits],
        input_split_sizes=[int(x) for x in input_splits],
        group=group,
    )
    return output


def _pick_token_id_dtype(num_tokens: int) -> torch.dtype:
    if num_tokens <= INT32_MAX:
        return torch.int32
    return torch.int64


def _stable_bucket_order_by_rank(dst_rank: torch.Tensor, world_size: int) -> torch.Tensor:
    if dst_rank.numel() == 0:
        return dst_rank.new_empty((0,), dtype=torch.int64)

    order_parts = []
    for rank in range(world_size):
        rank_order = torch.nonzero(dst_rank == rank, as_tuple=True)[0]
        if rank_order.numel() > 0:
            order_parts.append(rank_order)

    if not order_parts:
        return dst_rank.new_empty((0,), dtype=torch.int64)
    if len(order_parts) == 1:
        return order_parts[0]
    return torch.cat(order_parts, dim=0)


def _build_rank_permute_map(
    token_expert_ranks: torch.Tensor,
    world_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if token_expert_ranks.numel() == 0:
        empty = token_expert_ranks.new_empty((0,), dtype=torch.int64)
        send_sizes = token_expert_ranks.new_zeros((world_size,), dtype=torch.int64)
        rank_permute_map = token_expert_ranks.new_full(token_expert_ranks.shape, world_size, dtype=torch.int32)
        valid_mask = token_expert_ranks.new_zeros(token_expert_ranks.shape, dtype=torch.bool)
        return rank_permute_map, valid_mask, empty, empty, send_sizes

    sorted_ranks, _ = token_expert_ranks.to(dtype=torch.int64).sort(dim=1)
    unique_mask = torch.ones_like(sorted_ranks, dtype=torch.bool)
    if sorted_ranks.size(1) > 1:
        unique_mask[:, 1:] = sorted_ranks[:, 1:] != sorted_ranks[:, :-1]
    valid_mask = unique_mask & (sorted_ranks >= 0) & (sorted_ranks < world_size)
    invalid_rank = torch.full_like(sorted_ranks, world_size)
    rank_permute_map = torch.where(valid_mask, sorted_ranks, invalid_rank).to(torch.int32)

    token_idx, _ = torch.nonzero(valid_mask, as_tuple=True)
    if token_idx.numel() == 0:
        empty = sorted_ranks.new_empty((0,), dtype=torch.int64)
        send_sizes = sorted_ranks.new_zeros((world_size,), dtype=torch.int64)
        return rank_permute_map, valid_mask, empty, empty, send_sizes

    dst_rank = sorted_ranks[valid_mask]
    send_sizes = torch.bincount(dst_rank, minlength=world_size)
    order = _stable_bucket_order_by_rank(dst_rank, world_size)
    return rank_permute_map, valid_mask, token_idx.index_select(0, order), dst_rank.index_select(0, order), send_sizes


def _build_unique_token_rank_pairs(
    token_expert_ranks: torch.Tensor,
    world_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _, _, token_idx, dst_rank, send_sizes = _build_rank_permute_map(token_expert_ranks, world_size)
    return token_idx, dst_rank, send_sizes


@lru_cache(maxsize=1)
def _load_npu_moe_token_ops():
    try:
        from mindspeed.ops.npu_moe_token_permute import npu_moe_token_permute
        from mindspeed.ops.npu_moe_token_unpermute import npu_moe_token_unpermute
    except Exception:
        return None, None
    return npu_moe_token_permute, npu_moe_token_unpermute


def _permute_hidden_states(
    hidden_states: torch.Tensor,
    token_idx: torch.Tensor,
    rank_permute_map: torch.Tensor,
    num_out_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor | None, bool]:
    npu_moe_token_permute, _ = _load_npu_moe_token_ops()
    if (
        npu_moe_token_permute is None
        or hidden_states.device.type != "npu"
        or hidden_states.dtype != torch.bfloat16
        or num_out_tokens <= 0
    ):
        return hidden_states.index_select(0, token_idx), None, False

    permuted, sorted_indices = npu_moe_token_permute(
        hidden_states,
        rank_permute_map,
        num_out_tokens=int(num_out_tokens),
        padded_mode=False,
    )
    return permuted, sorted_indices, True


def _build_rank_combine_probs(valid_mask: torch.Tensor) -> torch.Tensor:
    return valid_mask.to(dtype=torch.bfloat16)


def _unpermute_hidden_states(
    hidden_states: torch.Tensor,
    sorted_indices: torch.Tensor,
    combine_probs: torch.Tensor,
    restore_shape: tuple[int, int],
) -> torch.Tensor:
    _, npu_moe_token_unpermute = _load_npu_moe_token_ops()
    return npu_moe_token_unpermute(
        hidden_states,
        sorted_indices.to(dtype=torch.int32),
        probs=combine_probs,
        padded_mode=False,
        restore_shape=restore_shape,
    )


class NPUAllToAllDispatcher:
    """Tensorized all-to-all dispatcher for EP token routing on NPU."""

    def __init__(self, parallel: str, num_experts: int):
        self.rank = PM.rank_in(parallel)
        self.world_size = PM.size_of(parallel)
        self.group = PM.group_of(parallel)

        self.num_local_experts = num_experts // self.world_size
        self.comm_record: (
            tuple[list[int], list[int], int, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None] | None
        ) = None
        self.last_dispatch_used_fused_permute = False
        self.last_combine_used_fused_unpermute = False

    def _build_send_layout(
        self,
        token_expert_ids: torch.Tensor,
        token_expert_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int], list[int], torch.Tensor, torch.Tensor]:
        token_expert_ranks = token_expert_ids // self.num_local_experts
        rank_permute_map, valid_mask, token_idx, dst_rank, send_sizes_tensor = _build_rank_permute_map(
            token_expert_ranks,
            self.world_size,
        )
        gathered_sizes = _all_gather_splits(send_sizes_tensor, self.group, self.world_size)
        recv_sizes_tensor = gathered_sizes[:, self.rank].contiguous()

        send_indices = token_expert_ids.index_select(0, token_idx)
        send_probs = token_expert_weights.index_select(0, token_idx)
        send_ranks = token_expert_ranks.index_select(0, token_idx)

        rank_offsets = (dst_rank.to(dtype=send_indices.dtype) * self.num_local_experts).unsqueeze(1)
        local_mask = send_ranks == dst_rank.unsqueeze(1)
        send_indices = send_indices - rank_offsets
        send_indices = torch.where(local_mask, send_indices, torch.full_like(send_indices, -1))
        send_probs = torch.where(local_mask, send_probs, torch.zeros_like(send_probs))

        send_sizes = send_sizes_tensor.cpu().tolist()
        recv_sizes = recv_sizes_tensor.cpu().tolist()
        return token_idx.to(torch.int64), send_indices, send_probs, send_sizes, recv_sizes, rank_permute_map, valid_mask

    def dispatch(
        self,
        hidden_states: torch.FloatTensor,
        token_expert_ids: torch.IntTensor,
        token_expert_weights: torch.FloatTensor,
    ):
        total_tokens = hidden_states.size(0)
        self.last_dispatch_used_fused_permute = False
        self.last_combine_used_fused_unpermute = False

        if self.world_size == 1:
            local_ids = token_expert_ids.clone()
            local_ids -= self.rank * self.num_local_experts
            self.comm_record = ([], [], total_tokens, None, None, None)
            return hidden_states, local_ids, token_expert_weights

        token_idx, send_indices, send_probs, send_sizes, recv_sizes, rank_permute_map, valid_mask = (
            self._build_send_layout(
                token_expert_ids,
                token_expert_weights,
            )
        )
        send_hidden, sorted_indices, self.last_dispatch_used_fused_permute = _permute_hidden_states(
            hidden_states,
            token_idx,
            rank_permute_map,
            token_idx.numel(),
        )
        send_token_ids = None
        if not self.last_dispatch_used_fused_permute:
            send_token_ids = token_idx.to(dtype=_pick_token_id_dtype(total_tokens))

        recv_hidden = _all_to_all_single_tensor(send_hidden, recv_sizes, send_sizes, self.group)
        recv_indices = _all_to_all_single_no_grad(send_indices, recv_sizes, send_sizes, self.group)
        recv_probs = _all_to_all_single_tensor(send_probs, recv_sizes, send_sizes, self.group)
        recv_token_ids = None
        combine_probs = None
        if self.last_dispatch_used_fused_permute:
            combine_probs = _build_rank_combine_probs(valid_mask)
        else:
            recv_token_ids = _all_to_all_single_no_grad(send_token_ids, recv_sizes, send_sizes, self.group)

        del send_hidden, send_indices, send_probs, send_token_ids, rank_permute_map
        self.comm_record = (send_sizes, recv_sizes, total_tokens, recv_token_ids, sorted_indices, combine_probs)
        return recv_hidden, recv_indices, recv_probs

    def combine(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.comm_record is None:
            raise RuntimeError("combine called before dispatch")

        send_sizes, recv_sizes, restore_tokens, token_ids, sorted_indices, combine_probs = self.comm_record
        if self.world_size == 1:
            self.comm_record = None
            return hidden_states

        recv_hidden = _all_to_all_single_tensor(hidden_states, send_sizes, recv_sizes, self.group)

        if sorted_indices is not None and combine_probs is not None:
            output = _unpermute_hidden_states(
                recv_hidden,
                sorted_indices,
                combine_probs,
                (restore_tokens, hidden_states.size(1)),
            )
            self.last_combine_used_fused_unpermute = True
        else:
            recv_token_ids = _all_to_all_single_no_grad(token_ids, send_sizes, recv_sizes, self.group)
            output = hidden_states.new_zeros((restore_tokens, hidden_states.size(1)))
            if recv_hidden.numel() > 0:
                output.index_add_(0, recv_token_ids.to(torch.int64), recv_hidden)
        self.comm_record = None
        return output
