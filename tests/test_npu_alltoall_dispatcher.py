import torch

from steptronoss.model.ep_dispatcher.npu_alltoall_dispatcher import (
    _build_rank_permute_map,
    _build_unique_token_rank_pairs,
    _pick_token_id_dtype,
    _stable_bucket_order_by_rank,
)


def _baseline_unique_token_rank_pairs(token_expert_ranks: torch.Tensor, world_size: int):
    token_idx_parts = []
    dst_rank_parts = []
    send_sizes = []

    for rank in range(world_size):
        this_rank = (token_expert_ranks == rank).any(dim=1)
        token_idx = torch.nonzero(this_rank, as_tuple=True)[0]
        token_idx_parts.append(token_idx)
        dst_rank_parts.append(torch.full_like(token_idx, rank, dtype=torch.int64))
        send_sizes.append(int(token_idx.numel()))

    if token_idx_parts:
        token_idx = torch.cat(token_idx_parts, dim=0)
        dst_rank = torch.cat(dst_rank_parts, dim=0)
    else:
        token_idx = torch.empty((0,), dtype=torch.int64)
        dst_rank = torch.empty((0,), dtype=torch.int64)

    return token_idx, dst_rank, torch.tensor(send_sizes, dtype=torch.int64)


def test_build_unique_token_rank_pairs_matches_baseline():
    generator = torch.Generator().manual_seed(1234)

    for world_size in (1, 2, 4, 8):
        for token_num in (0, 1, 7, 17):
            for topk in (1, 2, 4):
                token_expert_ranks = torch.randint(
                    -1,
                    world_size,
                    (token_num, topk),
                    generator=generator,
                    dtype=torch.int64,
                )
                got = _build_unique_token_rank_pairs(token_expert_ranks, world_size)
                expected = _baseline_unique_token_rank_pairs(token_expert_ranks, world_size)

                assert torch.equal(got[0], expected[0])
                assert torch.equal(got[1], expected[1])
                assert torch.equal(got[2], expected[2])


def test_build_unique_token_rank_pairs_deduplicates_same_rank():
    token_expert_ranks = torch.tensor(
        [
            [0, 0, 1, 1],
            [2, 2, 2, -1],
            [3, 1, 3, 1],
        ],
        dtype=torch.int64,
    )

    token_idx, dst_rank, send_sizes = _build_unique_token_rank_pairs(token_expert_ranks, world_size=4)

    assert torch.equal(token_idx, torch.tensor([0, 0, 2, 1, 2], dtype=torch.int64))
    assert torch.equal(dst_rank, torch.tensor([0, 1, 1, 2, 3], dtype=torch.int64))
    assert torch.equal(send_sizes, torch.tensor([1, 2, 1, 1], dtype=torch.int64))


def test_build_rank_permute_map_matches_fused_permute_sort_contract():
    token_expert_ranks = torch.tensor(
        [
            [0, 2, 2, 0],
            [1, 1, 3, 3],
            [0, 1, 2, 3],
        ],
        dtype=torch.int64,
    )
    world_size = 4

    rank_permute_map, valid_mask, token_idx, dst_rank, send_sizes = _build_rank_permute_map(
        token_expert_ranks, world_size
    )

    flat_rank_map = rank_permute_map.reshape(-1).to(torch.int64)
    flat_sorted_pos = flat_rank_map.argsort(stable=True)
    valid_count = int(valid_mask.sum().item())
    selected_pos = flat_sorted_pos[:valid_count]

    expected_token_idx = torch.div(selected_pos, token_expert_ranks.size(1), rounding_mode="floor")
    expected_dst_rank = flat_rank_map.index_select(0, selected_pos)

    assert torch.equal(token_idx, expected_token_idx)
    assert torch.equal(dst_rank, expected_dst_rank)
    assert torch.equal(send_sizes, torch.bincount(expected_dst_rank, minlength=world_size))


def test_stable_bucket_order_matches_stable_argsort():
    generator = torch.Generator().manual_seed(4321)

    for world_size in (1, 2, 4, 8):
        for token_num in (0, 1, 7, 31, 127):
            dst_rank = torch.randint(0, world_size, (token_num,), generator=generator, dtype=torch.int64)
            expected = dst_rank.argsort(stable=True)
            got = _stable_bucket_order_by_rank(dst_rank, world_size)
            assert torch.equal(got, expected)


def test_pick_token_id_dtype_prefers_int32_when_safe():
    assert _pick_token_id_dtype(0) is torch.int32
    assert _pick_token_id_dtype(1024) is torch.int32
    assert _pick_token_id_dtype(torch.iinfo(torch.int32).max + 1) is torch.int64
