from __future__ import annotations

from steptronoss.model.optimizations.moe_routing.triton import triton_index_scatter
from steptronoss.timers import timeit


def triton_routed_grouped_ffn_fused(
    w1,
    w2,
    act,
    x,
    token_expert_ids,
    token_weights,
):
    from steptronoss.model.utils.moe_utils import grouped_gemm, histogram, moe_weighted_gather

    experts_histogram = histogram(token_expert_ids, w1.shape[0])
    if experts_histogram.numel() == 0 or int(experts_histogram.sum().item()) == 0:
        return x * token_weights.sum()

    batch_sizes = experts_histogram.long()
    with timeit("moe-routed-ffn", level=2):
        x, scatter_index = triton_index_scatter(x, token_expert_ids, experts_histogram)
        x = grouped_gemm(x, w1, batch_sizes=batch_sizes, trans_b=True)
        x = act(x)
        x = grouped_gemm(x, w2, batch_sizes=batch_sizes, trans_b=True)
        x = moe_weighted_gather(x, scatter_index, token_weights)
    return x
