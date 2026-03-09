from steptronoss.exp.optimizer import MuonConfig


class Step3p5MuonConfig(MuonConfig):
    def __init__(self):
        super().__init__()
        self.weight_decay_on_1d_params = True

        self.muon_ns_steps = 6
        self.muon_newtonschulz_fn = "polar_express"

    def mark_muon_params(self, model) -> None:
        import torch
        from torch.nn import Parameter

        muon_exclude_names = self.muon_exclude_names
        if muon_exclude_names is None:
            muon_exclude_names = ()

        embedding_params: set[Parameter] = set()
        if self.muon_exclude_embeddings:
            for module in model.modules():
                if isinstance(module, torch.nn.Embedding) or "Embedding" in module.__class__.__name__:
                    weight = getattr(module, "weight", None)
                    if isinstance(weight, Parameter):
                        embedding_params.add(weight)

        for name, param in model.named_parameters():
            if getattr(param, "is_muon_param", False):
                is_muon_param = True
            elif (
                self.muon_param_attr_only
                or (self.muon_exclude_embeddings and param in embedding_params)
                or (muon_exclude_names and any(tag in name for tag in muon_exclude_names))
            ):
                is_muon_param = False
            else:
                is_muon_param = param.ndim >= 2

            param.is_muon_param = is_muon_param

    def mark_gather_ops(self, model) -> None:
        from steptronoss.checkpointing.reshape_ops import (
            ColumnParallel,
            FFNMergeGateUp,
            GQAMergeQKV,
            GQAMergeQKVBias,
            GQAMergeQKVG,
            Inverse,
            KeepThisTP,
            Rename,
            RowParallel,
            Sequential,
            UnbindMoE,
        )
        from steptronoss.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
        from steptronoss.model.common.feed_forward import FeedForward
        from steptronoss.model.common.grouped_query_attention import GroupedQueryAttention
        from steptronoss.model.common.moe_block import GroupedExperts, MoEGate

        identity = Sequential([])
        merge_ops: dict[int, object] = {}

        # Specialized packed weights (GQA, gate/up).
        for module in model.modules():
            if not isinstance(module, GroupedQueryAttention):
                continue
            wqkv = getattr(module, "wqkv", None)
            if wqkv is None:
                continue

            weight = getattr(wqkv, "weight", None)
            bias = getattr(wqkv, "bias", None)
            if weight is not None:
                if module.wqkv_extra_dims:
                    merge_op = Sequential([
                        Rename("grad.qkv: grad"),
                        Inverse(KeepThisTP()),
                        Inverse(
                            GQAMergeQKVG(
                                group_num=module.num_kv_heads,
                                head_dim=module.head_dim,
                                gate_dims=module.wqkv_extra_dims,
                            )
                        ),
                    ])
                else:
                    merge_op = Sequential([
                        Rename("grad.qkv: grad"),
                        Inverse(KeepThisTP()),
                        Inverse(
                            GQAMergeQKV(
                                group_num=module.num_kv_heads,
                                head_dim=module.head_dim,
                            )
                        ),
                    ])
                merge_ops[id(weight)] = merge_op

            if bias is not None and not module.wqkv_extra_dims:
                merge_ops[id(bias)] = Sequential([
                    Rename("grad.qkv: grad"),
                    Inverse(KeepThisTP()),
                    Inverse(
                        GQAMergeQKVBias(
                            group_num=module.num_kv_heads,
                            head_dim=module.head_dim,
                        )
                    ),
                ])

        for module in model.modules():
            if not isinstance(module, FeedForward):
                continue
            w1 = getattr(module, "w1", None)
            w2 = getattr(module, "w2", None)
            if w1 is not None and hasattr(w1, "weight"):
                group = "ETP" if getattr(w1, "use_moe", False) else "TP"
                merge_ops.setdefault(
                    id(w1.weight),
                    Sequential([
                        Rename("grad.gate_up: grad"),
                        Inverse(KeepThisTP(group=group)),
                        Inverse(FFNMergeGateUp(group=group)),
                    ]),
                )
            if w2 is not None and hasattr(w2, "weight"):
                group = "ETP" if getattr(w2, "use_moe", False) else "TP"
                merge_ops.setdefault(
                    id(w2.weight),
                    Inverse(RowParallel(group=group) + KeepThisTP(group=group)),
                )

        for module in model.modules():
            if not isinstance(module, GroupedExperts):
                continue
            if getattr(module, "w1", None) is not None:
                merge_ops.setdefault(
                    id(module.w1),
                    Sequential([
                        UnbindMoE(moe_key_prefix="grad"),
                        Inverse(ColumnParallel(group="ETP") + KeepThisTP(group="ETP")),
                    ]),
                )
            if getattr(module, "w2", None) is not None:
                merge_ops.setdefault(
                    id(module.w2),
                    Sequential([
                        UnbindMoE(moe_key_prefix="grad"),
                        Inverse(RowParallel(group="ETP") + KeepThisTP(group="ETP")),
                    ]),
                )

        # Generic TP-sharded linear weights.
        for module in model.modules():
            if isinstance(module, ColumnParallelLinear):
                weight = getattr(module, "weight", None)
                if weight is not None and weight.ndim == 2:
                    group = "ETP" if getattr(module, "use_moe", False) else "TP"
                    merge_ops.setdefault(
                        id(weight),
                        Inverse(ColumnParallel(group=group) + KeepThisTP(group=group)),
                    )
            elif isinstance(module, RowParallelLinear):
                weight = getattr(module, "weight", None)
                if weight is not None and weight.ndim == 2:
                    group = "ETP" if getattr(module, "use_moe", False) else "TP"
                    merge_ops.setdefault(
                        id(weight),
                        Inverse(RowParallel(group=group) + KeepThisTP(group=group)),
                    )
            elif isinstance(module, MoEGate):
                weight = getattr(module, "weight", None)
                if weight is not None:
                    merge_ops.setdefault(id(weight), identity)

        for _name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            merge_op = merge_ops.get(id(param))

            if merge_op is None:
                if getattr(param, "tensor_model_parallel", False):
                    partition_dim = getattr(param, "partition_dim", -1)
                    if partition_dim == 0:
                        merge_op = Inverse(ColumnParallel() + KeepThisTP())
                    elif partition_dim == 1:
                        merge_op = Inverse(RowParallel() + KeepThisTP())
                    else:
                        merge_op = identity
                else:
                    merge_op = identity

            param.merge_op = merge_op
