"""Utils for development, auto-analyze, regression tests."""

import os
from collections.abc import Callable

import torch
import torch.distributed as dist


def capture_trace(*, at: int, fw_file: str, bw_file: str, rank: int = 0, warmup: int = 2):
    """
    Capture a single call's forward/backward traces by running a short window profiler
    and then splitting the combined trace using explicit FW/BW markers.

    Behavior:
    - The decorator counts calls to the wrapped function.
    - When the call index reaches (warmup + at), it starts a profiler window for
      two consecutive calls (current and next).
    - It inserts `record_function` markers to bound FW and BW in that window and
      exports a combined trace, then splits into `fw_file` and `bw_file`.
    - Only the specified `rank` records traces; other ranks run normally.

    Notes:
    - Uses `torch.cuda.synchronize()` before/after the wrapped function when the
      forward trace is active, and keeps synchronize calls outside the profiler
      window to avoid capturing them.

    Example:
        @capture_trace(
            at=10,
            fw_file="moe_block_forward_rank0_call10.json",
            bw_file="moe_block_backward_rank0_call10.json",
            rank=0,
            warmup=2,
        )
        def forward(self, x):
            ...
            return out
    """

    def decorator(fn: Callable):
        from functools import wraps

        call_attr = f"_capture_trace_calls_{fn.__qualname__}"
        window_prof_attr = f"_capture_trace_window_prof_{fn.__qualname__}"
        window_active_attr = f"_capture_trace_window_active_{fn.__qualname__}"
        window_stop_call_attr = f"_capture_trace_window_stop_call_{fn.__qualname__}"
        bw_marked_attr = f"_capture_trace_bw_marked_{fn.__qualname__}"

        @wraps(fn)
        def wrapper(*args, **kwargs):
            import json

            if not hasattr(wrapper, call_attr):
                setattr(wrapper, call_attr, 0)
            cur = getattr(wrapper, call_attr) + 1
            setattr(wrapper, call_attr, cur)

            is_rank = not dist.is_initialized() or dist.get_rank() == rank
            if not is_rank:
                return fn(*args, **kwargs)

            def _split_trace(trace_path: str, fw_out: str, bw_out: str) -> None:
                try:
                    with open(trace_path) as handle:
                        data = json.load(handle)
                except (OSError, json.JSONDecodeError):
                    return

                events = data.get("traceEvents", [])

                def _find_marker(name: str):
                    for ev in events:
                        if ev.get("name") == name:
                            ts = ev.get("ts")
                            if ts is None:
                                continue
                            dur = ev.get("dur", 0) or 0
                            return ts, ts + dur
                    return None

                fw_marker = _find_marker("MOE_FW_START")
                fw_end_marker = _find_marker("MOE_FW_END")
                bw_marker = _find_marker("MOE_BW_START")
                bw_end_marker = _find_marker("MOE_BW_END")
                if fw_marker is None or fw_end_marker is None or bw_marker is None or bw_end_marker is None:
                    return

                fw_start, _ = fw_marker
                fw_end = fw_end_marker[1]
                bw_start, _ = bw_marker
                bw_end = bw_end_marker[1]

                def _filter_events(start: float, end: float):
                    filtered = []
                    for ev in events:
                        if ev.get("ph") == "M":
                            filtered.append(ev)
                            continue
                        ts = ev.get("ts")
                        if ts is None:
                            continue
                        dur = ev.get("dur", 0) or 0
                        ev_end = ts + dur
                        if ts <= end and ev_end >= start:
                            filtered.append(ev)
                    return filtered

                fw_events = _filter_events(fw_start, fw_end)
                bw_events = _filter_events(bw_start, bw_end)

                fw_data = dict(data)
                fw_data["traceEvents"] = fw_events
                bw_data = dict(data)
                bw_data["traceEvents"] = bw_events

                with open(fw_out, "w") as handle:
                    json.dump(fw_data, handle)
                with open(bw_out, "w") as handle:
                    json.dump(bw_data, handle)

            if hasattr(wrapper, window_active_attr) and getattr(wrapper, window_active_attr, False):
                stop_call = getattr(wrapper, window_stop_call_attr, None)
                if stop_call is not None and cur >= stop_call:
                    window_prof = getattr(wrapper, window_prof_attr)
                    torch.cuda.synchronize()
                    try:
                        window_prof.__exit__(None, None, None)
                        combined_path = os.path.abspath(f"{fw_file}.combined.json")
                        window_prof.export_chrome_trace(combined_path)
                        _split_trace(combined_path, os.path.abspath(fw_file), os.path.abspath(bw_file))
                    except RuntimeError:
                        pass
                    finally:
                        setattr(wrapper, window_active_attr, False)

            torch.cuda.synchronize()
            if cur == warmup + at and not getattr(wrapper, window_active_attr, False):
                window_prof = torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True,
                )
                window_prof.__enter__()
                setattr(wrapper, window_prof_attr, window_prof)
                setattr(wrapper, window_active_attr, True)
                setattr(wrapper, window_stop_call_attr, cur + 1)
                setattr(wrapper, bw_marked_attr, False)

            if getattr(wrapper, window_active_attr, False) and cur == warmup + at:
                with torch.profiler.record_function("MOE_FW_START"):
                    pass
            out = fn(*args, **kwargs)
            if getattr(wrapper, window_active_attr, False) and cur == warmup + at:
                torch.cuda.synchronize()
                with torch.profiler.record_function("MOE_FW_END"):
                    pass

            def _find_tensor(obj):
                if isinstance(obj, torch.Tensor):
                    return obj
                if isinstance(obj, (list, tuple)):
                    for item in obj:
                        t = _find_tensor(item)
                        if t is not None:
                            return t
                if isinstance(obj, dict):
                    for item in obj.values():
                        t = _find_tensor(item)
                        if t is not None:
                            return t
                return None

            if torch.is_grad_enabled() and cur == warmup + at and getattr(wrapper, window_active_attr, False):
                if not getattr(wrapper, bw_marked_attr, False):
                    out_tensor = _find_tensor(out)
                    if out_tensor is not None and out_tensor.requires_grad:
                        marked = {"done": False}

                        def _bw_start(_grad):
                            if marked["done"]:
                                return _grad
                            marked["done"] = True
                            with torch.profiler.record_function("MOE_BW_START"):
                                pass

                            def _bw_finish():
                                with torch.profiler.record_function("MOE_BW_END"):
                                    pass

                            torch.autograd.Variable._execution_engine.queue_callback(_bw_finish)
                            setattr(wrapper, bw_marked_attr, True)
                            return _grad

                        out_tensor.register_hook(_bw_start)

            return out

        return wrapper

    return decorator


class TBRecord(dict):
    def __init__(self, path=""):
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )

        self.cache = {}
        self.ea = EventAccumulator(path)
        self.refresh()

    def refresh(self):
        self.ea.Reload()
        self._keys = self.ea.Tags()["scalars"]

    def keys(self):
        return self._keys

    def __contains__(self, key: object) -> bool:
        return key in self._keys

    def __getitem__(self, key):
        self.refresh()
        return self.ea.Scalars(key)

    def asciichart(self, scalar_key: str, height: int = 18) -> str:
        import asciichartpy as ac

        events = self[scalar_key]
        values = [evt.value for evt in events]

        class Repr(str):
            def __repr__(self):
                return self

        return Repr(ac.plot(values, {"height": height, "format": "{:.3e}"}))
