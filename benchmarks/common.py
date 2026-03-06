from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from steptronoss.utils.optimizable import OPTIMIZABLE_REGISTER, set_optimization


@dataclass
class BackendResult:
    name: str
    ok: bool
    error: str | None
    payload: dict[str, object] | None


def list_backends(target: str) -> list[str | None]:
    if target not in OPTIMIZABLE_REGISTER:
        raise KeyError(f"Optimizable target not found: {target}")
    alternatives = list(OPTIMIZABLE_REGISTER[target]["alternatives"].keys())
    return [None, *alternatives]


def run_with_backends(
    target: str,
    runner: Callable[[str | None], dict[str, object]],
) -> list[BackendResult]:
    results: list[BackendResult] = []
    for backend in list_backends(target):
        try:
            if backend is None:
                set_optimization(**{target.split(".")[-1]: None})
                name = "baseline"
            else:
                set_optimization(**{target.split(".")[-1]: backend})
                name = backend
            payload = runner(backend)
            results.append(BackendResult(name=name, ok=True, error=None, payload=payload))
        except Exception as exc:
            results.append(BackendResult(name=name if backend else "baseline", ok=False, error=str(exc), payload=None))
    return results
