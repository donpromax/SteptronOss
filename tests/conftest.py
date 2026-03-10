import os

import pytest

_NODE2_HELP = "node2 tests require `torchrun --nproc-per-node=2 -m pytest -m node2`."
_NODE2_REQUIRED_ENV = ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE", "LOCAL_RANK")


def _node2_skip_reason() -> str | None:
    missing_env = [name for name in _NODE2_REQUIRED_ENV if name not in os.environ]
    if missing_env:
        return _NODE2_HELP

    try:
        world_size = int(os.environ["WORLD_SIZE"])
    except ValueError:
        return f"{_NODE2_HELP} Got invalid WORLD_SIZE={os.environ['WORLD_SIZE']!r}."

    if world_size != 2:
        return f"{_NODE2_HELP} Got WORLD_SIZE={world_size}."

    return None


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    del config

    reason = _node2_skip_reason()
    if reason is None:
        return

    skip_marker = pytest.mark.skip(reason=reason)
    for item in items:
        if item.get_closest_marker("node2") is not None:
            item.add_marker(skip_marker)
