#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-${ROOT}/../.venv/bin/python}"

echo "[benchmarks] grouped_gemm"
timeout 3m "${PY}" "${ROOT}/benchmark_grouped_gemm.py"

echo "[benchmarks] moe_scatter"
timeout 3m "${PY}" "${ROOT}/benchmark_moe_scatter.py"

echo "[benchmarks] dispatcher (torchrun)"
timeout 3m "${ROOT}/../.venv/bin/torchrun" --nproc-per-node 8 "${ROOT}/benchmark_dispatcher.py"
