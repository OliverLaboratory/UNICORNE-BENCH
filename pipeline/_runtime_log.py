"""
Shared runtime / GPU-usage logger for the prediction runners.

Each run_*.py in pipeline/02_compute_predictions/ calls log_ensemble()
when an ensemble's compute finishes (success or failure) so we have a
durable per-ensemble row for cost accounting.

Output:
    logs/runtime_accounting.csv

Columns:
    timestamp_utc    ISO-8601, when the row was written
    host             socket.gethostname()
    gpu_id           value of $CUDA_VISIBLE_DEVICES (string; empty if unset)
    gpu_model        nvidia-smi --query-gpu=name (looked up at row-write time)
    tool             tool name (e.g. "afsample2", "bioemu", "alphaflow_pdb")
    dataset          dataset name (e.g. "UNICORNE_500")
    ensemble         ensemble_ID
    seq_len          int
    structures       int - new structures written this invocation
    wall_seconds     float - wall-clock for the ensemble's compute
    status           "ok" | "failed" | "skip"
    note             optional free-text (e.g. "apptainer FAILED rc=1")

The file is append-only and uses a per-process flock so concurrent shards
on the same host don't clobber each other (works for NFS too on Linux).
"""
from __future__ import annotations

import csv
import datetime
import fcntl
import os
import socket
import subprocess
from pathlib import Path

_FIELDS = ("timestamp_utc", "host", "gpu_id", "gpu_model", "tool", "dataset",
           "ensemble", "seq_len", "structures", "wall_seconds", "status", "note")


def _gpu_model_for(gpu_id: str) -> str:
    """Best-effort `nvidia-smi` lookup. Returns empty string on any failure."""
    if not gpu_id:
        return ""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "-i", gpu_id],
            capture_output=True, text=True, timeout=5,
        )
        return r.stdout.strip().splitlines()[0] if r.returncode == 0 else ""
    except Exception:
        return ""


def log_ensemble(
    tool: str, dataset: str, ensemble: str,
    seq_len: int, structures: int, wall_seconds: float,
    status: str = "ok", note: str = "",
    csv_path: Path | None = None,
) -> None:
    """Append one row. Errors are swallowed so accounting failure can't
    take down a prediction run."""
    try:
        if csv_path is None:
            here = Path(__file__).resolve().parent
            project_root = here.parent
            csv_path = project_root / "logs" / "runtime_accounting.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        row = {
            "timestamp_utc": datetime.datetime.utcnow().isoformat(timespec="seconds"),
            "host": socket.gethostname(),
            "gpu_id": gpu_id,
            "gpu_model": _gpu_model_for(gpu_id),
            "tool": tool,
            "dataset": dataset,
            "ensemble": ensemble,
            "seq_len": seq_len,
            "structures": structures,
            "wall_seconds": f"{wall_seconds:.1f}",
            "status": status,
            "note": note,
        }

        new_file = not csv_path.exists()
        # Open + flock so concurrent shards on the same host don't interleave
        # partial CSV writes. NFS-safe with proper flock support (Linux NFSv4).
        with open(csv_path, "a", newline="") as f:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            except OSError:
                pass  # NFS might reject flock — accept interleave risk
            w = csv.DictWriter(f, fieldnames=_FIELDS)
            if new_file:
                w.writeheader()
            w.writerow(row)
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
    except Exception:
        # Accounting must never break the actual prediction job.
        pass
