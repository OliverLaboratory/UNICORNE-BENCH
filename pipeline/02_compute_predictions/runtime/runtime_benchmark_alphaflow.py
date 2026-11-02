#!/usr/bin/env python3
"""
AlphaFlow runtime benchmark — runs inside alphaflow_venv.
Loads model once, times only inference (model.inference calls).

Writes per-run timings to a CSV file.
"""
import csv
import json
import os
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

# Must run from alphaflow directory for imports to work
ALPHAFLOW_DIR = Path.home() / "alphaflow"
sys.path.insert(0, str(ALPHAFLOW_DIR))

from alphaflow.data.data_modules import collate_fn
from alphaflow.model.wrapper import ESMFoldWrapper
from alphaflow.utils.tensor_utils import tensor_tree_map
from alphaflow.data.inference import CSVDataset
from alphaflow.config import model_config

SAMPLE_COUNTS = [20, 40, 60, 100]
WEIGHTS_PATH = ALPHAFLOW_DIR / "weights" / "esmflow_md_distilled.pt"


def main():
    config_path = Path(sys.argv[1])
    results_csv = Path(sys.argv[2])

    with open(config_path) as f:
        config = json.load(f)
    proteins = config["proteins"]

    torch.set_float32_matmul_precision("high")

    # AlphaFlow config (same as predict.py)
    af_config = model_config('initial_training', train=True, low_prec=True)
    data_cfg = af_config.data
    data_cfg.common.use_templates = False
    data_cfg.common.max_recycling_iters = 0

    # Schedule for no_diffusion mode
    tmax = 1.0
    steps = 10
    schedule = np.linspace(tmax, 0, steps + 1)

    # ---- Load model ONCE ----
    print("Loading AlphaFlow (ESMFlow-MD) model...", flush=True)
    t_load_start = time.perf_counter()

    ckpt = torch.load(str(WEIGHTS_PATH), map_location='cpu', weights_only=False)
    model = ESMFoldWrapper(**ckpt['hyper_parameters'], training=False)
    model.model.load_state_dict(ckpt['params'], strict=False)
    model = model.cuda()
    model.eval()

    t_load = time.perf_counter() - t_load_start
    print(f"Model loaded in {t_load:.1f}s", flush=True)

    # ---- Prepare dataset items (tokenize sequences) ----
    # Write a temp CSV with all proteins for CSVDataset
    import tempfile
    tmp_csv = Path(tempfile.mktemp(suffix=".csv", dir=str(Path.home())))
    with open(tmp_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "seqres"])
        w.writeheader()
        for p in proteins:
            w.writerow({"name": p["uniprot"], "seqres": p["sequence"]})

    print("Tokenizing sequences...", flush=True)
    dataset = CSVDataset(data_cfg, str(tmp_csv))
    # Build lookup: name -> dataset item
    items = {}
    for i in range(len(dataset)):
        item = dataset[i]
        items[item['name']] = item
    tmp_csv.unlink()
    print(f"  {len(items)} sequences tokenized", flush=True)

    # ---- GPU info ----
    gpu_info = "unknown"
    try:
        import subprocess
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            gpu_info = out.stdout.strip()
    except Exception:
        pass

    # ---- Benchmark inference ----
    print("\nStarting inference benchmark...", flush=True)
    total_runs = len(proteins) * len(SAMPLE_COUNTS)
    run_idx = 0

    write_header = not results_csv.exists()
    fieldnames = ["tool", "uniprot", "sequence_length", "size_bin",
                  "num_samples", "wall_clock_seconds", "gpu", "timestamp"]

    for p in proteins:
        uid, seq, size_bin = p["uniprot"], p["sequence"], p["size_bin"]
        item = items[uid]

        for n_samples in SAMPLE_COUNTS:
            run_idx += 1
            print(f"  [{run_idx}/{total_runs}] {uid} ({len(seq)}aa, {size_bin}) | n={n_samples} ... ",
                  end="", flush=True)

            # Time only inference (N forward passes)
            torch.cuda.synchronize()
            t0 = time.perf_counter()

            with torch.no_grad():
                for j in range(n_samples):
                    batch = collate_fn([item])
                    batch = tensor_tree_map(lambda x: x.cuda(), batch)
                    prots = model.inference(
                        batch, as_protein=True, noisy_first=True,
                        no_diffusion=True, schedule=schedule, self_cond=False
                    )

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            print(f"{elapsed:.2f}s", flush=True)

            row = {
                "tool": "esmflow_pdb",
                "uniprot": uid,
                "sequence_length": len(seq),
                "size_bin": size_bin,
                "num_samples": n_samples,
                "wall_clock_seconds": round(elapsed, 3),
                "gpu": gpu_info,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            with open(results_csv, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if write_header:
                    writer.writeheader()
                    write_header = False
                writer.writerow(row)


if __name__ == "__main__":
    main()
