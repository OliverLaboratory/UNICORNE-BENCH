#!/usr/bin/env python3
"""
BioEMU runtime benchmark — runs inside bioemu_venv.
Loads model once, times only inference (generate_batch calls).

Writes per-run timings to a CSV file.
"""
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data.batch import Batch

from bioemu.sample import get_context_chemgraph, generate_batch
from bioemu.model_utils import load_model, load_sdes, maybe_download_checkpoint

SAMPLE_COUNTS = [20, 40, 60, 100]
BATCH_SIZE_100 = 100  # tuned for RTX 5090 32GB (production uses 500 for H200 80GB)


def main():
    config_path = Path(sys.argv[1])  # JSON with proteins list
    results_csv = Path(sys.argv[2])

    with open(config_path) as f:
        config = json.load(f)
    proteins = config["proteins"]

    # ---- Load model ONCE ----
    print("Loading BioEMU model...", flush=True)
    t_load_start = time.perf_counter()

    ckpt_path, model_config_path = maybe_download_checkpoint(
        model_name="bioemu-v1.1", ckpt_path=None, model_config_path=None
    )
    score_model = load_model(ckpt_path, model_config_path)
    sdes = load_sdes(model_config_path=model_config_path, cache_so3_dir=None)

    # Load denoiser config
    import yaml, hydra
    denoiser_config_path = Path(__file__).parent / "../bioemu_venv/lib/python3.11/site-packages/bioemu/config/denoiser/dpm.yaml"
    # Find it dynamically
    import bioemu
    denoiser_config_path = Path(bioemu.__path__[0]) / "config" / "denoiser" / "dpm.yaml"
    with open(denoiser_config_path) as f:
        denoiser_config = yaml.safe_load(f)
    denoiser = hydra.utils.instantiate(denoiser_config)

    t_load = time.perf_counter() - t_load_start
    print(f"Model loaded in {t_load:.1f}s", flush=True)

    # ---- Pre-compute MSA embeddings for all proteins ----
    # This is sequence-dependent and expensive, so we do it once per protein
    # and exclude it from inference timing
    print("Pre-computing MSA embeddings...", flush=True)
    embeddings = {}
    for p in proteins:
        uid, seq = p["uniprot"], p["sequence"]
        print(f"  Embedding {uid} ({len(seq)}aa)...", end=" ", flush=True)
        t0 = time.perf_counter()
        ctx = get_context_chemgraph(sequence=seq, cache_embeds_dir=None)
        embed_time = time.perf_counter() - t0
        embeddings[uid] = ctx
        print(f"{embed_time:.1f}s", flush=True)

    # ---- Benchmark inference ----
    print("\nStarting inference benchmark...", flush=True)

    # Get GPU info
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

    total_runs = len(proteins) * len(SAMPLE_COUNTS)
    run_idx = 0

    write_header = not results_csv.exists()
    fieldnames = ["tool", "uniprot", "sequence_length", "size_bin",
                  "num_samples", "wall_clock_seconds", "gpu", "timestamp"]

    for p in proteins:
        uid, seq, size_bin = p["uniprot"], p["sequence"], p["size_bin"]
        ctx = embeddings[uid]
        batch_size = max(1, int(BATCH_SIZE_100 * (100 / len(seq)) ** 2))

        for n_samples in SAMPLE_COUNTS:
            run_idx += 1
            print(f"  [{run_idx}/{total_runs}] {uid} ({len(seq)}aa, {size_bin}) | n={n_samples} ... ",
                  end="", flush=True)

            # Time only the generation (batched denoising)
            torch.cuda.synchronize()
            t0 = time.perf_counter()

            generated = 0
            while generated < n_samples:
                n = min(batch_size, n_samples - generated)
                context_batch = Batch.from_data_list([ctx] * n)
                device = torch.device("cuda:0")
                _ = denoiser(sdes=sdes, device=device, batch=context_batch, score_model=score_model)
                generated += n

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            print(f"{elapsed:.2f}s", flush=True)

            row = {
                "tool": "bioemu",
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
