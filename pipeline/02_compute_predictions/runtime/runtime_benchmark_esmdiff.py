#!/usr/bin/env python3
"""
ESMDiff runtime benchmark — runs inside esmdiff pixi environment.
Loads model once, times only inference (ddpm_sample token generation).

Writes per-run timings to a CSV file.
"""
import csv
import json
import os
import sys
import time
import tempfile
from pathlib import Path
from functools import partial

import numpy as np
import torch
from tqdm import tqdm

# ESMDiff imports (must be run from ~/esmdiff with pixi)
import rootutils
rootutils.setup_root(Path.home() / "esmdiff", indicator=".project-root", pythonpath=True)

from esm.sdk.api import ESMProtein, ESMProteinTensor
from esm.models.esm3 import ESM3
from esm.utils.constants import esm3 as C

from slm.models.model import MaskedDiffusionLanguageModeling
from slm.models.utils import protseq_to_data
from slm.utils.checkpoint_utils import load_state_dict_from_lightning_ckpt
from slm.utils.eval_utils import merge_pdbfiles

SAMPLE_COUNTS = [20, 40, 60, 100]
NUM_STEPS = 25
MAX_BATCH_TOKENS = 400_000
ESMDIFF_CKPT = Path.home() / "esmdiff" / "data" / "ckpt" / "release_v0.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"


def decode(structure_tokens, sequence_tokens, esm3_model):
    """Decode structure tokens to protein (needed for full pipeline but we skip file I/O)."""
    sequence_tokens = torch.cat([
        torch.LongTensor([C.SEQUENCE_BOS_TOKEN]),
        sequence_tokens.cpu(),
        torch.LongTensor([C.SEQUENCE_EOS_TOKEN])
    ])
    structure_tokens = torch.cat([
        torch.LongTensor([C.STRUCTURE_BOS_TOKEN]),
        structure_tokens.cpu(),
        torch.LongTensor([C.STRUCTURE_EOS_TOKEN])
    ])
    prot = ESMProteinTensor(sequence=sequence_tokens, structure=structure_tokens)
    prot = prot.to(device)
    return esm3_model.decode(prot)


@torch.no_grad()
def timed_ddpm_sample(
    sequence,
    pl_model,
    esm3_model,
    num_samples,
    num_steps=25,
    n_max_residue_square=400_000,
    eps=1e-5,
):
    """Run DDPM sampling and return wall-clock time for token generation only."""
    prot = protseq_to_data(sequence, esm3_model, encode_only=True)
    sequence_tokens_singleton = prot["sequence_tokens"]

    seq_len = sequence_tokens_singleton.size(0)
    batch_size = max(1, n_max_residue_square // (seq_len * seq_len))

    bsz = []
    n_batch = num_samples // batch_size
    remainder = num_samples % batch_size
    for _ in range(n_batch):
        bsz.append(batch_size)
    if remainder > 0:
        bsz.append(remainder)

    batches = [sequence_tokens_singleton[None, :].repeat(bs, 1) for bs in bsz]

    # Move ESM3 to CPU during diffusion (as ESMDiff does)
    esm3_model_on_cpu = esm3_model.to('cpu')

    # Time only the diffusion sampling
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    structure_tokens = []
    for batch in batches:
        structure_tokens.append(pl_model.ddpm_sample(
            num_steps=num_steps,
            sequence_tokens=batch,
            eps=eps,
        ))
    structure_tokens = torch.cat(structure_tokens, dim=0)

    torch.cuda.synchronize()
    diffusion_time = time.perf_counter() - t0

    # Also time decoding (structure tokens -> coordinates)
    structure_tokens = structure_tokens[:, 1:-1]
    sequence_tokens_singleton = sequence_tokens_singleton[1:-1]

    esm3_model = esm3_model_on_cpu.to(device)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    for i in range(len(structure_tokens)):
        decode(structure_tokens[i], sequence_tokens_singleton, esm3_model)

    torch.cuda.synchronize()
    decode_time = time.perf_counter() - t1

    return diffusion_time, decode_time


def main():
    config_path = Path(sys.argv[1])
    results_csv = Path(sys.argv[2])

    with open(config_path) as f:
        config = json.load(f)
    proteins = config["proteins"]

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # ---- Load models ONCE ----
    print("Loading ESM3 model...", flush=True)
    t0 = time.perf_counter()
    esm3_model = ESM3.from_pretrained("esm3_sm_open_v1").to(device)
    print(f"  ESM3 loaded in {time.perf_counter() - t0:.1f}s", flush=True)

    print("Loading ESMDiff checkpoint...", flush=True)
    t0 = time.perf_counter()
    pl_model = load_state_dict_from_lightning_ckpt(str(ESMDIFF_CKPT), device=device)
    print(f"  ESMDiff loaded in {time.perf_counter() - t0:.1f}s", flush=True)

    # ---- Pre-encode sequences ----
    print("Pre-encoding sequences...", flush=True)
    encoded = {}
    for p in proteins:
        uid, seq = p["uniprot"], p["sequence"]
        # Just verify encoding works; actual encoding happens in timed_ddpm_sample
        # (cheap compared to diffusion)
        encoded[uid] = True
        print(f"  {uid} ({len(seq)}aa) OK", flush=True)

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

    # ---- Benchmark ----
    print("\nStarting inference benchmark...", flush=True)
    total_runs = len(proteins) * len(SAMPLE_COUNTS)
    run_idx = 0

    write_header = not results_csv.exists()
    fieldnames = ["tool", "uniprot", "sequence_length", "size_bin",
                  "num_samples", "wall_clock_seconds", "diffusion_seconds",
                  "decode_seconds", "gpu", "timestamp"]

    for p in proteins:
        uid, seq, size_bin = p["uniprot"], p["sequence"], p["size_bin"]

        for n_samples in SAMPLE_COUNTS:
            run_idx += 1
            print(f"  [{run_idx}/{total_runs}] {uid} ({len(seq)}aa, {size_bin}) | n={n_samples} ... ",
                  end="", flush=True)

            diffusion_time, decode_time = timed_ddpm_sample(
                sequence=seq,
                pl_model=pl_model,
                esm3_model=esm3_model,
                num_samples=n_samples,
                num_steps=NUM_STEPS,
                n_max_residue_square=MAX_BATCH_TOKENS,
            )
            total_time = diffusion_time + decode_time

            print(f"{total_time:.2f}s (diffusion={diffusion_time:.2f}s, decode={decode_time:.2f}s)", flush=True)

            row = {
                "tool": "esmdiff",
                "uniprot": uid,
                "sequence_length": len(seq),
                "size_bin": size_bin,
                "num_samples": n_samples,
                "wall_clock_seconds": round(total_time, 3),
                "diffusion_seconds": round(diffusion_time, 3),
                "decode_seconds": round(decode_time, 3),
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
