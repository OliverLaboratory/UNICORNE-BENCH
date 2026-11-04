#!/usr/bin/env python3
"""
Print the Table 4 metrics for any scored tool, next to the published tools.

Reads results/<DATASET>/<TOOL>/results.csv (written by compute_metrics.py) and
averages over the evaluated ensembles (filtered_ensembles_v2.csv minus the
fold-quality blacklist), exactly as tab_overall_performance.py does for the
paper. Ensembles missing from a tool's results.csv (no output, or fewer
predictions than compute_metrics.py --min-preds) are left out of its averages
and lower its N.

Usage:
  python pipeline/03_compute_results/summarize_tools.py my_model
  python pipeline/03_compute_results/summarize_tools.py my_model other_model --no-published
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PUBLISHED = ["bioemu", "esmflow_pdb", "esmdiff", "alphaflow_pdb",
             "cfrandom", "afsample2", "afsample3_biorxiv", "speach_af"]


def load_table_module():
    path = PROJECT_ROOT / "pipeline" / "04_build_figures" / "tab_overall_performance.py"
    spec = importlib.util.spec_from_file_location("tab_overall_performance", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tools", nargs="+", help="Tool names under results/<DATASET>/")
    ap.add_argument("--dataset", default="UNICORNE_BENCH")
    ap.add_argument("--no-published", action="store_true",
                    help="Leave out the eight tools from the paper")
    args = ap.parse_args()

    tab = load_table_module()
    n_eval = len(tab._load_filter(args.dataset))
    tools = args.tools + ([] if args.no_published else [t for t in PUBLISHED if t not in args.tools])

    print(f"{args.dataset}: {n_eval} evaluated ensembles\n")
    print(f"{'Tool':<20s} {'N':>4s}  {'Recall@0.9 %':>13s}  {'FRR %':>5s}  {'TM-bar':>13s}  {'EMD':>13s}")
    for tool in tools:
        if tool == PUBLISHED[0] and not args.no_published:
            print("-- published --")
        m = tab.compute_metrics(args.dataset, tool)
        if m is None:
            print(f"{tool:<20s} no results/{args.dataset}/{tool}/results.csv")
            continue
        print(f"{tool:<20s} {m['n']:>4d}  {m['dr'] * 100:>5.1f} ± {m['dr_sd'] * 100:>4.1f}  "
              f"{m['cr'] * 100:>5.1f}  {m['tm_bar']:.3f} ± {m['tm_bar_sd']:.3f}  {m['emd']:.3f} ± {m['emd_sd']:.3f}")
    print(f"\nN < {n_eval}: ensembles missing from the tool's results.csv are left out of its averages.")


if __name__ == "__main__":
    main()
