#!/usr/bin/env python3
"""
Table: Overall performance on DANCE and UNICORNE.

Computes macro-averaged DR, CR, EU, EMD per dataset x tool from the
per-ensemble results CSV and TM matrices.

Input:
  results/{DATASET}/{TOOL}/results.csv
  results/{DATASET}/{TOOL}/matrices_tmalign/{eid}/tm_xy.npy

Output:
  figures/tab_overall_performance.tex (LaTeX table rows)
  stdout: formatted table

Usage:
  python pipeline/04_build_figures/tab_overall_performance.py
  python pipeline/04_build_figures/tab_overall_performance.py --datasets DANCE WILL_TM
"""
from __future__ import annotations
import argparse, csv
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Two groups, separated by \midrule in the LaTeX output.
GENERATIVE_TOOLS = ["bioemu", "esmflow_pdb", "esmdiff", "alphaflow_pdb"]
# AFsample3 is reported from the `afsample3_biorxiv` run, which uses the masking
# sweep from Table 1 of the AFsample3 paper ({0.0 .. 0.5}) -- i.e. the method at
# its own published settings. An earlier run under our AFsample2-matched
# schedule ({0.05 .. 0.30}) exists at results/<ds>/afsample3/ but is not
# reported: running AF3 at AF2's masking levels understates it.
MSA_SUBSAMPLING_TOOLS = ["cfrandom", "afsample2", "afsample3_biorxiv", "speach_af"]
TOOLS = GENERATIVE_TOOLS + MSA_SUBSAMPLING_TOOLS
TOOL_DISPLAY = {"esmflow_pdb":"\\esmflow","bioemu":"\\bioemu","cfrandom":"\\cfrandom",
                 "esmdiff":"\\esmdiff","alphaflow_pdb":"\\alphaflow",
                 "afsample2":"\\afsampletwo",
                 "afsample3_biorxiv":"\\afsamplethree",
                 "speach_af":"\\speachaf"}

# Kernel: "tmlocked" (sequence-locked TM-score, primary) or "tmalign" (legacy)
import os as _os
KERNEL = _os.environ.get("CONF_BENCH_KERNEL", "tmalign")
if KERNEL == "tmlocked":
    MAT_SUBDIR = "matrices_tmlocked"
    GT_MAT_SUBDIR = "gt_matrices_tmlocked"
    XY_FILES = ["tm_xy.npy"]
    XX_FILES = ["tm_xx.npy"]
    RECALL_COL = "recall_at_tmlocked_0.9"
else:
    MAT_SUBDIR = "matrices_tmalign"
    GT_MAT_SUBDIR = "gt_matrices_tmalign"
    XY_FILES = ["tm_xy.npy", "tm_xy_ab.npy"]
    XX_FILES = ["tm_xx.npy", "tm_xx_ab.npy"]
    RECALL_COL = "recall_at_tm_0.9"

def _load_filter(ds):
    """Load filtered_ensembles_v2.csv minus the fold-quality blacklist.

    The blacklist (`fold_quality_blacklist.csv`, one UniProt + reason per
    row) excludes ensembles whose GT structures are artifactual, IDP, or
    otherwise unsuitable for benchmarking conformational coverage.
    """
    fp = PROJECT_ROOT / "data" / "inputs" / ds / "filtered_ensembles_v2.csv"
    if not fp.exists():
        return None
    ids = set()
    with open(fp) as f:
        for r in csv.DictReader(f):
            eid = r.get("ensemble_ID", "").strip()
            if eid: ids.add(eid)
    bl = PROJECT_ROOT / "data" / "inputs" / ds / "fold_quality_blacklist.csv"
    if bl.exists():
        for r in csv.DictReader(open(bl)):
            ids.discard((r.get("uniprot") or "").strip())
    return ids


def compute_metrics(ds, tool):
    """Macro-average dr/cr/eu/emd from the per-ensemble results.csv.

    The new compute_metrics.py emits dr/cr/eu/emd as columns directly, so we
    just average them. Falls back to the old recompute-from-matrix path when
    the new columns are absent (legacy results.csv from compute_results.py).
    """
    csv_path = PROJECT_ROOT / "results" / ds / tool / "results.csv"
    if not csv_path.exists():
        return None
    keep_ids = _load_filter(ds)

    drs, crs, eus, emds = [], [], [], []
    # Per-ensemble dicts so we can run paired tests across tools later.
    drs_eid: dict[str, float] = {}
    crs_eid: dict[str, float] = {}
    eus_eid: dict[str, float] = {}
    emds_eid: dict[str, float] = {}
    have_new_cols = None  # set by inspecting the first row
    rows = list(csv.DictReader(open(csv_path)))
    if not rows:
        return None
    have_new_cols = all(k in rows[0] for k in ("dr", "cr", "eu", "emd"))

    if have_new_cols:
        for row in rows:
            eid = row.get("ensemble_ID", "").strip()
            if keep_ids is not None and eid not in keep_ids:
                continue
            def _f(k):
                v = row.get(k, "")
                try:
                    x = float(v)
                except (TypeError, ValueError):
                    return None
                return None if np.isnan(x) else x
            dr = _f("dr")
            if dr is None:
                # Treat NaN as zero discovery (ensemble evaluated but
                # predictions failed any GT at TM>=0.9). Keeps N aligned.
                dr = 0.0
            drs.append(dr); drs_eid[eid] = dr
            cr = _f("cr")
            cr_v = cr if cr is not None else 0.0
            crs.append(cr_v); crs_eid[eid] = cr_v
            eu = _f("eu")
            if eu is not None:
                eus.append(eu); eus_eid[eid] = eu
            emd = _f("emd")
            if emd is not None:
                emds.append(emd); emds_eid[eid] = emd
        if not drs:
            return None
        def _sd(xs):
            return float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0
        return {"n": len(drs),
                "dr": float(np.mean(drs)),
                "dr_sd": _sd(drs),
                "dr_eid": drs_eid,
                "cr": float(np.mean(crs)),
                "cr_sd": _sd(crs),
                "cr_eid": crs_eid,
                "eu": float(np.mean(eus)) if eus else float("nan"),
                "eu_sd": _sd(eus) if eus else float("nan"),
                "eu_eid": eus_eid,
                "emd": float(np.mean(emds)) if emds else float("nan"),
                "emd_sd": _sd(emds) if emds else float("nan"),
                "emd_eid": emds_eid}

    # Legacy path: recompute eu/emd from on-disk matrices_tmalign npy
    mat_dir = PROJECT_ROOT / "results" / ds / tool / MAT_SUBDIR
    for row in rows:
        eid = row.get("ensemble_ID", "").strip()
        if keep_ids is not None and eid not in keep_ids:
            continue
        try:
            dr = float(row.get(RECALL_COL, ""))
        except (TypeError, ValueError):
            continue
        if np.isnan(dr):
            dr = 0.0
        drs.append(dr)
        for name in XY_FILES:
            p = mat_dir / eid / name
            if p.exists():
                tm = np.load(p)
                eus.append(float(tm.max(axis=1).mean()))
                try:
                    import ot
                    floor = 0.3
                    sim = np.clip((tm - floor) / (1 - floor), 0, 1)
                    cost = 1 - sim
                    a = np.ones(cost.shape[0]) / cost.shape[0]
                    b = np.ones(cost.shape[1]) / cost.shape[1]
                    emds.append(float(ot.emd2(a, b, cost)))
                except Exception:
                    pass
                break

    if not drs:
        return None
    cr_indicators = [1.0 if d == 1.0 else 0.0 for d in drs]
    def _sd(xs):
        return float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0
    return {"n": len(drs),
            "dr": float(np.mean(drs)), "dr_sd": _sd(drs),
            "cr": float(np.mean(cr_indicators)), "cr_sd": _sd(cr_indicators),
            "eu": float(np.mean(eus)) if eus else float("nan"),
            "eu_sd": _sd(eus) if eus else float("nan"),
            "emd": float(np.mean(emds)) if emds else float("nan"),
            "emd_sd": _sd(emds) if emds else float("nan")}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+",
                        default=["DANCE_BENCH", "UNICORNE_BENCH"])
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "figures" / "tab_overall_performance.tex")
    args = parser.parse_args()

    print(f"{'Tool':<12s}", end="")
    for ds in args.datasets:
        print(f"  {'N':>6s} {'DR%':>6s} {'CR%':>6s} {'EU%':>6s} {'EMD':>6s}", end="")
    print()

    # Precompute all metrics once so we can rank columns
    metrics_by_tool_ds = {
        tool: {ds: compute_metrics(ds, tool) for ds in args.datasets}
        for tool in TOOLS
    }

    # For each (ds, metric, direction) rank tools.  We bold the tool with
    # the best mean AND any tool whose paired Wilcoxon signed-rank vs that
    # best tool is not significant after Bonferroni correction across the
    # remaining tools (p ≥ alpha / k).  Underline = best mean among the
    # not-bold-tied set.
    from scipy.stats import wilcoxon
    METRIC_KEYS = [("dr", "max"), ("cr", "max"), ("eu", "max"), ("emd", "min")]
    ALPHA = 0.05  # before Bonferroni
    rank_by_ds = {}
    pvals_by_ds: dict[str, dict[str, dict[str, float]]] = {}
    for ds in args.datasets:
        rank_by_ds[ds] = {}
        pvals_by_ds[ds] = {}
        for key, direction in METRIC_KEYS:
            vals = [(tool, metrics_by_tool_ds[tool][ds][key])
                    for tool in TOOLS
                    if metrics_by_tool_ds[tool][ds] is not None
                    and not np.isnan(metrics_by_tool_ds[tool][ds][key])]
            if not vals:
                continue
            vals.sort(key=lambda x: x[1], reverse=(direction == "max"))
            best_tool = vals[0][0]
            best_val = vals[0][1]
            best_eid = metrics_by_tool_ds[best_tool][ds].get(f"{key}_eid", {})

            # Paired Wilcoxon vs the best tool, on the intersection of
            # ensembles where both tools have a finite value.
            n_other = len(vals) - 1
            alpha_corr = ALPHA / max(1, n_other)
            tied: set[str] = {best_tool}
            pvals: dict[str, float] = {best_tool: float("nan")}
            for tool, _ in vals[1:]:
                other_eid = metrics_by_tool_ds[tool][ds].get(f"{key}_eid", {})
                common = set(best_eid) & set(other_eid)
                pairs = [(best_eid[e], other_eid[e]) for e in common
                         if not np.isnan(best_eid[e])
                         and not np.isnan(other_eid[e])]
                if len(pairs) < 10:
                    pvals[tool] = float("nan")
                    continue
                a = np.array([p[0] for p in pairs])
                b = np.array([p[1] for p in pairs])
                if np.allclose(a, b):
                    pvals[tool] = 1.0
                else:
                    try:
                        pvals[tool] = float(
                            wilcoxon(a, b, zero_method="wilcox").pvalue)
                    except ValueError:
                        pvals[tool] = float("nan")
                if (not np.isnan(pvals[tool])) and pvals[tool] >= alpha_corr:
                    tied.add(tool)
            pvals_by_ds[ds][key] = pvals

            # Underline = best mean amongst the non-tied set.
            rest = [(t, v) for t, v in vals if t not in tied]
            second_val = rest[0][1] if rest else None
            second_tools = ({t for t, v in rest if v == second_val}
                            if second_val is not None else set())
            rank_by_ds[ds][key] = (tied, second_tools)
            print(f"  {ds} {key}: best={best_tool} ({best_val:.3f}); "
                  f"tied (p>={alpha_corr:.3f} Bonferroni): {sorted(tied)}; "
                  f"p-values: " + ", ".join(
                      f"{t}={pvals[t]:.2g}" for t, _ in vals[1:]))

    def fmt(val, tool, ds, key, spec):
        txt = format(val, spec)
        best, second = rank_by_ds[ds].get(key, (set(), set()))
        if tool in best:
            return f"\\textbf{{{txt}}}"
        if tool in second:
            return f"\\underline{{{txt}}}"
        return txt

    def fmt_pm(mean_val, sd_val, tool, ds, key, spec):
        """Format a cell as ``mean $\\pm$ sd''.  Ranking (bold / underline)
        applies to the cell as a whole, based on the mean ranking."""
        m_txt = format(mean_val, spec)
        s_txt = format(sd_val, spec) if (sd_val is not None
                                          and not np.isnan(sd_val)) else "--"
        cell = f"{m_txt}\\,$\\pm$\\,{s_txt}"
        best, second = rank_by_ds[ds].get(key, (set(), set()))
        if tool in best:
            return f"\\textbf{{{cell}}}"
        if tool in second:
            return f"\\underline{{{cell}}}"
        return cell

    def fmt_n(n):
        # LaTeX-friendly thousands separator
        return f"{n:,}".replace(",", "{,}")

    # Emit only the data rows so the paper's main.tex can own \toprule,
    # \cmidrule, column headers, \midrule, and \bottomrule. Also: the LAST
    # row is emitted WITHOUT a trailing "\\" — main.tex provides it after
    # the \input. A trailing "\\" on the last line of an \input-ed file
    # triggers "Misplaced \noalign" when \bottomrule follows, because TeX
    # inserts something between \\ and \bottomrule across the \input boundary.
    def build_rows(datasets: list[str]) -> str:
        """Build the row body for a given subset of datasets, with italic
        group labels and a \\midrule both between blocks AND beneath each
        label. Avoids \\multicolumn (some setups error "Misplaced \\omit"
        right after \\midrule)."""
        empty_cells = " & " * 5
        gen_label = f"\\textit{{Generative}}{empty_cells}"
        msa_label = f"\\textit{{MSA perturbation}}{empty_cells}"

        # Each entry is ("row", text) — gets a trailing "\\\\" — or
        # ("rule", text) — emitted verbatim on its own line.
        # Tight version of \midrule (0pt vertical padding above/below) used
        # immediately under the italic group labels so they sit snug against
        # the data rows below them.
        tight_rule = "\\specialrule{\\lightrulewidth}{0pt}{0pt}"
        entries: list[tuple[str, str]] = [
            ("row", gen_label),
            ("rule", tight_rule),
        ]
        for tool in TOOLS:
            print(f"{TOOL_DISPLAY.get(tool,tool):<12s}", end="")
            row_parts = [TOOL_DISPLAY.get(tool, tool)]
            for ds in datasets:
                m = metrics_by_tool_ds[tool][ds]
                if m:
                    print(f"  {m['n']:>6d} "
                          f"{m['dr']*100:>5.1f}±{m['dr_sd']*100:>5.1f} "
                          f"{m['cr']*100:>5.1f}±{m['cr_sd']*100:>5.1f} "
                          f"{m['eu']:>5.3f}±{m['eu_sd']:>5.3f} "
                          f"{m['emd']:>5.3f}±{m['emd_sd']:>5.3f}", end="")
                    row_parts.append(fmt_n(m['n']))
                    row_parts.append(fmt_pm(m['dr']*100,  m['dr_sd']*100,  tool, ds, "dr",  ".1f"))
                    # FRR is a binary indicator; its SD ≈ √(p(1-p)) is
                    # uninformative, so report just the mean.
                    row_parts.append(fmt(m['cr']*100, tool, ds, "cr", ".1f"))
                    # TM-bar reported as a fraction (0–1), three decimals.
                    row_parts.append(fmt_pm(m['eu'],       m['eu_sd'],      tool, ds, "eu",  ".3f"))
                    row_parts.append(fmt_pm(m['emd'],      m['emd_sd'],     tool, ds, "emd", ".3f"))
                else:
                    print(f"  {'--':>6s} {'--':>6s} {'--':>6s} {'--':>6s} {'--':>6s}", end="")
                    row_parts.extend(["--"] * 5)
            print()
            entries.append(("row", " & ".join(row_parts)))
            if tool == GENERATIVE_TOOLS[-1]:
                entries.append(("rule", "\\midrule"))   # full-spacing separator between groups
                entries.append(("row", msa_label))
                entries.append(("rule", tight_rule))    # tight rule under MSA label

        # Last data row stays bare so main.tex closes it after \input.
        last_row_idx = max(i for i, (k, _) in enumerate(entries) if k == "row")
        out_lines: list[str] = []
        for i, (kind, txt) in enumerate(entries):
            if kind == "rule":
                out_lines.append(txt)
            else:
                out_lines.append(txt if i == last_row_idx else txt + " \\\\")
        return "\n".join(out_lines) + "%\n"

    main_dataset = "UNICORNE_BENCH"
    appendix_datasets = [d for d in args.datasets if d != main_dataset]

    main_out = args.output if args.output else (
        PROJECT_ROOT / "figures" / "tab_overall_performance.tex"
    )
    main_out.parent.mkdir(parents=True, exist_ok=True)

    if main_dataset in args.datasets:
        print("\n--- main table (UNICORNE_BENCH) ---")
        main_out.write_text(build_rows([main_dataset]))
        print(f"Wrote {main_out}")

    if appendix_datasets:
        appendix_out = main_out.parent / (
            main_out.stem + "_appendix" + main_out.suffix
        )
        print(f"\n--- appendix table ({', '.join(appendix_datasets)}) ---")
        appendix_out.write_text(build_rows(appendix_datasets))
        print(f"Wrote {appendix_out}")


if __name__ == "__main__": main()
