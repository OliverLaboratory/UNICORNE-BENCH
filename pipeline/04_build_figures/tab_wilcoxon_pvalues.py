#!/usr/bin/env python3
"""
Pairwise paired-Wilcoxon p-values for each leaderboard metric.

For each metric (Recall_{0.9}, FRR_{0.9}, TM-bar, EMD), we compute a
NxN matrix of paired-Wilcoxon-signed-rank p-values across the
evaluated tools, on the intersection of evaluable ensembles for each
pair.  Output: a single LaTeX file with four sub-tables (one per
metric) ready to be `\\input{}`-ed into the appendix or compiled on
its own.

Output:
  figures/tab_wilcoxon_pvalues.tex
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET = "UNICORNE_BENCH"
RESULTS = PROJECT_ROOT / "results" / DATASET

# AFsample3 is the afsample3_biorxiv run (masking sweep from the AFsample3
# paper); see tab_overall_performance.py.
TOOLS = ["bioemu", "esmflow_pdb", "esmdiff", "alphaflow_pdb",
         "cfrandom", "afsample2", "afsample3_biorxiv", "speach_af"]
TOOL_DISPLAY = {
    "bioemu":        r"\bioemu",
    "esmflow_pdb":   r"\esmflow",
    "esmdiff":       r"\esmdiff",
    "alphaflow_pdb": r"\alphaflow",
    "cfrandom":      r"\cfrandom",
    "afsample2":     r"\afsampletwo",
    "afsample3_biorxiv": r"\afsamplethree",
    "speach_af":     r"\speachaf",
}
METRICS = [
    ("dr",  r"$\text{Recall}_{0.9}$"),
    ("cr",  r"$\text{FRR}_{0.9}$"),
    ("eu",  r"$\overline{\text{TM}}$"),
    ("emd", r"EMD"),
]


def load_filter():
    fp = PROJECT_ROOT / "data" / "inputs" / DATASET / "filtered_ensembles_v2.csv"
    if not fp.exists():
        return None
    ids = set()
    with open(fp) as f:
        for r in csv.DictReader(f):
            eid = r.get("ensemble_ID", "").strip()
            if eid:
                ids.add(eid)
    bl = PROJECT_ROOT / "data" / "inputs" / DATASET / "fold_quality_blacklist.csv"
    if bl.exists():
        for r in csv.DictReader(open(bl)):
            ids.discard((r.get("uniprot") or "").strip())
    return ids


def load_per_ensemble(tool: str, key: str,
                      keep: set[str] | None) -> dict[str, float]:
    out: dict[str, float] = {}
    fp = RESULTS / tool / "results.csv"
    if not fp.exists():
        return out
    with open(fp) as f:
        for row in csv.DictReader(f):
            eid = row.get("ensemble_ID", "").strip()
            if keep is not None and eid not in keep:
                continue
            v = row.get(key, "")
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if not np.isnan(fv):
                out[eid] = fv
    return out


def fmt_p(p: float) -> str:
    if np.isnan(p):
        return "--"
    if p < 1e-50:
        return "$<\\!10^{-50}$"
    if p < 1e-3:
        # scientific notation, e.g. 4.3 \times 10^{-7}
        exp = int(np.floor(np.log10(p)))
        coef = p / (10 ** exp)
        return f"${coef:.1f}{{\\times}}10^{{{exp}}}$"
    if p < 0.01:
        return f"{p:.3f}"
    return f"{p:.2f}"


def build_metric_table(metric_key: str, metric_label: str,
                       data: dict[str, dict[str, float]]) -> str:
    """Return one tabular for the given metric: row=tool, col=tool, entry=p."""
    n = len(TOOLS)
    P = np.full((n, n), np.nan)
    N = np.zeros((n, n), dtype=int)
    for i, ti in enumerate(TOOLS):
        for j, tj in enumerate(TOOLS):
            if i == j:
                continue
            di, dj = data[ti], data[tj]
            common = set(di) & set(dj)
            pairs = [(di[e], dj[e]) for e in common
                     if not np.isnan(di[e]) and not np.isnan(dj[e])]
            if len(pairs) < 10:
                continue
            a = np.array([p[0] for p in pairs])
            b = np.array([p[1] for p in pairs])
            if np.allclose(a, b):
                P[i, j] = 1.0
            else:
                try:
                    P[i, j] = float(
                        wilcoxon(a, b, zero_method="wilcox").pvalue)
                except ValueError:
                    P[i, j] = float("nan")
            N[i, j] = len(pairs)

    # Header row: empty corner + tool labels.
    header = ["", *(TOOL_DISPLAY[t] for t in TOOLS)]
    lines: list[str] = []
    lines.append(r"\begin{tabular}{l" + "c" * n + "}")
    lines.append(r"\toprule")
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")
    for i, ti in enumerate(TOOLS):
        cells = [TOOL_DISPLAY[ti]]
        for j, tj in enumerate(TOOLS):
            if i == j:
                cells.append(r"\textemdash")
            else:
                cells.append(fmt_p(P[i, j]))
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    body = "\n".join(lines)

    return (
        "\\begin{table}[h]\n"
        "\\centering\n"
        f"\\caption{{Paired Wilcoxon signed-rank $p$-values for {metric_label} on \\unicornebench.\n"
        "Each cell is the $p$-value for the null hypothesis that the row tool's per-ensemble metric distribution\n"
        "is the same as the column tool's, computed on the intersection of evaluable ensembles.\n"
        "$p$-values are uncorrected; the leaderboard in Table~\\ref{tab:overall} applies\n"
        f"Bonferroni correction over the {n - 1} pairwise tests per metric ($\\alpha/{n - 1} \\approx {0.05 / (n - 1):.3f}$).}}\n"
        f"\\label{{tab:wilcoxon_{metric_key}}}\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        + body + "}\n"
        "\\end{table}\n"
    )


def main():
    keep = load_filter()
    print(f"filter: {len(keep) if keep else 0} ensembles")

    out_path = PROJECT_ROOT / "figures" / "tab_wilcoxon_pvalues.tex"
    sections: list[str] = []
    for key, label in METRICS:
        col = {"dr": "dr", "cr": "cr", "eu": "eu", "emd": "emd"}[key]
        data = {t: load_per_ensemble(t, col, keep) for t in TOOLS}
        ns = {t: len(data[t]) for t in TOOLS}
        print(f"  {key}: per-tool n = {ns}")
        sections.append(build_metric_table(key, label, data))

    body = "\n\n".join(sections)
    out_path.write_text(body)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
