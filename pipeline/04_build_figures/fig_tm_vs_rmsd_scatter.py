#!/usr/bin/env python3
"""
Figure: TM-score vs RMSD scatter for OOD60 + DOMAINMOTION two-state benchmark.

Shows best-by-TM prediction per GT conformation (y-axis = TMalign local RMSD
of that prediction vs GT).  Green dashed line = MIN *global* (Kabsch,
whole-chain) RMSD between the two GT conformations of each ensemble — the
smallest natural motion amplitude observed in the benchmark, adopted as a
conservative allowable-error tolerance for predictions.  Red vertical
line = lowest TM above which every prediction's local RMSD is below that
tolerance.  Green shaded region = "safe zone" (pred local RMSD < min
observed natural motion).

Input:
  results/{OOD60,DOMAINMOTION}/{TOOL}/matrices_tmalign/{eid}/tm_xy.npy, rmsd_xy_ab.npy
  data/inputs/{OOD60,DOMAINMOTION}/input_dataset.csv
  data/inputs/{OOD60,DOMAINMOTION}/input_pdbs/<eid>/*.pdb  (for global RMSD)

Output:
  figures/tm_vs_rmsd_scatter.png

Usage:
  python pipeline/04_build_figures/fig_tm_vs_rmsd_scatter.py
"""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import paper_style  # noqa: E402  (serif font; text sized for the paper)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASETS = ["OOD60", "DOMAINMOTION"]
TOOLS = ["bioemu", "esmflow_pdb", "alphaflow_pdb", "esmdiff", "cfrandom"]


def _parse_ca(pdb_path: Path) -> dict[tuple[str, int, str], np.ndarray]:
    """Return {(chain, resseq, icode): xyz} for all CA atoms in a PDB file."""
    out: dict[tuple[str, int, str], np.ndarray] = {}
    try:
        with open(pdb_path) as f:
            for line in f:
                if not line.startswith("ATOM"):
                    continue
                if line[12:16].strip() != "CA":
                    continue
                chain = line[21].strip() or "A"
                try:
                    resseq = int(line[22:26])
                except ValueError:
                    continue
                icode = line[26].strip()
                key = (chain, resseq, icode)
                if key not in out:
                    out[key] = np.array([
                        float(line[30:38]), float(line[38:46]), float(line[46:54])
                    ])
    except OSError:
        pass
    return out


def _kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    """Global CA RMSD after optimal Kabsch superposition."""
    P_c = P - P.mean(axis=0)
    Q_c = Q - Q.mean(axis=0)
    H = P_c.T @ Q_c
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    P_rot = P_c @ R.T
    diff = P_rot - Q_c
    return float(np.sqrt((diff ** 2).sum() / P.shape[0]))


def _global_gt_rmsd(pdb_a: Path, pdb_b: Path) -> float | None:
    """Global (Kabsch, whole-chain) RMSD between two GT conformations.
    Match residues by (chain, resseq, icode); fall back to (resseq, icode)
    within each file's first chain if direct intersection is tiny.
    Returns None if fewer than 3 CAs can be matched."""
    a = _parse_ca(pdb_a)
    b = _parse_ca(pdb_b)
    if not a or not b:
        return None
    common = sorted(set(a) & set(b))
    if len(common) < 3:
        chain_a = min(set(k[0] for k in a))
        chain_b = min(set(k[0] for k in b))
        a_rm = {(k[1], k[2]): v for k, v in a.items() if k[0] == chain_a}
        b_rm = {(k[1], k[2]): v for k, v in b.items() if k[0] == chain_b}
        shared = sorted(set(a_rm) & set(b_rm))
        if len(shared) < 3:
            return None
        P = np.stack([a_rm[k] for k in shared])
        Q = np.stack([b_rm[k] for k in shared])
    else:
        P = np.stack([a[k] for k in common])
        Q = np.stack([b[k] for k in common])
    return _kabsch_rmsd(P, Q)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "figures" / "tm_vs_rmsd_scatter.png")
    args = parser.parse_args()

    tms, rmsds, lengths, dsets = [], [], [], []
    gt_tms, gt_rmsds = [], []

    for ds in DATASETS:
        seq_lens = {}
        with open(PROJECT_ROOT / "data" / "inputs" / ds / "input_dataset.csv") as f:
            for row in csv.DictReader(f):
                eid = (row.get("ensemble_ID") or row.get("uniprot", "")).strip()
                seq = (row.get("ref_sequence") or row.get("sequence", "")).strip()
                seq_lens[eid] = len(seq)

        # Try the new tmalign_raw JSON store first; fall back to legacy npy.
        from sys import path as _sys_path
        _sys_path.insert(0, str(PROJECT_ROOT / "pipeline" / "03_compute_results"))
        from tmalign_io import load_xy_pair, list_available_eids

        for tool in TOOLS:
            for eid in list_available_eids(ds, tool):
                sl = seq_lens.get(eid, 0)
                if sl == 0: continue
                out = load_xy_pair(ds, tool, eid)
                if out is None: continue
                tm, _tm2, rmsd = out  # tm normalized by chain1 (GT length, default)
                if tm is None or rmsd is None: continue
                tm_filled = np.where(np.isnan(tm), 0.0, tm)
                rmsd_filled = np.where(np.isnan(rmsd), np.nan, rmsd)
                for gi in range(tm_filled.shape[0]):
                    bi = int(tm_filled[gi].argmax())
                    if np.isnan(rmsd_filled[gi, bi]): continue
                    tms.append(float(tm_filled[gi, bi]))
                    rmsds.append(float(rmsd_filled[gi, bi]))
                    lengths.append(sl)
                    dsets.append(ds)

        # GT pairwise global (Kabsch, whole-chain) RMSD — natural motion scale
        pdbs_root = PROJECT_ROOT / "data" / "inputs" / ds / "input_pdbs"
        gt_dir = PROJECT_ROOT / "results" / ds / "gt_matrices_tmalign"
        if pdbs_root.is_dir():
            for eid_subdir in sorted(pdbs_root.iterdir()):
                if not eid_subdir.is_dir():
                    continue
                pdbs = sorted(eid_subdir.glob("*.pdb"))
                # Cached TM (symmetric) for the (i,j) pair, if available
                tm_mat = None
                if gt_dir.is_dir():
                    tm_f = gt_dir / eid_subdir.name / "tm_xx.npy"
                    if tm_f.exists():
                        tm_mat = np.load(tm_f)
                for i in range(len(pdbs)):
                    for j in range(i + 1, len(pdbs)):
                        rmsd_g = _global_gt_rmsd(pdbs[i], pdbs[j])
                        if rmsd_g is None:
                            continue
                        gt_rmsds.append(rmsd_g)
                        if tm_mat is not None and tm_mat.shape[0] > max(i, j):
                            gt_tms.append(float(tm_mat[i, j]))

    tms, rmsds = np.array(tms), np.array(rmsds)
    avg_gt_rmsd = float(np.min(gt_rmsds)) if gt_rmsds else 3.42

    # Green horizontal line = MIN *global* (whole-chain Kabsch) GT-GT RMSD,
    # representing the smallest natural amplitude of motion observed between
    # paired GT conformations in the benchmark.  We treat this conservative
    # value as the allowable prediction error tolerance: any prediction whose
    # local (TMalign) RMSD to GT is below this scale is "within the tightest
    # natural motion budget" we observe.
    # Red vertical line = lowest TM above which every prediction (local RMSD)
    # satisfies this bound.
    sorted_idx = np.argsort(tms)[::-1]
    threshold_tm = float(tms.max()) if len(tms) else 1.0
    for idx in sorted_idx:
        if rmsds[idx] > avg_gt_rmsd:
            break
        threshold_tm = float(tms[idx])

    # Drawn at its printed size (2.7 in, wrapped beside the text). Title, n and
    # the marker-size key are in the figure caption.
    fig, ax = plt.subplots(1, 1, figsize=(2.70, 1.95))
    y_top = 9.0   # headroom for the legend above the data
    ax.axhspan(0, avg_gt_rmsd, xmin=(threshold_tm - 0.25) / (1.02 - 0.25),
               xmax=1.0, color="#2CA02C", alpha=0.08)
    DS_COLORS = {"OOD60": "#4878CF", "DOMAINMOTION": "#EE854A"}
    dsets_arr = np.array(dsets)
    sizes = np.array([max(3, s * 0.03) for s in lengths])   # marker area ~ sequence length
    for ds, color in DS_COLORS.items():
        mask = dsets_arr == ds
        if not mask.any():
            continue
        ax.scatter(tms[mask], rmsds[mask], s=sizes[mask], c=color, alpha=0.55,
                   edgecolors="white", linewidth=0.2, label=ds)
    ax.axhline(avg_gt_rmsd, color="#2CA02C", linestyle="--", alpha=0.8, linewidth=0.9)
    ax.text(0.27, avg_gt_rmsd + 0.35, f"min GT RMSD = {avg_gt_rmsd:.1f} Å",
            color="#2CA02C", fontsize=7)
    ax.axvline(threshold_tm, color="#D65F5F", linestyle="-", alpha=0.9, linewidth=0.9)
    ax.text(threshold_tm + 0.01, 6.3, f"TM =\n{threshold_tm:.2f}", color="#D65F5F",
            fontsize=7, va="center", ha="left")
    ax.set_xlabel("TM-score of best prediction")
    ax.set_ylabel("C$\\alpha$ RMSD (Å)")
    ax.legend(loc="upper left", ncol=2, markerscale=1.5, handletextpad=0.1,
              columnspacing=0.8, borderpad=0.3, framealpha=0.9)
    ax.set_xlim(0.25, 1.02); ax.set_ylim(0, y_top)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.15)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    paper_style.finalize(fig, 2.70)
    fig.savefig(args.output, dpi=300)
    fig.savefig(str(args.output).rsplit(".", 1)[0] + ".pdf")
    plt.close()
    print(f"Saved {args.output} ({len(tms)} points, threshold TM={threshold_tm:.3f})")

if __name__ == "__main__": main()
