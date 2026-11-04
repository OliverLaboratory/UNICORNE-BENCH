#!/usr/bin/env python3
"""
Figure: best BioEMU and AFsample2 predictions on Q5U925 (dehydratase
activator), superposed on its two GT states (paper Fig. 7).

Each panel shows a GT state as a rainbow cartoon with the best prediction of
each tool (highest TM-score normalized by GT length) superposed by TM-align
and drawn translucent in the tool's color.

Two steps:
  --render  PyMOL renders from the full output (GT structures, predictions and
            results/UNICORNE_BENCH/<tool>/tmalign_raw/Q5U925.json); writes
            results/case_study/{<GT>.png, meta.json}. Run with PyMOL's Python:
              pymol -cq pipeline/04_build_figures/fig_case_study.py -- --render
  (default) compose the figure from those renders, which ship with the repo.

Output:
  figures/case_study_unicorne.{png,pdf}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA = PROJECT_ROOT / "results" / "case_study"
EID = "Q5U925"
STATES = [("4EHT_A-aligned.cif", "Dominant state (4EHT)"),
          ("4EIA_A-aligned.cif", "Alternate state (4EIA)")]
TOOLS = {"bioemu": ("BioEMU", "#C0392B"), "afsample2": ("AFsample2", "#2980B9")}


def render(src_root: Path) -> None:
    """PyMOL renders of each GT with the tools' best predictions superposed."""
    from pymol import cmd

    meta = {}
    for gt, _ in STATES:
        cmd.reinitialize()
        for key, value in (("ray_opaque_background", 0), ("cartoon_fancy_helices", 1),
                           ("cartoon_smooth_loops", 1), ("ray_shadow", 0),
                           ("antialias", 2), ("specular", 0.2)):
            cmd.set(key, value)
        cmd.load(str(src_root / "data/inputs/UNICORNE_BENCH/input_pdbs" / EID / gt), "gt")
        cmd.hide("everything")
        cmd.show("cartoon", "gt")
        cmd.spectrum("count", "rainbow", "gt and name CA")
        meta[gt] = {}
        for tool, (_, hex_color) in TOOLS.items():
            d = json.loads((src_root / f"results/UNICORNE_BENCH/{tool}/tmalign_raw/{EID}.json").read_text())
            pairs = [p for p in d["pairs"] if Path(p["gt"]).name == gt and p.get("tm_norm_chain1") is not None]
            p = max(pairs, key=lambda q: q["tm_norm_chain1"])
            cmd.load(p["pred"], tool)
            # TM-align's rotation maps the GT onto the prediction (x_pred = u x_gt + t);
            # bring the prediction into the GT frame: u^T (x_pred - t).
            u, t = np.array(p["rotation"]["u"]), np.array(p["rotation"]["t"])
            cmd.load_coords(((cmd.get_coords(tool) - t) @ u).tolist(), tool)
            cmd.set_color(f"{tool}_c", [int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5)])
            cmd.show("cartoon", tool)
            cmd.color(f"{tool}_c", tool)
            cmd.set("cartoon_transparency", 0.55, tool)
            meta[gt][tool] = {"pred": Path(p["pred"]).name, "tm": p["tm_norm_chain1"]}
        cmd.orient("gt")
        cmd.zoom("gt", 2)
        cmd.ray(1800, 1350)
        DATA.mkdir(parents=True, exist_ok=True)
        cmd.png(str(DATA / f"{gt.split('_')[0]}.png"), dpi=300)
    (DATA / "meta.json").write_text(json.dumps(meta, indent=1))


def crop(img: np.ndarray, pad: int = 12) -> np.ndarray:
    """Trim fully transparent margins."""
    ys, xs = np.nonzero(img[..., 3] > 0)
    return img[max(ys.min() - pad, 0):ys.max() + pad, max(xs.min() - pad, 0):xs.max() + pad]


def compose(output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import paper_style  # noqa: F401  (serif font)

    meta = json.loads((DATA / "meta.json").read_text())
    imgs = [crop(plt.imread(DATA / f"{gt.split('_')[0]}.png")) for gt, _ in STATES]
    # Drawn at its printed size (full text width); the two states side by side.
    width, gap, title_h, legend_h = 5.5, 0.1, 0.18, 0.22
    panel_w = (width - gap) / len(imgs)
    img_h = max(panel_w * im.shape[0] / im.shape[1] for im in imgs)
    fig_h = title_h + img_h + legend_h
    fig = plt.figure(figsize=(width, fig_h))
    for k, ((gt, title), im) in enumerate(zip(STATES, imgs)):
        x0 = k * (panel_w + gap) / width
        cx = x0 + panel_w / width / 2
        fig.text(cx, 1 - 0.02 / fig_h, title, ha="center", va="top", fontsize=8, fontweight="bold")
        ax = fig.add_axes((x0, legend_h / fig_h, panel_w / width, img_h / fig_h))
        ax.imshow(im, interpolation="bilinear")
        ax.axis("off")
        handles = [Patch(facecolor=color, alpha=0.75, label=f"{name}, TM {meta[gt][tool]['tm']:.2f}")
                   for tool, (name, color) in TOOLS.items()]
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(cx, 0.0),
                   ncol=2, frameon=False, fontsize=7, handlelength=1.2, handletextpad=0.4,
                   columnspacing=1.2)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    fig.savefig(output.with_suffix(".pdf"), dpi=600)   # dpi sets the embedded renders' resolution
    plt.close(fig)
    print(f"Saved {output}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", action="store_true", help="Re-render with PyMOL from the full output")
    ap.add_argument("--src-root", type=Path, default=PROJECT_ROOT,
                    help="Checkout holding the predictions and tmalign_raw JSON (for --render)")
    ap.add_argument("--output", type=Path, default=PROJECT_ROOT / "figures" / "case_study_unicorne.png")
    args = ap.parse_args()
    if args.render:
        render(args.src_root)
    else:
        compose(args.output)


if __name__ == "__main__":
    main()
