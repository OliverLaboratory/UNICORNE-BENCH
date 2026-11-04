#!/usr/bin/env python3
"""
Figure 1 (curation diagram): redraw the labels of the hand-drawn diagram in
the paper's serif font at a readable size.

The artwork (external/curation_diagram/curation_diagram_unlabeled_source.png,
the Inkscape export) is kept; each label box listed in label_boxes.json (pixel
box, fill and edge color, text) and each ensemble label is cleared and redrawn
at the same place. The small ensemble thumbnails are moved outward so the
larger labels fit.

Output:
  figures/curation_diagram.{png,pdf}

Usage:
  python pipeline/04_build_figures/fig_curation_diagram.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_style  # noqa: E402,F401  (serif font)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC = PROJECT_ROOT / "external" / "curation_diagram"
OUT = PROJECT_ROOT / "figures" / "curation_diagram.png"
BOX_PT, LABEL_PT = 6.0, 6.0
img = np.asarray(Image.open(SRC / "curation_diagram_unlabeled_source.png").convert("RGB")).copy()
H, W = img.shape[:2]
boxes = json.loads((SRC / "label_boxes.json").read_text())
# Ensemble thumbnails moved outward to make room for the larger labels:
# (search x0, x1, y0, y1, dx). The thumbnail is the ink inside the search box.
THUMBS = {
    "aA": (140, 365, 880, 1110, -40), "aB": (995, 1205, 880, 1110, 40),
    "dA": (130, 355, 2140, 2420, -50), "dB": (955, 1205, 2140, 2420, 50),
    "eA": (1480, 1735, 2140, 2460, -80), "eB": (2295, 2565, 2140, 2460, 80),
    "fM": (2780, 3090, 2140, 2460, -70), "fO": (3660, 4010, 2140, 2460, 130),
    "bA": (2105, 2320, 880, 1110, 30), "cA": (3010, 3245, 840, 1020, 0),
}
# Old ensemble labels to clear (x0, y0, x1, y1)
CLEAR = [(370, 940, 612, 1060), (745, 940, 978, 1060), (1852, 940, 2100, 1060),
         (3271, 940, 3520, 1060), (360, 2210, 602, 2335), (727, 2198, 962, 2318),
         (1734, 2214, 1981, 2335), (2050, 2214, 2297, 2335), (3090, 2202, 3364, 2376),
         (3392, 2202, 3660, 2376)]
for b in boxes:
    x0, y0, x1, y1 = b["bbox"]; img[y0 - 3:y1 + 4, x0 - 3:x1 + 4] = 255
for x0, y0, x1, y1 in CLEAR:
    img[y0:y1, x0:x1] = 255
thumb_box = {}
for key, (x0, x1, y0, y1, dx) in THUMBS.items():
    ink = np.nonzero(img[y0:y1, x0:x1].min(axis=2) < 245)
    by0, by1 = y0 + ink[0].min(), y0 + ink[0].max() + 1
    bx0, bx1 = x0 + ink[1].min(), x0 + ink[1].max() + 1
    block = img[by0:by1, bx0:bx1].copy()
    img[by0:by1, bx0:bx1] = 255
    img[by0:by1, bx0 + dx:bx1 + dx] = np.minimum(img[by0:by1, bx0 + dx:bx1 + dx], block)
    thumb_box[key] = (bx0 + dx, by0, bx1 + dx, by1)
GAP = 30
def right_of(k): return thumb_box[k][2] + GAP
def left_of(k): return thumb_box[k][0] - GAP
def mid_y(k): return (thumb_box[k][1] + thumb_box[k][3]) / 2
# (anchor x, anchor y, ha, lines)
LABELS = [
    (right_of("aA"), 1001, "left", ["Ensemble A", "1 member"]),
    (left_of("aB"), 1001, "right", ["Ensemble B", "1 member"]),
    (left_of("bA"), 1001, "right", ["Ensemble A", "2 members"]),
    (right_of("cA"), 1001, "left", ["Ensemble A", "1 member"]),
    (right_of("dA"), 2273, "left", ["Ensemble A", "1 member"]),
    (left_of("dB"), 2259, "right", ["Ensemble B", "1 member"]),
    (right_of("eA"), 2275, "left", ["Ensemble A", "2 members"]),
    (left_of("eB"), 2275, "right", ["Ensemble B", "2 members"]),
    (right_of("fM"), 2290, "left", ["Monomeric", "Ensemble A", "4 members"]),
    (left_of("fO"), 2290, "right", ["Oligomeric", "Ensemble A", "1 member"]),
]

fig = plt.figure(figsize=(5.5, 5.5 * H / W))
ax = fig.add_axes((0, 0, 1, 1)); ax.imshow(img, interpolation="bilinear"); ax.axis("off")
ax.set_xlim(0, W); ax.set_ylim(H, 0)
artists = []
for b in boxes:
    x0, y0, x1, y1 = b["bbox"]
    lines = b["lines"]
    if lines[0].startswith("UniProt ID"):
        uid = lines[-1].split()[-1]
        text = "UniProt\n$\\mathbf{" + uid + "}$" if len(lines) == 2 else "UniProt $\\mathbf{" + uid + "}$"
    elif lines[0].startswith("Entity"):
        text = "Entity $\\mathbf{" + lines[0].split()[-1] + "}$"
    else:
        text = "$\\mathbf{Biological\\ assembly\\ 1}$"
    t = ax.text((x0 + x1) / 2, (y0 + y1) / 2, text, ha="center", va="center", fontsize=BOX_PT,
                linespacing=1.0, multialignment="center",
                bbox=dict(boxstyle="square,pad=0.2", fc=np.array(b["fill"]) / 255,
                          ec=np.array(b["edge"]) / 255, lw=0.8))
    artists.append(("box", t))
for ax_x, ax_y, ha, lines in LABELS:
    parts = []
    for ln in lines:
        if ln in ("Monomeric", "Oligomeric") or ln.startswith("Ensemble"):
            parts.append("$\\mathbf{" + ln.replace(" ", "\\ ") + "}$")
        else:
            parts.append("$\\mathit{" + ln.replace(" ", "\\ ") + "}$")
    t = ax.text(ax_x, ax_y, "\n".join(parts), ha=ha, va="center", fontsize=LABEL_PT,
                linespacing=1.05, multialignment=ha if ha != "center" else "center")
    artists.append(("label", t))

fig.canvas.draw()
r = fig.canvas.get_renderer()
ext = [(k, t.get_text()[:24], t.get_window_extent(r) if k == "label" else t.get_bbox_patch().get_window_extent(r)) for k, t in artists]
bad = [(a[1], b[1]) for i, a in enumerate(ext) for b in ext[i + 1:] if a[2].overlaps(b[2])]
print("label/box overlaps:", bad or "none")
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT.with_suffix(".pdf"), dpi=600)
fig.savefig(OUT, dpi=300)
print(f"Saved {OUT}")
