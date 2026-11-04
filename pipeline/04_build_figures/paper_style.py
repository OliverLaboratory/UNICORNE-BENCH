"""
Shared figure style: a serif font that matches the paper's Times body text,
and text sized for the width each figure is printed at in the paper.

Importing this module sets the font. Call finalize(fig, print_width_in) just
before saving: it rescales every text element so that, once LaTeX scales the
figure to print_width_in inches, the median text is about 8 pt and nothing is
smaller than 7 pt or larger than 10 pt, then re-runs the layout.
"""
from __future__ import annotations

import matplotlib
import numpy as np
from matplotlib.layout_engine import ConstrainedLayoutEngine
from matplotlib.text import Text

TEXT_WIDTH_IN = 5.5   # NeurIPS \textwidth (397.5 pt)

# Shared margins of the three side-by-side panels of the recall figure
# (diversity, ensemble size, motion amplitude), so their plot areas align.
FIG3_MARGINS = dict(left=0.25, right=0.97, bottom=0.42, top=0.97)
# Same for the sample-efficiency pair (Recall@k and its gain per doubling).
AT_K_MARGINS = dict(left=0.18, right=0.97, bottom=0.25, top=0.97)

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def finalize(fig, print_width_in: float, median_pt: float = 8.0,
             min_pt: float = 7.0, max_pt: float = 10.0, relayout: bool = True) -> None:
    """Resize all text for a figure printed print_width_in inches wide."""
    scale = print_width_in / fig.get_figwidth()      # printed pt per figure pt
    fig.canvas.draw()                                 # materialize tick labels
    texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
    if not texts:
        return
    printed = np.array([t.get_fontsize() * scale for t in texts])
    k = median_pt / float(np.median(printed))

    def resized(size_pt: float) -> float:
        return float(np.clip(size_pt * scale * k, min_pt, max_pt)) / scale

    # Tick labels are rebuilt at draw time from the axis tick parameters, so
    # their size is set there (recorded before any text is resized).
    tick_sizes = []
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            for which, ticks in (("major", axis.get_major_ticks()), ("minor", axis.get_minor_ticks())):
                if ticks:
                    tick_sizes.append((axis, which, ticks[0].label1.get_fontsize()))
    for t in texts:
        t.set_fontsize(resized(t.get_fontsize()))
    for axis, which, size in tick_sizes:
        axis.set_tick_params(which=which, labelsize=resized(size))
    if relayout and not isinstance(fig.get_layout_engine(), ConstrainedLayoutEngine):
        try:
            fig.tight_layout()
        except Exception:
            pass
