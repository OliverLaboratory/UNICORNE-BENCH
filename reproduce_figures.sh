#!/bin/bash
# Rebuild every benchmark figure and table in the paper from the data shipped
# in this repository (results/ + data/). Runs on a CPU in a few minutes.
# Outputs go to figures/.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p figures
F=pipeline/04_build_figures
run() { echo "+ $*"; python "$@"; }

# Section 4.1: TM threshold from natural motions; OOD60 / DOMAINMOTION tables
run $F/fig_tm_vs_rmsd_scatter.py --output figures/tm_vs_rmsd_scatter.png
run $F/tab_ood60_domainmotion.py

# Section 4.2: overall performance and significance
run $F/tab_overall_performance.py --datasets UNICORNE_BENCH
run $F/tab_wilcoxon_pvalues.py

# Section 4.3: failure modes
run $F/fig_dr_boxplots.py --metric dr --y-lim 0 0.5
run $F/fig_cofail_mechanism.py

# Section 4.4: mode collapse and sample efficiency
run $F/fig_mode_count.py
run $F/fig_dr_at_k.py --metric dr
run $F/fig_at_k_gain.py --metric dr --doubling --absolute

# Section 4.5: principal angle examples
run $F/fig_pa_heatmaps.py

# Section 4.6: gold-standard panel and AF2-OOD
run $F/fig_unicorne_gold_standard.py --output figures/unicorne_gold_standard.png
run $F/fig_unicorne_gold_standard.py --metric tm_bar --ood-only --output figures/unicorne_gold_standard_ood.png
run $F/fig_ood_boxplot.py
echo "+ tab_unicorne_gold_standard.py"
python $F/tab_unicorne_gold_standard.py > figures/tab_unicorne_gold_standard.tex
python $F/tab_unicorne_gold_standard.py --tools bioemu afsample2 > figures/tab_unicorne_gold_standard_reduced.tex

# Appendix: dataset characterization and inference runtime
run $F/fig_dataset_characterization.py
run $F/plot_runtime_benchmark.py

echo "All figures and tables written to figures/"
