# UNICORNE-BENCH

Code and data to reproduce the benchmark results in

> **Benchmarking sequence-to-ensemble predictors on UNICORNEdb, a UniProt-grouped database of PDB-derived conformational ensembles.**
> W. F. Vanderbilt-Fried, L. Butskhrikidze, B. P. Brown, H. S. Mchaourab, C. Oliver. *NeurIPS 2026, Evaluations & Datasets Track.*

UNICORNE-BENCH-1 is a set of 839 multi-conformation proteins (829 after a fold-quality review), each a single UniProt entry with at least two experimentally resolved conformations. Eight sequence-to-ensemble predictors are scored on it: four generative models (BioEMU, ESMFlow, ESMDiff, AlphaFlow) and four MSA-perturbation methods (CFrandom, AFsample2, AFsample3, SPEACH_AF). The `-1` marks the first release of the benchmark; later releases will be numbered as the PDB grows. In paths and command-line arguments the dataset is called `UNICORNE_BENCH`.

The benchmark data, including the ground-truth structures, is archived at <https://doi.org/10.5281/zenodo.23928693>. The database itself is browsable at <https://unicornedb.org>.

## Reproduce the paper's figures and tables

Everything in the paper's results section rebuilds from this repository in a few minutes on a CPU:

```bash
pip install -r requirements.txt
bash reproduce_figures.sh        # writes figures/
```

No GPU, TM-align binary or large download is needed for this step. The repository ships the per-ensemble scores and small per-GT summaries that the figure scripts read (see [Data](#data)).

| Paper item | Script (`pipeline/04_build_figures/`) | Output (`figures/`) |
|---|---|---|
| Fig. 2: TM vs. RMSD, TM ≥ 0.9 threshold | `fig_tm_vs_rmsd_scatter.py` | `tm_vs_rmsd_scatter.{png,pdf}` |
| Table 4: overall performance | `tab_overall_performance.py` | `tab_overall_performance.tex` |
| Fig. 3a–b: recall vs. diversity, ensemble size | `fig_dr_boxplots.py` | `dr_lineplot_{tm,ngt}_UNICORNE_BENCH.*` |
| Fig. 3c: recall vs. motion amplitude | `fig_cofail_mechanism.py` | `unicorne_mechanism.*` |
| Fig. 4a: effective mode count | `fig_mode_count.py` | `unicorne_mode_count.*` |
| Fig. 6a: best TM on AF2-OOD ensembles | `fig_unicorne_gold_standard.py --ood-only` | `unicorne_gold_standard_ood.*` |
| Fig. 6b: gold-standard panel (BioEMU, AFsample2) | `tab_unicorne_gold_standard.py --tools bioemu afsample2` | `tab_unicorne_gold_standard_reduced.tex` |
| Tables 10–11: OOD60 / DOMAINMOTION per protein | `tab_ood60_domainmotion.py` | `tab_{ood60,domainmotion}_full.tex` |
| Fig. 8: dataset characterization | `fig_dataset_characterization.py` | `fig_dataset_characterization.*` |
| Fig. 9: sample efficiency | `fig_dr_at_k.py`, `fig_at_k_gain.py` | `dr_at_k_*` |
| Fig. 10: recall vs. sequence length | `fig_dr_boxplots.py` | `dr_lineplot_len_UNICORNE_BENCH.*` |
| Tables 6–9: Wilcoxon signed-rank tests | `tab_wilcoxon_pvalues.py` | `tab_wilcoxon_pvalues.tex` |
| Fig. 12: inference runtime | `plot_runtime_benchmark.py` | `runtime_benchmark.*` |
| Fig. 13: best TM vs. diversity, all proteins | `fig_unicorne_gold_standard.py` | `unicorne_gold_standard.*` |
| Fig. 14: in-train vs. AF2-OOD | `fig_ood_boxplot.py` | `unicorne_ood_boxplot.*` |
| Fig. 15: co-failure and motion amplitude | `fig_cofail_mechanism.py` | `unicorne_cofail.*` |
| Table 12: gold-standard panel, all tools | `tab_unicorne_gold_standard.py` | `tab_unicorne_gold_standard.tex` |

AF2-OOD panels (Fig. 6a and the in-train vs. AF2-OOD boxplot) leave out AFsample3: they are defined by AlphaFold2's training cutoff (2018-05-01), and AFsample3 runs AlphaFold3.

Metric names in the code differ from the paper: `dr` is Recall@0.9, `cr` is the full-recall rate (FRR), and `eu` is the mean best-match TM ($\overline{\mathrm{TM}}$).

### Not regenerated here

These paper items come from outside this pipeline. Their final versions are in `external/` for reference.

| Item | Why |
|---|---|
| Fig. 1: curation diagram | Drawn by hand |
| Fig. 4b: P49258 UMAP | Needs the predicted structures. The script (`plot_kde_gt_callouts.py`) is included and runs once the predictions are in `data/outputs/` |
| Fig. 5: principal-angle heatmaps | Produced by the UNICORNEdb database pipeline |
| Fig. 7: Q5U925 case study | Rendered in PyMOL |
| Table 1: UNICORNEdb statistics | Produced by the UNICORNEdb database pipeline |
| Table 2: dataset comparison | Compiled from the cited datasets' publications |
| Table 3: prediction tools | Written by hand |
| Table 5: training-set overlap | Computed separately (release-date and sequence-identity partitions) |

## Repository layout

```
pipeline/
  00_build_source/          curated CSV -> data/inputs/<DATASET>/input_pdbs/
  01_build_benchmark/       GT x GT TM-align, filtering, max displacement, OOD60/DOMAINMOTION setup
  02_compute_predictions/   one launcher per tool, MSA generation, runtime benchmark
  03_compute_results/       GT x prediction TM-align, per-ensemble metrics, per-GT summaries
  04_build_figures/         every figure and table above
src/conformation_benchmark/ structure parsing and TM-align helpers used by stages 01-03
tool_scripts/               tool installers (TM-align, AlphaFlow, BioEMU, ESMDiff, AFsample2)
data/inputs/                benchmark definitions (CSV) and OOD60 / DOMAINMOTION structures
results/                    per-ensemble scores and per-GT summaries for every tool
external/                   paper assets this pipeline does not produce
reproduce_figures.sh        stage 04 end to end
```

## Data

### Included

| Path | Contents |
|---|---|
| `data/inputs/UNICORNE_BENCH/benchmarking-dataset-13.csv` | The benchmark: UniProt, reference sequence, representative GT structures |
| `data/inputs/UNICORNE_BENCH/filtered_ensembles{,_v2}.csv` | Ensembles kept by the stage-01 filters, with GT diversity and length |
| `data/inputs/UNICORNE_BENCH/fold_quality_blacklist.csv` | The 10 ensembles excluded by manual fold-quality review, with reasons |
| `data/inputs/UNICORNE_BENCH/uniprot_pdb_ids.csv` | Every deposited PDB entry per UniProt (for the AF2-OOD flag) |
| `data/inputs/{OOD60,DOMAINMOTION}/` | The two-state calibration sets, including GT structures |
| `data/pdb_deposition_dates.json` | PDB deposition dates (for the AF2-OOD flag) |
| `data/unicorne_training_overlap.csv` | Per-UniProt training-set membership (gold-standard ★) |
| `results/UNICORNE_BENCH/<tool>/results.csv` | Per-ensemble scores: recall, FRR, mean best-match TM, EMD, recall@k |
| `results/<DATASET>/<tool>/per_gt_summary.csv` | Per GT conformation: best prediction's TM and RMSD, and max TM |
| `results/<DATASET>/gt_matrices_tmalign/` | GT x GT TM and maximum-displacement matrices |
| `results/UNICORNE_BENCH/_mode_count_cache.npz` | Per-ensemble effective mode counts (Fig. 4a) |
| `results/runtime_benchmark*.csv` | Wall-clock measurements for the runtime figure |
| `results/tm_rmsd/tm_rmsd_scatter_points.csv` | Every point in Fig. 2: best prediction per GT conformation, with its TM and RMSD |
| `results/tm_rmsd/tm_rmsd_sample_counts.csv` | Scored predictions per OOD60 / DOMAINMOTION ensemble and tool (78–1,000, median 250) |

`per_gt_summary.csv` and `_mode_count_cache.npz` are condensed from the full GT x prediction TM-align output (about 85 GB). The figure scripts read the full output when it is present and fall back to these files otherwise; both give identical results. `pipeline/03_compute_results/export_per_gt_summary.py` and `fig_mode_count.py --rebuild` regenerate them from the full output. `export_tm_rmsd_points.py` regenerates `results/tm_rmsd/` the same way.

The ground-truth structures for UNICORNE-BENCH-1 (`data/inputs/UNICORNE_BENCH/input_pdbs/`, 116 MB) are not in git. They are in the benchmark archive at <https://doi.org/10.5281/zenodo.23928693>, which unpacks into this layout:

```bash
tar -xzf UNICORNE-BENCH-1.tar.gz --strip-components=1
```

### Not included

Predicted structures (1,000 per tool per ensemble, 6M+ in total) and the raw TM-align output. A sample with the GT inputs and the AFsample2 and BioEMU predictions is archived at <https://zenodo.org/records/20069060>.

## Tests

The tests check the evaluation from structure files to the published numbers:

```bash
bash tool_scripts/install_tmalign.sh ~/bin && export TMALIGN_EXE=~/bin/TMalign
python -m pytest
```

| File | Checks |
|---|---|
| `tests/test_tmalign_scoring.py` | Re-running TM-align on the shipped GT and prediction structures reproduces the stored scores (both normalizations, RMSD, lengths) exactly. Chain 1 is the GT, so the benchmark's TM-score is normalized by GT length; an independent C-alpha reader and the TM-score formula recompute both normalizations from TM-align's superposition. |
| `tests/test_metrics_from_structures.py` | Recall@0.9, full recovery and mean best TM recomputed from the fixture structures equal the published `results.csv` rows, including best TMs of 0.8996 (missed) and 0.9004 (recovered). |
| `tests/test_metric_definitions.py` | The metric definitions on hand-built TM matrices: inclusive 0.9 threshold, full recovery, failed pairs, recall@k, EMD. |
| `tests/test_headline_results.py` | Table 4 (N, Recall@0.9, FRR, TM-bar, EMD for all eight tools) from the shipped results, including the 8.2% full-recovery rate (68 of 828 ensembles for AFsample2 and AFsample3), and agreement between `results.csv` and the per-GT best scores for all 829 evaluated ensembles. |

`tests/data/` holds six ensembles chosen to cover full recovery, partial recall, best TMs on either side of 0.9, mixed mmCIF/PDB ground truths, and GTs shorter than the predictions (where normalizing by GT or prediction length changes which states count as recovered). For every tool it keeps each GT's best prediction from the full run, with the stored TM-align output for every GT x prediction pair; `tests/build_fixtures.py` regenerates it from the full output. Tests that need TM-align are skipped when it is not installed. TM-align is built from `third_party/TMalign/TMalign.cpp` (version 20220412), the source that produced the published scores.

## Running the full pipeline

The stages below rebuild everything from the curated benchmark CSV. Stages 01 and 03 need the TM-align binary; stage 02 needs GPUs and each tool's own installation.

```bash
bash tool_scripts/install_tmalign.sh ~/bin
export TMALIGN_EXE=~/bin/TMalign
```

### 00 - Inputs

```bash
python pipeline/00_build_source/build_inputs_from_csv.py \
    --csv data/inputs/UNICORNE_BENCH/benchmarking-dataset-13.csv \
    --dataset UNICORNE_BENCH --source-root /path/to/unicornedb/aligned-pdbs
python pipeline/01_build_benchmark/setup_bioemu_benchmarks.py --datasets OOD60 DOMAINMOTION
```

`--source-root` points at the UNICORNEdb structure tree (one directory of aligned PDB/mmCIF files per UniProt).

### 01 - Benchmark construction

```bash
python pipeline/01_build_benchmark/compute_gt_matrices.py UNICORNE_BENCH --workers 16
python pipeline/01_build_benchmark/filter_ensembles.py UNICORNE_BENCH
python pipeline/01_build_benchmark/compute_max_displacement.py UNICORNE_BENCH
python pipeline/01_build_benchmark/flag_fibril_like.py UNICORNE_BENCH   # candidates for the fold-quality review
```

The filters keep sequences of at most 300 residues, merge GT structures at TM ≥ 0.85 (complete linkage, one medoid per cluster), and require at least two distinct conformations and 80% coverage of the UniProt reference sequence.

### 02 - Predictions

Every tool writes `data/outputs/<tool>/UNICORNE_BENCH/output_pdbs/<UniProt>/*.pdb` with 1,000 samples per ensemble. All launchers accept `--resume` and `--gpu-shard i/N`. The MSA-based tools read ColabFold a3m files from `data/msas/<DATASET>/<UniProt>/a3m/`; generate them with

```bash
python pipeline/02_compute_predictions/generate_msas_local.py --dataset UNICORNE_BENCH
```

| Tool | Command (from the repository root) |
|---|---|
| BioEMU | `~/bioemu_venv/bin/python pipeline/02_compute_predictions/run_bioemu.py data/inputs/UNICORNE_BENCH UNICORNE_BENCH --num-samples 1000` |
| ESMFlow | `python pipeline/02_compute_predictions/run_esmflow.py data/inputs/UNICORNE_BENCH UNICORNE_BENCH --num-samples 1000` |
| AlphaFlow | `bash pipeline/02_compute_predictions/run_alphaflow_pdb.sh UNICORNE_BENCH --num-samples 1000` |
| ESMDiff | `python pipeline/02_compute_predictions/run_esmdiff.py data/inputs/UNICORNE_BENCH UNICORNE_BENCH --num-samples 1000` |
| CFrandom | `~/localcolabfold/.pixi/envs/default/bin/python pipeline/02_compute_predictions/run_cfrandom.py data/inputs/UNICORNE_BENCH UNICORNE_BENCH --msa-dir data/msas/UNICORNE_BENCH --num-samples 1000` |
| SPEACH_AF | `~/localcolabfold/.pixi/envs/default/bin/python pipeline/02_compute_predictions/run_speach_af.py data/inputs/UNICORNE_BENCH UNICORNE_BENCH --msa-dir data/msas/UNICORNE_BENCH --num-samples 1000 --seeds-per-set 9 --seeds-base 3` |
| AFsample2 | `bash pipeline/02_compute_predictions/run_afsample2.sh UNICORNE_BENCH UNICORNE_BENCH --num-samples 1000` |
| AFsample3 | `bash pipeline/02_compute_predictions/run_afsample3.sh UNICORNE_BENCH UNICORNE_BENCH --num-samples 1000 --tool afsample3_biorxiv --msa-fractions 0.0 0.1 0.2 0.3 0.4 0.5` |

Notes:
- AFsample2 rotates MSA-randomization fractions {0.05, …, 0.30}. AFsample3 is run at the masking sweep of its own paper ({0.0, …, 0.5}, Table 1 there), scored as `afsample3_biorxiv`. `reuse_fractions.py` can carve the overlapping fractions out of an earlier AFsample3 run instead of resampling them.
- SPEACH_AF: see `SPEACH_AF_MASKING_FIX.md` for the alanine-masking convention.
- SLURM array templates for SPEACH_AF and AFsample3 are in the same directory; adjust the account and partition lines for your cluster.
- Default tool locations (virtual environments, checkpoints, containers under `~`) are listed in each launcher's header and can be overridden by flags or environment variables.
- `runtime/` holds the inference-timing benchmark behind the runtime figure.

### 03 - Scoring

```bash
for t in bioemu esmflow_pdb esmdiff alphaflow_pdb cfrandom afsample2 afsample3_biorxiv speach_af; do
  python pipeline/03_compute_results/compute_tmalign_raw.py UNICORNE_BENCH $t --workers 32 --resume
  python pipeline/03_compute_results/compute_metrics.py UNICORNE_BENCH $t --filtered-only
  python pipeline/03_compute_results/export_per_gt_summary.py UNICORNE_BENCH $t
done
python pipeline/04_build_figures/fig_mode_count.py --rebuild
```

`compute_tmalign_raw.py` stores every GT x prediction TM-align result as JSON (about 80 minutes per tool on 32 cores). `compute_metrics.py` turns these into `results.csv`. Similarity between a GT and a prediction is TM-score normalized by GT length; the gold-standard table uses the larger of the two normalizations.

### 04 - Figures and tables

```bash
bash reproduce_figures.sh
```

## License

MIT (see `LICENSE`).
