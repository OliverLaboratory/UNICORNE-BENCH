# SPEACH-AF masking fix — corrected vs. OLD outputs (READ BEFORE ANALYZING speach_af DATA)

**TL;DR:** As of 2026-07-06 the corrected-masking speach_af outputs are **NOT in separately-named
folders**. Corrected and OLD (wrong-masking) `.pdb`s both live under the same
`data/outputs/speach_af/<dataset>/output_pdbs/<eid>/`. They are distinguished only by *which
filesystem* and by hand-moved `output_pdbs_*_wrongmask*` sibling backups. **Do not treat
`output_pdbs/` as "all corrected" until the consolidation step below is done.**

---

## 1. What changed in run_speach_af.py (why old data is invalid)

Two independent bugs made the original speach_af benchmark under-represent the method, especially
on larger proteins (mode collapse — e.g. Q8U2X0 never flipped to 6FHZ_A):

1. **Coverage truncation** (fixed: commit `e7c57ade`). `--seeds-per-set` was fixed at 9 with an
   early `break` once `--num-samples` was hit → only ~22 masking windows ever ran. Fix: default
   `--seeds-per-set 0` (AUTO — sized so every window is covered at any length) and removed the
   break.
2. **Wrong masking** (fixed: commit `36b7f0a8`, faithful port of RSvan/SPEACH_AF). Rewrote
   `interaction_sets()` to the canonical all-atom scheme: scan pLDDT>mean region, non-overlapping
   10-residue windows, all-atom 4 Å contacts, mask long-range contact PAIRS. Arg defaults changed:
   `--window 11→10`, `--contact-dist 8.0→4.0`.

**Consequence:** any speach_af output produced BEFORE these commits used wrong masking / truncated
coverage and must be regenerated. That regeneration is the "corrected" run described below.

## 2. Where the corrected UNICORNE_V3 re-run computed each ensemble (2026-07-06)

The 1,985 V3 ensembles were split across pools that write to DIFFERENT filesystems. All write to
`data/outputs/speach_af/UNICORNE_V3/output_pdbs/<eid>/` on their own FS:

| Pool | Ensembles (CSV index) | Filesystem | Notes |
|------|-----------------------|------------|-------|
| ACCRE array `12336932` | most of 0–1584 | ACCRE `/data/oliver_lab/...` | old moved to `output_pdbs_wrongmask_bak` (687 dirs) |
| bf64 (2 GPUs) | `idx % 40 ∈ {38,39}` (~98) | workstation **NFS** | old moved to `output_pdbs_bf64slice_wrongmask` |
| local pool (bf65 4 GPUs + augustine/aquinas/ignatius/origen/chesterton) | 1585–1984 minus bf64 residues (380), via subset CSV `data/inputs/UNICORNE_V3_localslice/input_dataset.csv` | workstation **NFS** | old moved to `output_pdbs_localslice_wrongmask` |

So on **NFS**, `output_pdbs/` is a MIX:
- **corrected**: bf64's ~98 + local's 380 = ~478 ensembles
- **STALE wrong-masking**: the ~1,400 ACCRE-assigned ensembles (never cleaned on NFS — their
  corrected versions live on ACCRE `/data`, not NFS).

On **ACCRE `/data`**, `output_pdbs/` is all corrected (old is in `output_pdbs_wrongmask_bak`).

## 3. The `--resume` stale-skip trap (how corrected runs silently do nothing)

`run_speach_af.py --resume` skips an ensemble when `output_pdbs/<eid>/output_*.pdb` count
`>= --num-samples`. If a stale wrong-masking dir is still there, resume counts it as done and
**produces zero corrected data** (log symptom: `Done: 38/42 ensembles, 13s total`). Therefore
**before any corrected re-run you MUST move the stale dirs aside** (reversible `mv`, not delete):

```bash
# example: move a slice's stale dirs aside before launching corrected speach
BAK=data/outputs/speach_af/UNICORNE_V3/output_pdbs_<slice>_wrongmask
mkdir -p "$BAK"
while IFS=, read eid _; do [ "$eid" = ensemble_ID ] && continue
  [ -d output_pdbs/$eid ] && mv output_pdbs/$eid "$BAK/$eid"; done < <subset>.csv
```

## 4. Consolidation — how to produce the final "corrected-only" set

When all pools finish, the canonical corrected set = ACCRE `/data` (corrected) ∪ NFS corrected
slices (bf64 98 + local 380). The NFS `output_pdbs` ~1,400 stale ACCRE-assigned dirs must be
REPLACED by ACCRE's corrected versions, not trusted:

1. On ACCRE, confirm completeness (`output_pdbs/<eid>` has ≥ target pdbs for the full 1,985).
2. Pull ACCRE `/data/.../speach_af/UNICORNE_V3/output_pdbs` → NFS, overwriting the stale
   ACCRE-assigned dirs (rsync of DATA between machines is allowed; code still goes via git).
   Do NOT overwrite the bf64/local corrected dirs — those exist only on NFS.
3. After consolidation, `output_pdbs/` = all corrected. The `output_pdbs_*_wrongmask*` backups
   can then be archived/deleted.

## 5. Quick "is this corrected?" checks
- ACCRE `/data` `output_pdbs/` → corrected (post-2026-07-04 mtime).
- NFS `output_pdbs/<eid>` → corrected ONLY if `<eid>` is in the bf64 slice (`idx%40∈{38,39}`) or
  the local slice (`UNICORNE_V3_localslice/input_dataset.csv`); otherwise STALE until consolidation.
- Anything under a `*_wrongmask*` dir is OLD — never use for results.

_Last updated 2026-07-06. See also the memory notes `speach-af-coverage-fix` and
`speach-resume-stale-skip`._
