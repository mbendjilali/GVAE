# Localization progress (Jun 2026)

Half-page recap of the fine supernode **position (`pos`)** push on DALES tiles (29 train / 11 val). Metric definitions: [architecture.md](architecture.md), [training.md](training.md).

## Three deliberate steps

| Step | Change | Fine `pos` (val) | Fine `anc` |
|------|--------|------------------|------------|
| 1 | Anchor losses + GT→h anchor curriculum | 0.28 → 0.17 | ~0.24 |
| 2 | **Z-only position readout** at `p_anchor` (default with `USE_Z_ONLY_DECODER`) | **0.14** | 0.21 |
| 3 | Stronger anchor λ (2.5 / 1.0 mid), `hzonly` λ 0.5 | **0.10** @ ep132 | **0.16** |

Best after step 3: `checkpoint/anchor_localize/best.pth` — val **11.24** @ ep132 (ep150 val 11.39; use `best.pth`). With step 2, **`pos` = `hzpos`** on every epoch.

## Reading the stack

```text
p_gt  ←  Z MLP refines ~0.04–0.06  ←  p_anchor (anc ≈ 0.16)  ←  Linear(h)
```

| Metric | ~best | Meaning |
|--------|-------|---------|
| **`anc`** | 0.16 | Where **h** places the anchor — **current bottleneck** |
| **`pos` / `hzpos`** | 0.10 | Z readout at that anchor — true localization for DDM |
| **`zpos`** | 0.08 | Z at **GT slot** (oracle); linear probe **0.11** → Z carries position |
| **`smiou`** | 76% | Deformable h+Z semantics — OK |
| **`occ`** | 50% | Fine grid IoU — flat; not fixed by localization work |

Total val loss (~11–12) includes anchor + hz recon terms — **not comparable** to older runs (~8).

## Recommended recipe

```bash
python train.py --ckpt-dir checkpoint/<new_run> \
  --lambda-recon-h 1.5 --lambda-recon-zonly 1.0 --lambda-recon-hzonly 0.5 \
  --lambda-anchor-fine 2.5 --lambda-anchor-mid 1.0
```

Probes run once at end → `probe_report.txt`, `probe_summary.json` (`--no-probe` to skip). **Use a fresh `--ckpt-dir` per experiment** — reusing `anchor_localize/` overwrites prior runs.

## Next experiment (`localize_v3`)

Code defaults (Jun 2026): **anchor MLP** (`USE_ANCHOR_MLP`), **splat trunc floor** (`SPLAT_MIN_TRUNC_VOXEL_FRAC`), **`LAMBDA_SEM_ZONLY`**, **coarse norm contrast**. Same CLI recipe as `r_focus_v2` with stronger anchor λ. Target **`anc` ≤ 0.10**, **`pos` ≤ 0.08**, **`zsmiou` ≥ 50%**.

Superseded May handoff: [archive/next_step-202605-handoff.md](archive/next_step-202605-handoff.md).
