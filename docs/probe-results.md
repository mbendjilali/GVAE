# Probe results

The three latent probes, exported for every checkpoint that still exists.
Probe definitions live in `gvae/probes/latent.py`; what they are *for* is in
[architecture.md](architecture.md) and [training.md](training.md#latent-probes).

| Probe | Key | Asks |
|---|---|---|
| **A** `mix_0_pos` | `anchor.<level>.mix_0_pos_err` | Can the decoder place supernodes with **no GT anchor** blended in? |
| **B** `linear_pos` | `linear.<level>.linear_pos_err_val` | Is position **linearly decodable** from `Z(p)` on held-out tiles? |
| **C** `norm_ratio` | `signal.<level>.norm_ratio_gt_over_empty` | Is `‖Z‖` larger at object sites than at empty voxels? Gate: **> 1.0** |

Scene coordinates are normalised to `[-1, 1]³`, so a position error of 1.0 is the
scale of the scene itself, not a small miss.

## Regenerating

```bash
python utils/probe_latent.py --checkpoint checkpoint/<run>/best.pth \
  -o checkpoint/<run>/probe_report.txt          # writes probe_summary.json too
python utils/export_probe_table.py checkpoint/<run>        # summary -> this table
```

Probes also run automatically at the end of training unless `--no-probe`.

## Current checkpoints

<!-- generated: python utils/export_probe_table.py checkpoint/latent_z96 checkpoint/latent_z96_zonly_ft -->

| Run | Probe | Direction | fine | mid | coarse |
|---|---|---|---|---|---|
| `latent_z96` | A `mix_0_pos` | lower is better | 0.14 | 0.14 | 1.08 |
| `latent_z96` | B `linear_pos` | lower is better | 0.67 | 0.67 | 0.39 |
| `latent_z96` | C `norm_ratio` | higher is better | 1.89 | 1.63 | 1.47 |
| `latent_z96_zonly_ft` | A `mix_0_pos` | lower is better | 1.39 | 1.12 | 1.07 |
| `latent_z96_zonly_ft` | B `linear_pos` | lower is better | 0.67 | 0.68 | 0.39 |
| `latent_z96_zonly_ft` | C `norm_ratio` | higher is better | 1.86 | 1.63 | 1.47 |

**Reported to two decimals on purpose.** Re-running the probes on the *same*
`latent_z96/best.pth` reproduces A to ±0.003 and B to ±0.007, but C only to
±0.03 (fine 1.89 → 1.92, coarse 1.47 → 1.45). The VAE samples `Z = μ + σ⊙ε`, the
empty voxels are drawn at random, and the linear probe is randomly initialised.
A third decimal is noise.

### Reading these

- **C passes at every level, in both runs** — 1.86–1.89 fine, ~1.63 mid, ~1.47
  coarse, all clear of the 1.0 gate. `‖Z‖` genuinely peaks at object sites.
- **B fails at fine and mid** — 0.67, which is the "good metrics, bad latents"
  figure (~0.68) from §4.1 of the [retrospective](development-retrospective.md).
  Position is *present* in `Z` as magnitude but is **not linearly decodable**.
  C and B disagreeing is the central negative result here.
- **B is best at coarse** (0.39) while A is worst there (1.08). Coarse `Z` carries
  a scene envelope a linear map can read, but the decoder cannot place individual
  coarse supernodes at all — consistent with the known limitation that coarse is
  adequate for an envelope and not for instance pinning.
- **`latent_z96_zonly_ft` collapses probe A** (fine 0.14 → 1.39) while its B and C
  are unchanged to within tolerance. That is the expected signature of a
  frozen-encoder finetune: A exercises the decoder, which was retrained; B and C
  read the encoder, which was not. It is a consistency check on the freeze
  (`gvae/training/freeze.py`), not a result about `Z`.

## Runs whose weights no longer exist

`norm_depth3_150` (ep 145) and `anchor_localize` (ep 132) are **not on disk** —
`checkpoint/` holds only the two runs above. Their probes cannot be re-exported,
and the numbers below are transcribed from prose, not regenerated. Treat them as
a historical record at the precision they were written down.

| Run | A fine | A mid | A coarse | B fine | B mid | B coarse | C fine | C mid | C coarse |
|---|---|---|---|---|---|---|---|---|---|
| `norm_depth3_150` | 0.18 | 0.15 | — | 0.14 | 0.11 | — | 1.51 | 1.41 | ~0.75 |
| `anchor_localize` | — | — | — | 0.11 | — | — | — | — | — |

Sources: [development-retrospective.md](development-retrospective.md) §5 for
`norm_depth3_150`; [localization-progress.md](localization-progress.md) for
`anchor_localize`, which records console metrics (`pos` 0.10, `anc` 0.16, `zpos`
0.08, `smiou` 76%) and a single linear-probe figure.

**A full 3×3 table for these two runs cannot be reconstructed.** Eight of the
eighteen cells survive; the rest were never written down and the weights that
would regenerate them are gone.

### Comparing across the two groups

Do not read `norm_depth3_150` B fine 0.14 against `latent_z96` B fine 0.67 as a
straight regression. The runs differ in latent width (72/144/288 → 96/192/384),
in decode path (h+Z with anchor heads → `LatentGraphDecoder`, no `h`), and in
what the objective asks for. The June model is scored on a strictly harder task.
What the comparison does support is narrower and still worth stating: **the
linear decodability that the May localization work bought back was not retained
by the latent-graph formulation.**
