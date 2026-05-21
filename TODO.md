# GVAE — TODO

Documentation: [docs/](docs/) · Architecture: [docs/architecture.md](docs/architecture.md)

Backlog for work **outside** Layer A (PR2) implementation.

**Baseline checkpoint:** `checkpoint/20260520_163246/stage4_best.pth` (PR1 + Z-only decoder, val 0.664 @ S4 ep15).

**Decision metrics (go/no-go):** `inst_pos_err`, `occ_iou` + `occ_precision`, `pos_err_mid` (Z-only), `miou_fine` (after PR2). Treat mid `soft_miou_mid` as diagnostic only until A2 is done.

---

## In progress — Layer A (PR2)

- [x] `Z_fine` encoder path from instance graph `G_L` (instantiable nodes only)
- [x] Instance reconstruction loss with one-hot labels
- [x] `occ_fine` cache + occupancy readout at fine grid (rebuild caches required)
- [x] Training stages / stage-aware forward (skip unused U-Nets per stage)
- [x] Primary metrics: `pos_err_fine`, `miou_fine`, `occ_iou_fine`
- [x] Grids updated: fine 64³, mid 32³, coarse 16³
- [ ] Rebuild all occupancy caches: `python utils/build_scene_graph.py <data_root>`
- [ ] Full training run + baseline comparison

---

## A. Metric trust & diagnostics (~1 day, do first)

Answer “is the number lying?” before tuning on it.

| ID | Task | Action | Done when |
|----|------|--------|-----------|
| A1 | One-shot full-metrics eval | Run val on `stage4_best.pth` with `LOG_FULL_METRICS=True` | Have `recon_sem_mid`, `soft_miou_coarse`, per-level pos err |
| A2 | Decompose `soft_miou` ceiling | Log per-supernode: entropy of `s_lm1`, #classes with mass > ε, argmax vs soft target | Confirmed low mIoU is merged-label ambiguity, not zero gradient |
| A3 | Instance vs supernode gap | Track `inst_pos_err_mid` vs `pos_err_mid` delta in TensorBoard | Stable gap → `S1` projection is the bottleneck |
| A4 | Occ metric sanity | Visual check: 2–3 scenes, `occ_mid` pred vs cache GT | 73% IoU is not a bbox-trivial artifact |
| A5 | Freeze decision dashboard | Document trusted metrics in this file / run notes | Mid `soft_miou` excluded from go/no-go until A2 done |

- [ ] A1 — Full-metrics eval on best checkpoint
- [ ] A2 — Supernode label ambiguity analysis
- [ ] A3 — Instance vs supernode gap logging
- [ ] A4 — Occ visual sanity check
- [ ] A5 — Decision metrics documented

---

## B. Training schedule & checkpoint hygiene (~half day)

| ID | Task | Action |
|----|------|--------|
| B1 | Shorten stage 4 | `NUM_EPOCHS_STAGE4=15–20` + early-stop on val total (best was ep 15, drift after) |
| B2 | Stage 1 early stop | S1 ≤ 30 epochs (best pool @ ep 28) |
| B3 | Checkpoint policy | Always use `stage{N}_best.pth`, not `final.pth` |
| B4 | Metrics with checkpoint | On best save, dump val metrics JSON next to `.pth` |

- [ ] B1 — Shorten stage 4
- [ ] B2 — Stage 1 early stop
- [x] B3 — Checkpoint policy documented ([docs/training.md](docs/training.md))
- [ ] B4 — Save metrics JSON with best checkpoints

---

## C. `inst_pos_err` still ~0.25 (ablations — after PR2 unless noted)

| ID | Task | Hypothesis | Experiment |
|----|------|------------|------------|
| C1 | S1 quality audit | Bad hard assign → bad instance projection | Log distance from instance to assigned supernode centroid |
| C2 | Pool vs layout tradeoff | `REDUCTION_RATIO=0.03` over-merges | One run at 0.05–0.08; watch `inst_pos_err` vs pool loss |
| C3 | Anchor curriculum | Z-only anchors too hard early | `DECODER_GT_ANCHOR_MIX`: 0.5 → 0.0 over S2 (one run) |
| C4 | Instance loss weight | Geometry starved vs occ | After PR2: bump `LAMBDA_POS` on instance recon only |
| C5 | Grad balance | Sem/pos/occ fighting in shared decoder | Per-term grad norms on one diagnostic run |

- [ ] C1 — S1 quality logging (free, do anytime)
- [ ] C2 — `REDUCTION_RATIO` ablation
- [ ] C3 — Anchor curriculum ablation
- [ ] C4 — Instance loss weight (post-PR2)
- [ ] C5 — Grad balance check (optional)

**Priority:** C1 first. C2 highest-signal mid-level ablation. C3–C5 only if PR2 `inst_pos_err` still > 0.2.

---

## D. `occ_iou` plateau ~73%

| ID | Task | Hypothesis | Experiment |
|----|------|------------|------------|
| D1 | Recall vs precision | Conservative blob (high prec, low recall) | Add `occ_recall_mid` to metrics |
| D2 | Grid resolution | 16³ too coarse | Compare `occ_iou_mid` vs `occ_iou_fine` after PR2 |
| D3 | Query sampling | Queries miss hard negatives | Inspect `sample_occupancy_queries`; more surface samples |
| D4 | λ_occ sweep | Occ underweighted vs recon | One run: `LAMBDA_OCC ∈ {0.5, 2.0}` |
| D5 | Cache coverage | 500k subsample drops thin structures | Rebuild one scene at full points; compare IoU |

- [ ] D1 — Add occ recall to metrics
- [ ] D2 — Mid vs fine occ comparison (post-PR2)
- [ ] D3 — Occ query sampling review
- [ ] D4 — `LAMBDA_OCC` ablation
- [ ] D5 — Full-point occ cache spot-check

**Priority:** D1 → D4. D2 answered by Layer A fine grid.

---

## E. Mid `soft_miou` ~10% — monitor, don’t optimize yet

| ID | Task | When |
|----|------|------|
| E1 | Accept mid ceiling; diagnostic only | Now |
| E2 | Instance hard `miou_fine` in PR2 | With Layer A |
| E3 | Optional mid ablation: `LAMBDA_SEM=2` or higher `REDUCTION_RATIO` | Only if A2 shows sem loss decreasing but mIoU stuck |
| E4 | Coarse `soft_miou` baseline from full metrics | After A1 |

- [ ] E1 — Acknowledged (no mid-only tuning sprint)
- [ ] E2 — `miou_fine` in PR2
- [ ] E3 — Mid sem ablation (conditional)
- [ ] E4 — Coarse soft mIoU baseline

---

## F. Architecture / XCube alignment (post-PR2, strategic)

| ID | Task | Notes |
|----|------|-------|
| F1 | Z level nesting | `Z_mid` and `Z_coarse` are independent grids, not voxel-nested |
| F2 | Cross-level consistency | Optional: loss tying `Z_coarse` ↔ downsample(`Z_mid`) |
| F3 | Hybrid cascade scope | Graph GVAE L1–L3; XCube-style sparse cascade L4+ |
| F4 | Background channel | B policy excludes ground/veg/fence from coarsening; confirm occ carries terrain shell |

- [ ] F1 — Document Z conditioning for DDM integration
- [ ] F2 — Cross-level consistency (future)
- [ ] F3 — Hybrid cascade decision (after PR2 metrics)
- [ ] F4 — Background / terrain signal for DDM

---

## G. Data & scale (ongoing)

| ID | Task | Action |
|----|------|--------|
| G1 | Dataset size | 29 train / 11 val — plan expansion or note val variance |
| G2 | Val split stability | Fixed test-tile list documented or committed |
| G3 | Per-scene failure log | Worst-3 scenes by `inst_pos_err` each epoch |

- [ ] G1 — Dataset expansion plan
- [ ] G2 — Fixed val split documented
- [ ] G3 — Worst-scene logging per epoch

---

## Suggested order

```
Week 0 (parallel, ~1 day):
  A1 → A2 → B1/B3 → D1

During PR2:
  Layer A + E2 + G3

After first PR2 run:
  Compare inst_pos_err / miou_fine / occ_iou_fine vs PR1 baseline
  → pick ONE ablation from C2, C3, D4 based on worst metric

Defer until PR2 numbers:
  C2, F2, F3, D2
```

---

## Worry → track → metric

| Worry | Track | Primary metric (post-PR2) |
|-------|-------|---------------------------|
| Layout still weak | Layer A + C | `inst_pos_err`, `pos_err_fine` |
| Semantics look broken | E2 (not mid tuning) | `miou_fine` |
| Occ blob plateau | D | `occ_iou_fine`, recall |
| Training wastes time | B | val total at best epoch |
| Metrics untrustworthy | A | frozen dashboard above |
| Scale / generalization | G | val variance across runs |

---

## Explicitly deferred

- Full XCube pivot
- Multi-week mid-only semantic tuning
- Rewriting coarsening (FPS fixed; only `mlp_S` trains)
- Chasing `soft_miou_mid` above ~15% before instance-level metrics exist
