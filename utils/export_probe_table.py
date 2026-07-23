#!/usr/bin/env python3
"""
Export probe_summary.json files as one markdown table (probes A / B / C x level).

The probe report is written for reading in a terminal; this is the same numbers
in the shape a paper table wants. Point it at any number of checkpoint dirs:

    python utils/export_probe_table.py checkpoint/latent_z96 checkpoint/latent_z96_zonly_ft

Probes, as defined in gvae/probes/latent.py:
  A  mix_0_pos_err              decoder position error with no GT anchor blended in
  B  linear_pos_err_val         ridge-free linear readout Z(p) -> p, held-out tiles
  C  norm_ratio_gt_over_empty   ||Z|| at GT sites over ||Z|| at empty occ voxels
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

LEVELS = ("fine", "mid", "coarse")

PROBES = (
    ("A", "mix_0_pos", "anchor", "mix_0_pos_err", "lower"),
    ("B", "linear_pos", "linear", "linear_pos_err_val", "lower"),
    ("C", "norm_ratio", "signal", "norm_ratio_gt_over_empty", "higher"),
)


def load(path: pathlib.Path) -> dict:
    summary = path / "probe_summary.json" if path.is_dir() else path
    if not summary.exists():
        sys.exit(f"no probe_summary.json at {summary} — run utils/probe_latent.py first")
    return json.loads(summary.read_text())


def cell(data: dict, section: str, key: str, level: str) -> str:
    """Two decimals on purpose: reruns of the same checkpoint move by up to 0.03
    at probe C (VAE sampling, empty-voxel draw, linear-probe init), so a third
    decimal is noise. See docs/probe-results.md."""
    value = data.get(section, {}).get(level, {}).get(key)
    return "—" if value is None else f"{value:.2f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoints", nargs="+", type=pathlib.Path,
                    help="Checkpoint dirs (or probe_summary.json paths)")
    ap.add_argument("-o", "--output", type=pathlib.Path, help="Write markdown here")
    args = ap.parse_args()

    runs = [(p.name if p.is_dir() else p.parent.name, load(p)) for p in args.checkpoints]

    lines = ["| Run | Probe | Direction | " + " | ".join(LEVELS) + " |",
             "|---|---|---|" + "---|" * len(LEVELS)]
    for name, data in runs:
        for tag, label, section, key, direction in PROBES:
            cells = " | ".join(cell(data, section, key, lvl) for lvl in LEVELS)
            lines.append(f"| `{name}` | {tag} `{label}` | {direction} is better | {cells} |")

    out = "\n".join(lines)
    if args.output:
        args.output.write_text(out + "\n")
        print(f"wrote {args.output}")
    else:
        print(out)


if __name__ == "__main__":
    main()
