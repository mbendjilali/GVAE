# gvae/training/console.py
# ANSI terminal styling (plain text in train.log via Tee strip)

from __future__ import annotations

import math
import re
import sys

from tqdm import tqdm

ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    return ANSI_ESCAPE.sub("", text)


class Style:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    GRAY = "\033[90m"


class Term:
    """TTY-aware colored output; use write() so tqdm bars stay intact."""

    def __init__(self, enabled: bool | None = None):
        self.enabled = (
            enabled if enabled is not None
            else getattr(sys.stdout, "isatty", lambda: False)()
        )

    def paint(self, text: str, *codes: str) -> str:
        if not self.enabled or not codes:
            return text
        return f"{''.join(codes)}{text}{Style.RESET}"

    def write(self, text: str = "", *, end: str = "\n") -> None:
        tqdm.write(text, end=end)

    def banner(self, title: str, lines: list[str]) -> None:
        width = max(len(title), *(len(l) for l in lines), 52)
        rule = self.paint("─" * width, Style.DIM)
        self.write(rule)
        self.write(self.paint(title, Style.BOLD, Style.CYAN))
        for line in lines:
            self.write(line)
        self.write(rule)

    def epoch_header(
        self,
        epoch: int,
        total: int,
        lr: float,
        train_loss: float,
        val_loss: float,
        *,
        is_best: bool = False,
    ) -> None:
        def _loss(v: float, color: str) -> str:
            s = f"{v:.4f}" if math.isfinite(v) else "nan"
            return self.paint(s, color)

        ep = self.paint(f"Epoch {epoch}/{total}", Style.BOLD)
        lr_s = self.paint(f"lr={lr:.1e}", Style.DIM)
        tr = _loss(train_loss, Style.GREEN)
        va = _loss(val_loss, Style.CYAN)
        star = self.paint("  ★ best", Style.YELLOW, Style.BOLD) if is_best else ""
        self.write(f"\n{ep}  {lr_s}  train {tr}  val {va}{star}")

    def metrics_line(self, metrics: dict) -> None:
        if not metrics:
            return
        parts = []
        if "pos_err_fine" in metrics:
            parts.append(
                f"fine pos={metrics['pos_err_fine']:.3f} "
                f"miou={metrics.get('miou_fine', 0):.0%} "
                f"occ={metrics.get('occ_iou_fine', 0):.0%}"
            )
        parts.append(
            f"mid inst={metrics.get('inst_pos_err_mid', 0):.3f} "
            f"occ={metrics.get('occ_iou_mid', 0):.0%} "
            f"smiou={metrics.get('soft_miou_mid', 0):.0%}"
        )
        line = "  │ " + self.paint("metrics", Style.MAGENTA) + "  " + "  ·  ".join(parts)
        self.write(line)

    def warn(self, msg: str) -> None:
        self.write(self.paint(f"  ⚠ {msg}", Style.YELLOW))

    def ok(self, msg: str) -> None:
        self.write(self.paint(f"  ✓ {msg}", Style.GREEN))

    def dim(self, msg: str) -> None:
        self.write(self.paint(msg, Style.DIM))
