"""
utils/logging_utils.py
======================
Lightweight pretty logging helpers. No external dependencies.
"""

import time
from datetime import datetime


def _timestamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


def log(message: str, icon: str = "ℹ️") -> None:
    print(f"[{_timestamp()}] {icon}  {message}")


def log_section(title: str) -> None:
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def log_stats(label: str, stats: dict) -> None:
    print(f"\n📊 {label}")
    max_key = max(len(k) for k in stats.keys()) if stats else 10
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"   {k:<{max_key}} : {v:.4f}")
        elif isinstance(v, int):
            print(f"   {k:<{max_key}} : {v:,}")
        else:
            print(f"   {k:<{max_key}} : {v}")


class Timer:
    def __init__(self, label: str = ""):
        self.label = label
        self.start = None

    def __enter__(self):
        self.start = time.time()
        if self.label:
            log(f"Starting: {self.label}", "⏱️")
        return self

    def __exit__(self, *args):
        elapsed = time.time() - self.start
        minutes = int(elapsed // 60)
        seconds = elapsed % 60
        if minutes > 0:
            log(f"Done: {self.label} — {minutes}m {seconds:.1f}s", "✅")
        else:
            log(f"Done: {self.label} — {seconds:.2f}s", "✅")