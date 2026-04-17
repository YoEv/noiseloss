"""
Pairwise preference → per-item scalar scores via hinge (ReLU) + tie penalty.

Same objective as exp10 MusicPrefs v2 / train_musicprefs_prefmodel.py:
  win:  ReLU(m + s_loser - s_winner)  [optional square]
  tie:  (s_a - s_b)^2

Fits one scalar per track ID with gradient descent (Bradley–Terry-style),
then optionally maps scores to [low, high] (default 1–5) with min–max.

No PyTorch required. Intended for release pipelines (phase7_release).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, List, Sequence, Tuple, Union

import numpy as np

__all__ = [
    "Outcome",
    "Pair",
    "fit_pairwise_scores",
    "minmax_map_to_range",
]


class Outcome(str, Enum):
    WIN_A = "win_a"
    WIN_B = "win_b"
    TIE = "tie"


@dataclass(frozen=True)
class Pair:
    """One comparison between item_a and item_b (opaque string IDs, e.g. filenames)."""

    item_a: str
    item_b: str
    outcome: Outcome


def fit_pairwise_scores(
    pairs: Sequence[Pair],
    *,
    margin: float = 0.2,
    squared: bool = False,
    lr: float = 0.1,
    epochs: int = 200,
    seed: int = 42,
    clip_min: float = 0.0,
    clip_max: float = 10.0,
) -> Dict[str, float]:
    """
    Fit one score per unique item appearing in pairs.

    Gradients match train_musicprefs_prefmodel.fit_bradley_terry (hinge + tie MSE).
    """
    tracks = set()
    for p in pairs:
        tracks.add(p.item_a)
        tracks.add(p.item_b)
    if not tracks:
        return {}

    track_list = sorted(tracks)
    n = len(track_list)
    idx: Dict[str, int] = {t: i for i, t in enumerate(track_list)}
    rng = np.random.default_rng(seed)
    s = rng.uniform(2.0, 4.0, size=n)

    pairs_list: List[Tuple[int, int, Outcome]] = []
    for p in pairs:
        pairs_list.append((idx[p.item_a], idx[p.item_b], p.outcome))

    for _ in range(epochs):
        grad = np.zeros(n)
        for i, j, outcome in pairs_list:
            si, sj = s[i], s[j]
            if outcome == Outcome.WIN_A:
                diff = margin + sj - si
                if diff > 0:
                    if squared:
                        grad[i] -= 2 * diff
                        grad[j] += 2 * diff
                    else:
                        grad[i] -= 1.0
                        grad[j] += 1.0
            elif outcome == Outcome.WIN_B:
                diff = margin + si - sj
                if diff > 0:
                    if squared:
                        grad[j] -= 2 * diff
                        grad[i] += 2 * diff
                    else:
                        grad[j] -= 1.0
                        grad[i] += 1.0
            else:  # TIE
                diff = si - sj
                grad[i] += 2 * diff
                grad[j] -= 2 * diff
        s -= lr * grad
        s = np.clip(s, clip_min, clip_max)

    return {t: float(s[idx[t]]) for t in track_list}


def minmax_map_to_range(
    scores: Dict[str, float],
    low: float = 1.0,
    high: float = 5.0,
) -> Dict[str, float]:
    """Linear map score values to [low, high] by min–max over observed items."""
    if not scores:
        return {}
    vals = np.array(list(scores.values()), dtype=np.float64)
    mn, mx = float(vals.min()), float(vals.max())
    if mx <= mn:
        mid = (low + high) / 2.0
        return {k: mid for k in scores}
    scale = (high - low) / (mx - mn)
    return {k: low + scale * (float(v) - mn) for k, v in scores.items()}
