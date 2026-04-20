"""
overlapping_window.py — Manages a set of staggered overlapping windows so that
consecutive predictions share context ticks, ensuring smooth temporal transitions
and avoiding cliff-edge classification changes at window boundaries.
"""

from collections import deque
from typing import Dict, List, Optional, Tuple
from core.data_generator import StockTick
from core.config import CFG


class OverlappingWindowSlot:
    """One slot in the staggered overlap schedule."""

    def __init__(self, slot_id: int, window_size: int, stride: int, phase_offset: int):
        self.slot_id      = slot_id
        self.window_size  = window_size
        self.stride       = stride
        self.phase_offset = phase_offset          # Ticks before first fire
        self._buf: deque  = deque(maxlen=window_size)
        self._ticks_since_fire = -phase_offset    # Negative = not yet warmed up
        self._fire_count  = 0

    def ingest(self, tick: StockTick) -> Optional[List[StockTick]]:
        """Returns a window snapshot if this slot fires, else None."""
        self._buf.append(tick)
        self._ticks_since_fire += 1
        if (self._ticks_since_fire >= self.stride and
                len(self._buf) >= self.window_size):
            self._ticks_since_fire = 0
            self._fire_count += 1
            return list(self._buf)[-self.window_size:]
        return None

    def resize(self, new_size: int) -> None:
        self.window_size = new_size
        new_stride = max(1, int(new_size * (1 - CFG.window.overlap_fraction)))
        self.stride = new_stride
        new_buf: deque = deque(maxlen=new_size)
        for t in self._buf:
            new_buf.append(t)
        self._buf = new_buf


class OverlappingWindowScheduler:
    """
    Maintains N_SLOTS staggered windows, each offset by (stride / N_SLOTS) ticks.
    This creates a dense, smoothly-overlapping stream of classification windows
    that collectively cover each tick multiple times from different temporal contexts.

    Benefits:
        - Eliminates boundary effects where a signal appears just before/after a window edge.
        - Provides N_SLOTS independent probability estimates that can be averaged.
        - Gives finer effective time resolution than a single window.
    """

    N_SLOTS = 4   # Number of staggered slots

    def __init__(self, window_size: Optional[int] = None):
        ws = window_size or CFG.window.base_window_size
        stride = max(1, int(ws * (1 - CFG.window.overlap_fraction)))
        slot_offset = max(1, stride // self.N_SLOTS)

        self._slots: List[OverlappingWindowSlot] = [
            OverlappingWindowSlot(
                slot_id=i,
                window_size=ws,
                stride=stride,
                phase_offset=i * slot_offset,
            )
            for i in range(self.N_SLOTS)
        ]
        self._total_ticks = 0
        self._fire_log: deque = deque(maxlen=1000)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest(self, tick: StockTick) -> List[Tuple[int, List[StockTick]]]:
        """
        Ingest a tick into all slots.
        Returns a list of (slot_id, window_ticks) for every slot that fired.
        """
        self._total_ticks += 1
        results = []
        for slot in self._slots:
            window = slot.ingest(tick)
            if window:
                results.append((slot.slot_id, window))
                self._fire_log.append({
                    "tick": self._total_ticks,
                    "slot": slot.slot_id,
                    "window_size": len(window),
                })
        return results

    def resize_all(self, new_window_size: int) -> None:
        """Propagate a window resize to all slots."""
        for slot in self._slots:
            slot.resize(new_window_size)

    def overlap_coverage(self) -> float:
        """
        Fraction of ticks that are covered by more than one window.
        With N_SLOTS and 50% overlap, nearly every tick is seen by 2+ windows.
        """
        ws = self._slots[0].window_size
        stride = self._slots[0].stride
        # Effective overlap = ws / stride  (> 1 means multi-coverage)
        coverage = ws / (stride + 1e-9)
        return round(coverage, 2)

    def aggregate_windows(self, fired_windows: List[Tuple[int, List[StockTick]]],
                          feature_fn, predict_fn) -> Optional[Dict]:
        """
        Given windows fired this tick, compute features + predictions from each,
        then aggregate into an ensemble result.
        """
        if not fired_windows:
            return None

        labels_votes: Dict[str, float] = {}
        confs = []
        for slot_id, ticks in fired_windows:
            feats = feature_fn(ticks)
            if feats is None:
                continue
            label, conf = predict_fn(feats)
            labels_votes[label] = labels_votes.get(label, 0.0) + conf
            confs.append(conf)

        if not labels_votes:
            return None

        ensemble_label = max(labels_votes, key=labels_votes.get)
        total_weight   = sum(labels_votes.values())
        ensemble_conf  = labels_votes[ensemble_label] / (total_weight + 1e-9)

        return {
            "ensemble_label":  ensemble_label,
            "ensemble_conf":   round(ensemble_conf, 4),
            "n_slots_fired":   len(fired_windows),
            "mean_conf":       round(sum(confs) / len(confs), 4),
            "vote_dist":       {k: round(v / total_weight, 3) for k, v in labels_votes.items()},
        }

    def stats(self) -> Dict:
        return {
            "n_slots":          self.N_SLOTS,
            "total_ticks":      self._total_ticks,
            "overlap_coverage": self.overlap_coverage(),
            "slot_fire_counts": [s._fire_count for s in self._slots],
            "window_size":      self._slots[0].window_size,
            "stride":           self._slots[0].stride,
        }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from core.data_generator import StockDataGenerator
    gen = StockDataGenerator(seed=3)
    gen.start()
    sched = OverlappingWindowScheduler(window_size=30)

    total_fires = 0
    for tick in gen.iter_ticks(max_ticks=300):
        fired = sched.ingest(tick)
        for slot_id, window in fired:
            total_fires += 1
            if total_fires <= 5:
                print(f"Slot {slot_id} fired | window={len(window)} ticks "
                      f"| coverage={sched.overlap_coverage():.1f}x")

    gen.stop()
    print(f"\nTotal window fires: {total_fires}")
    print(sched.stats())
