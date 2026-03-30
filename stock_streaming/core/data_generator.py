"""
data_generator.py — Simulates a live stock price feed using geometric Brownian motion
with regime switches and random volatility spikes to stress-test the streaming system.
"""

import time
import math
import queue
import random
import threading
from dataclasses import dataclass, field
from typing import Optional, Iterator
from core.config import CFG


@dataclass
class StockTick:
    """A single market tick emitted by the generator."""
    symbol: str
    timestamp: float          # Unix epoch (seconds)
    price: float
    volume: int
    bid: float
    ask: float
    tick_index: int           # Monotonic counter since stream start

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def mid_price(self) -> float:
        return (self.bid + self.ask) / 2.0


class GBMParameters:
    """Geometric Brownian Motion parameters, possibly regime-switching."""
    def __init__(self, mu: float = 0.0001, sigma: float = 0.005,
                 vol_spike_prob: float = 0.02, vol_spike_factor: float = 5.0):
        self.mu = mu
        self.sigma = sigma
        self.vol_spike_prob = vol_spike_prob
        self.vol_spike_factor = vol_spike_factor
        self._base_sigma = sigma
        self._spike_active = False
        self._spike_remaining = 0

    def step_sigma(self) -> float:
        """Return sigma for this tick, possibly entering/exiting a spike."""
        if self._spike_active:
            self._spike_remaining -= 1
            if self._spike_remaining <= 0:
                self._spike_active = False
                self.sigma = self._base_sigma
        elif random.random() < self.vol_spike_prob:
            self._spike_active = True
            self._spike_remaining = random.randint(5, 20)
            self.sigma = self._base_sigma * self.vol_spike_factor
        return self.sigma


class StockDataGenerator:
    """
    Generates synthetic stock ticks at a configurable rate.
    Ticks are placed into a thread-safe queue consumed by the streaming pipeline.
    """

    def __init__(self, symbol: str = "SYNTH", start_price: float = 150.0,
                 seed: Optional[int] = None):
        self.symbol = symbol
        self.price = start_price
        self.gbm = GBMParameters()
        self._tick_index = 0
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self.queue: queue.Queue = queue.Queue(maxsize=CFG.stream.max_queue_size)
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _next_price(self) -> float:
        sigma = self.gbm.step_sigma()
        z = self._rng.gauss(0, 1)
        dt = CFG.stream.tick_interval_ms / (252 * 6.5 * 3600 * 1000)  # fraction of trading year
        self.price *= math.exp((self.gbm.mu - 0.5 * sigma ** 2) * dt + sigma * math.sqrt(dt) * z)
        return round(self.price, 4)

    def _next_volume(self) -> int:
        base = self._rng.randint(100, 10_000)
        spike = self._rng.random() < 0.05
        return base * self._rng.randint(5, 20) if spike else base

    def _make_tick(self) -> StockTick:
        price = self._next_price()
        half_spread = round(self._rng.uniform(0.01, 0.05), 4)
        tick = StockTick(
            symbol=self.symbol,
            timestamp=time.time(),
            price=price,
            volume=self._next_volume(),
            bid=round(price - half_spread, 4),
            ask=round(price + half_spread, 4),
            tick_index=self._tick_index,
        )
        self._tick_index += 1
        return tick

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start emitting ticks into self.queue in a background thread."""
        self._running = True
        self._thread = threading.Thread(target=self._emit_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal the background thread to stop."""
        self._running = False

    def _emit_loop(self) -> None:
        interval_s = CFG.stream.tick_interval_ms / 1000.0
        while self._running:
            tick = self._make_tick()
            try:
                self.queue.put(tick, timeout=0.2)
            except queue.Full:
                pass  # Drop tick under back-pressure; real system would apply flow control
            time.sleep(interval_s)

    def iter_ticks(self, max_ticks: Optional[int] = None) -> Iterator[StockTick]:
        """Blocking iterator — yields ticks as they arrive."""
        count = 0
        while True:
            try:
                tick = self.queue.get(timeout=1.0)
                yield tick
                count += 1
                if max_ticks and count >= max_ticks:
                    break
            except queue.Empty:
                if not self._running:
                    break

    def generate_batch(self, n: int) -> list:
        """Synchronously generate n ticks without threading (for batch tests)."""
        return [self._make_tick() for _ in range(n)]


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    gen = StockDataGenerator(symbol="AAPL", start_price=180.0, seed=42)
    gen.start()
    for i, tick in enumerate(gen.iter_ticks(max_ticks=10)):
        print(f"[{tick.tick_index:04d}] {tick.symbol} price={tick.price:.4f} "
              f"vol={tick.volume:,} spread={tick.spread:.4f}")
    gen.stop()
    print("Generator stopped.")
