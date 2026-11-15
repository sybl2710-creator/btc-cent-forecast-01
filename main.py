"""Conservative, causal four-hour price forecast using only supplied closes."""

import json
import math
import os
from pathlib import Path


def weighted_mean(values, half_life):
    """Exponentially weight the newest observation most heavily."""
    decay = math.exp(-math.log(2.0) / half_life)
    total = weight_sum = 0.0
    weight = 1.0
    for value in reversed(values):
        total += weight * value
        weight_sum += weight
        weight *= decay
    return total / weight_sum


def forecast(prices):
    if not prices or any(not math.isfinite(p) or p <= 0 for p in prices):
        raise ValueError("Prices must be finite and positive")
    last = prices[-1]
    logs = [math.log(p) for p in prices]
    returns = [b - a for a, b in zip(logs, logs[1:])]
    if not returns:
        return round(min(9_999_999.99999, max(0.00001, last)), 5)

    recent_vol = math.sqrt(weighted_mean([r * r for r in returns[-240:]], 60))
    long_vol = math.sqrt(weighted_mean([r * r for r in returns], 720))
    momentum_rate = sum(
        weight * weighted_mean(returns[-window:], window / 2)
        for window, weight in ((60, 0.25), (120, 0.35), (240, 0.40))
    )
    momentum = 0.12 * 240 * momentum_rate

    anchor = weighted_mean(logs[-4320:], 720)
    reversion_strength = 0.04 * min(1.0, long_vol / max(recent_vol, 1e-12))
    reversion = reversion_strength * (anchor - logs[-1])
    shrinkage = 1.0 / (1.0 + (recent_vol / 0.0007) ** 2)
    # Equal component weights limit reliance on noisy short-term extrapolation.
    adjustment = last * (0.50 * momentum + 0.50 * reversion) * shrinkage
    adjustment = 25.0 * math.tanh(adjustment / 25.0)
    return round(min(9_999_999.99999, max(0.00001, last + adjustment)), 5)


def main():
    input_dir = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
    output_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    for source in sorted(input_dir.glob("*.in")):
        case = json.loads(source.read_text(encoding="utf-8"))
        if len(case["markets"]) != 1:
            raise ValueError("Expected exactly one market")
        price = forecast(case["markets"][0]["prices"])
        result = {"forecasts": [{"case": case["case"], "price": price}]}
        (output_dir / (source.stem + ".out")).write_text(
            json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
