"""GTA VI trophy rarity forecaster (make.fun gta6-trophy-rarity-forecast-01).

Reads one case file per run from the input directory: the house's reads of
Exophase's GTA VI PS5 trophy page up to a checkpoint, and writes, for every
trophy, five quantiles (0.05, 0.25, 0.5, 0.75, 0.95) of its earn rate — the
percent of tracked players who have earned it — at the case's target.

Approach. A rate is earners over players tracked, and the two move for
different reasons: the tracked base keeps growing after launch (diluting
every rate), while each trophy's earners grow as the base plays further
into the game. So the forecast models the two separately, on the case's own
history:

  * the page only changes when Exophase refreshes it, so the history is
    first compressed to its distinct page states;
  * players tracked and each trophy's earners are extrapolated as power
    laws in log(1 + hours since release), fitted by recency-weighted least
    squares over the states; a trophy's own earners slope is shrunk toward
    the pooled slope of all trophies (from the reads' trophies_earned),
    harder the fewer earners it has, and the prediction is anchored on the
    last observed state rather than on a fitted intercept;
  * the median rate is the ratio of the two extrapolations, and the spread
    on the logit scale combines the house random-walk scale with the
    fit's residual scatter, floored for trophies with a handful of
    earners, and widened upward for rare trophies, whose earners can
    multiply but cannot fall below zero;
  * when the history holds a single page state that is already hours old
    at the checkpoint, no dynamics are observable and a refresh of the
    page is overdue: the median is shifted down the logit scale (a
    refresh mostly dilutes, as new tracked players hold fewer trophies)
    and the spread widened to match.

Pure standard library, deterministic: the same case always writes the
same bytes. Outside the grader, MAKEFUN_INPUT_DIR / MAKEFUN_OUTPUT_DIR
stand in for /input and /output.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The directory of this file goes first on sys.path (the grader's -P flag
# leaves it off), so a sibling module would import cleanly.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# Standard normal quantiles at LEVELS.
Z = [-1.644854, -0.674490, 0.0, 0.674490, 1.644854]
# Random-walk scale of a rate's logit, per square-root day (the house's).
RW_SIGMA = 0.5


def clamp(x, lo, hi):
    return min(hi, max(lo, x))


def logit(pct):
    p = clamp(pct, 0.01, 99.99)
    return math.log(p / (100.0 - p))


def inv_logit(lg):
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x):
    """Round to 4 decimals, half up; whole numbers come back as int."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if float(value).is_integer() else value


def epoch(iso):
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()


def wlsq(xs, ys, ws):
    """Weighted least squares y ~ a + b*x; returns (a, b)."""
    sw = sum(ws)
    mx = sum(w * x for x, w in zip(xs, ws)) / sw
    my = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws))
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws))
    if sxx <= 1e-12:
        return my, 0.0
    b = sxy / sxx
    return my - b * mx, b


def page_states(history):
    """Compress reads to distinct page states (a state ends when the
    players-tracked count or any rate changes)."""
    states = []
    prev_key = None
    for read in history:
        key = (read["players_tracked"], tuple(read["rates"]))
        if key != prev_key:
            states.append(read)
            prev_key = key
    return states


def forecast_case(case):
    if case["levels"] != LEVELS:
        raise ValueError("unexpected levels")
    history = case["history"]
    if not history:
        raise ValueError("no history")
    release = datetime.fromisoformat(case["game"]["release_date"]).replace(
        tzinfo=timezone.utc
    ).timestamp()

    def hours(read):
        return (epoch(read["at"]) - release) / 3600.0

    def x_of(t_hours):
        # Power-law clock: log(1 + hours since release).
        return math.log(1.0 + max(t_hours, 0.0))

    states = page_states(history)
    xs = [x_of(hours(s)) for s in states]
    ws = [j + 1 for j in range(len(states))]  # recency weights
    x_last = xs[-1]
    x_star = x_of((epoch(case["target"]) - release) / 3600.0)
    horizon_h = case["horizon_minutes"] / 60.0
    horizon_d = case["horizon_minutes"] / 1440.0
    n_last = states[-1]["players_tracked"]

    # Players tracked at the target: power-law extrapolation, never below
    # the last count and never an absurd multiple of it.
    if len(states) >= 2:
        a_n, b_n = wlsq(xs, [math.log(s["players_tracked"]) for s in states], ws)
        b_n = clamp(b_n, 0.0, 1.2)
        n_star = clamp(math.exp(a_n + b_n * x_star), n_last, 12 * n_last)
        _, b_pool = wlsq(xs, [math.log(s["trophies_earned"] + 1) for s in states], ws)
        b_pool = clamp(b_pool, -0.5, 1.5)
    else:
        n_star = float(n_last)
        b_pool = 0.0

    # Per-trophy earners fits: log(E + 0.5) on the power-law clock.
    fits = []
    for i in range(len(case["trophies"])):
        series = [
            (xs[j], states[j]["rates"][i] * states[j]["players_tracked"] / 100.0)
            for j in range(len(states))
            if states[j]["rates"][i] is not None
        ]
        e_last = series[-1][1] if series else 0.0
        fit = None
        if len(series) >= 3 and e_last > 0:
            sx = [p[0] for p in series]
            sy = [math.log(p[1] + 0.5) for p in series]
            sw = [j + 1 for j in range(len(series))]
            a_i, b_i = wlsq(sx, sy, sw)
            resid = sum(
                w * (y - (a_i + b_i * x)) ** 2 for x, y, w in zip(sx, sy, sw)
            ) / sum(sw)
            fit = (b_i, resid, len(series))
        fits.append((e_last, fit))
    resid_pool = 0.09
    all_resid = sorted(f[1][1] for f in fits if f[1])
    if all_resid:
        resid_pool = all_resid[len(all_resid) // 2]

    # A single, hours-old page state: no dynamics observable, a refresh is
    # overdue, and a refresh mostly dilutes.
    stale_shift = 0.0
    stale_sigma = 0.0
    if len(states) == 1:
        state_age = (epoch(case["at"]) - release) / 3600.0 - hours(states[0])
        if state_age >= 1.5:
            stale_shift = min(1.9, 0.55 + 0.42 * (state_age + horizon_h))
            stale_sigma = 1.02

    rows = []
    for i, trophy in enumerate(case["trophies"]):
        # A trophy every tracked player has earned at every read has its
        # earners equal to the player count by construction; no fit may
        # drag it below 100.
        if all(s["rates"][i] == 100 for s in states):
            rows.append({"id": trophy["id"], "q": [100, 100, 100, 100, 100]})
            continue
        e_last, fit = fits[i]
        if fit:
            b_i, resid_i, n_i = fit
            # Shrink the trophy's own slope toward the pooled slope; the
            # fewer earners (and states) behind it, the harder the shrink.
            w_own = 2.0 + min(e_last, 40.0) / 8.0 + n_i / 4.0
            w_pool = 3.0 + 30.0 / (e_last + 2.0)
            slope = clamp(
                (w_own * b_i + w_pool * b_pool) / (w_own + w_pool), -0.6, 1.6
            )
            e_star = math.exp(math.log(e_last + 0.5) + slope * (x_star - x_last)) - 0.5
            resid = (n_i * resid_i + 6 * resid_pool) / (n_i + 6)
        elif e_last > 0:
            e_star = e_last * math.exp(b_pool * (x_star - x_last))
            resid = resid_pool
        else:
            e_star = 0.0
            resid = resid_pool
        e_star = clamp(e_star, 0.0, n_star)
        r_med = clamp(100.0 * e_star / n_star, 0.01, 99.99) if n_star > 0 else 0.01

        # Spread on the logit scale: random-walk scale plus fit scatter,
        # floored for near-zero earners, capped overall.
        sigma = math.sqrt(
            (0.85 * RW_SIGMA * math.sqrt(horizon_d)) ** 2
            + resid * (1.0 + horizon_h / 36.0)
        )
        sigma = max(sigma, 0.38, min(0.95, 0.45 + 0.55 / math.sqrt(e_last + 1.0)))
        sigma = min(sigma, 1.8)

        centre = logit(r_med)
        if stale_shift and 0.0 < r_med < 99.99:
            centre -= stale_shift
            sigma = max(sigma, stale_sigma)

        quantiles = []
        for z in Z:
            zz = z
            if r_med < 0.5:  # rare trophies surprise upward, not downward
                zz = z * 1.22 if z > 0 else z * 0.92
            quantiles.append(round4(inv_logit(centre + zz * sigma)))
        for k in range(1, len(quantiles)):
            if quantiles[k] < quantiles[k - 1]:
                quantiles[k] = quantiles[k - 1]
        rows.append({"id": trophy["id"], "q": quantiles})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        result = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(result, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
