# Dilution & Progression — GTA VI Trophy Rarity Forecaster

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/adzinsyariif/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/adzinsyariif/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

An entry for the make.fun bounty [“Forecast how many GTA VI players earn each trophy in week one”](https://make.fun/bounty/gta6-trophy-rarity-forecast-01).

`main.py` (Python 3.13, standard library only, no network) reads one case
file — the house's reads of Exophase's Grand Theft Auto VI PS5 trophy page
up to a checkpoint — and writes five quantiles (0.05, 0.25, 0.5, 0.75,
0.95) of every trophy's earn rate at the case's target, scored by pinball
loss on the logit scale.

## How it forecasts

An earn rate is a ratio, and its two sides move for different reasons
after a launch: the tracked player base keeps growing, which dilutes
every rate, while each trophy's earners grow as the base plays further
into the game. The forecaster models the two sides separately, from the
case's own history:

- **Page states.** Exophase only changes the page when it refreshes its
  counts, so the history is compressed to its distinct page states
  before anything is fitted.
- **Power-law clock.** Players tracked and each trophy's earners are
  extrapolated as power laws in `log(1 + hours since release)`, fitted by
  recency-weighted least squares over the states. A trophy's own earners
  slope is shrunk toward the pooled slope of all trophies — harder the
  fewer earners stand behind it — and the prediction is anchored on the
  last observed state, not on a fitted intercept. The median rate is the
  ratio of the two extrapolations.
- **Calibrated spread.** The spread on the logit scale combines the
  random-walk scale with the fit's residual scatter, is floored for
  trophies with a handful of earners, and is widened upward for rare
  trophies, whose earners can multiply but cannot fall below zero. A
  trophy every tracked player has earned at every read is pinned at 100:
  its earners are the player count by construction.
- **Stale pages.** If the history holds a single page state that is
  already hours old at the checkpoint, no dynamics are observable and a
  refresh is overdue; since a refresh mostly dilutes (new tracked players
  hold fewer trophies), the median is shifted down the logit scale and
  the spread widened to match.

The program is deterministic — the same case always produces the same
bytes — and runs in milliseconds.

## Running it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Each `<id>.in` in the input directory produces a `<id>.out` holding
`{"trophies": [{"id": …, "q": [q05, q25, q50, q75, q95]}, …]}`.
