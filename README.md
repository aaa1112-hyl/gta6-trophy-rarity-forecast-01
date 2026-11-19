# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/aaa1112-hyl/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/aaa1112-hyl/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecast, for every Grand Theft Auto VI trophy on PlayStation 5, the
share of the players Exophase tracks who will hold it at
2026-11-26T00:00:00Z, as five quantiles per trophy (5%, 25%, 50%,
75%, 95%). It reads one case file and writes one forecast file,
offline, with the Python standard library only.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

`main.py` reads every `*.in` case in the input directory and writes
`/output/<id>.out` for each, one object with a `trophies` list of
`{"id": n, "q": [q05, q25, q50, q75, q95]}` rows.

## How it forecasts

An earn rate is a share of players, so it rises toward a ceiling as
players progress and then saturates. Each trophy is fitted with a
saturating curve, `rate(t) = C * (1 - exp(-(t - d) / s))` in hours
since release; for a fixed ceiling the curve linearises, so candidate
ceilings are fitted by least squares and the one with the smallest
logit-space error wins, gently pulled toward a tier-typical ceiling
when the data cannot tell ceilings apart. The fitted curve is blended
with the trophy's last rate and with a straight logit-rate against
log-time fit, so a trophy whose rate is falling -- new players
tracked diluting it faster than players earn it -- is not forced onto
a rising curve. Story trophies are kept monotone (Chapter 1 at or
above Chapter 2, down to the final mission). When a case's whole
history is one unchanging page, Exophase has not refreshed it yet:
the next refresh dilutes the rates, so the median is pulled down and
the spread widened. Quantile spread grows with the horizon and with
each trophy's own fit residual.

On the practice cases the model scores about 0.19-0.34 mean pinball
loss on the logit scale against the two synthetic answers built from
the launch-night refresh (the house persistence baseline scores about
0.92-1.69 on the same answers); on synthetic final cases with a 4.6
day horizon it scores about 0.32 against 0.44-0.47 for persistence
and 0.66-0.74 for the curve baseline.

---

<sub>Entered on [make.fun](https://make.fun/?ref=fhvxzr4c&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
