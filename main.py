#!/usr/bin/env python3
"""Forecast GTA VI trophy earn rates on Exophase at a case target.

Reads every *.in case in MAKEFUN_INPUT_DIR (else /input) and writes
/output/<id>.out with five quantiles per trophy.

Method, per trophy.  An earn rate is a share of players, so it rises
toward a ceiling as players progress and saturates: it is fitted with
a saturating curve rate(t) = C * (1 - exp(-(t - d) / s)) in hours since
release.  For a fixed ceiling C the curve linearises -- with
y = -ln(1 - rate / C) the model is y = (t - d) / s, a straight line --
so each candidate ceiling is fitted by least squares and the ceiling
with the smallest logit-space error wins, gently pulled toward a
tier-typical ceiling when the data cannot tell ceilings apart.  The
fitted curve is blended in logit space with the trophy's last rate
(persistence) and with a straight logit-rate against log-time fit
(the house curve's shape), so a trophy whose rate is falling -- new
players tracked diluting it faster than players earn it -- is not
forced onto a rising curve.  Story trophies named "Chapter N" are
kept monotone (chapter 1 at or above chapter 2, down to the final
mission).  When a case's whole history is one unchanging page --
Exophase has not refreshed it yet -- a refresh is due, and the next
refresh dilutes the rates: new players tracked hold far fewer
trophies than the launch cohort, so the median is pulled down and
the spread widened.  Quantile spread grows with the horizon and
with each trophy's own fit residual, so a trophy whose history jumps
at refreshes gets a wider band.
"""

import json
import math
import os
from datetime import datetime, timezone

RELEASE = datetime(2026, 11, 19, tzinfo=timezone.utc)
Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722)
LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)

# Loose ceilings by tier, only to stop a runaway extrapolation.
TIER_CAP = {"platinum": 8.0, "gold": 30.0, "silver": 50.0, "bronze": 99.99}
# Tier-typical ceilings, the gentle prior when data cannot tell.
TIER_PRIOR = {"platinum": 1.5, "gold": 8.0, "silver": 12.0, "bronze": 20.0}

# Stale-page refresh: the next refresh is expected to multiply the
# players tracked about fivefold, and a brand-new tracked player has
# typically earned almost none of the trophies yet (the launch-night
# refresh took tracked players 638 -> 3,868 while trophies earned went
# 943 -> 4,276, a rate sum of 147.8% -> 110.5%, i.e. each rate cut to
# about a fifth).  The one trophy everyone holds stays where it is.
DILUTION_MULT = 5.0
DILUTION_NEW_SHARE = 0.0

# Blend weights in logit space: saturating fit, last rate, log-time fit.
W_SAT, W_LAST, W_CURVE = 0.75, 0.15, 0.10

# Spread calibration from backtests: the raw spread runs a little
# wide against the median's actual error, so it is scaled down.
SD_SCALE = 0.85


def parse_ts(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_since_release(when):
    return max(0.0, (when - RELEASE).total_seconds() / 3600.0)


def log_time(when):
    return math.log1p(hours_since_release(when))


def clamp_rate(x):
    return min(99.99, max(0.01, x))


def logit_pct(x):
    p = clamp_rate(x) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(x):
    if x > 30.0:
        return 99.99
    if x < -30.0:
        return 0.01
    return 100.0 / (1.0 + math.exp(-x))


def fit_line(xs, ys):
    """Least-squares y = a + b*x; returns (a, b, residual_sd, n)."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0, 0
    if n == 1:
        return ys[0], 0.0, 0.0, 1
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0.0:
        return my, 0.0, 0.0, n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    sd = math.sqrt(sum(r * r for r in resid) / max(1, n - 2))
    return a, b, sd, n


def chapter_rank(description):
    """Story order for monotone chapter trophies: 1..4, final mission 5."""
    text = (description or "").lower()
    if "final mission" in text:
        return 5
    for k in (1, 2, 3, 4):
        if "chapter %d" % k in text:
            return k
    return None


def saturating_fit(times_h, rates, tier):
    """Fit rate(t) = C * (1 - exp(-(a + b*t))) by trying ceilings.

    For fixed C, y = -ln(1 - rate/C) is linear in t, so each ceiling
    is one least-squares fit.  Returns (forecast_rate, logit_sd) at
    the target hour, or None when nothing fits.
    """
    r_last = rates[-1]
    if r_last >= 99.99:
        return None
    cands = set()
    for c in (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 15.0, 25.0, 40.0,
              60.0, 85.0, 99.99):
        if c >= r_last * 0.98:
            cands.add(c)
    for m in (1.02, 1.1, 1.25, 1.5, 2.0, 3.0, 5.0):
        c = round(r_last * m, 4)
        if 0.05 <= c <= 99.99:
            cands.add(c)
    cands = sorted(cands)
    cap = TIER_CAP.get(tier, 99.99)
    cands = [c for c in cands if c <= cap]
    if not cands:
        return None
    prior = max(TIER_PRIOR.get(tier, 20.0), r_last)
    best = None
    for c in cands:
        xs, ys = [], []
        for t, r in zip(times_h, rates):
            if r <= 0.01 or r >= c * 0.995:
                continue
            xs.append(t)
            ys.append(-math.log(1.0 - r / c))
        if len(xs) < 3:
            continue
        a, b, _sd, _n = fit_line(xs, ys)
        if b <= 0.0:
            continue
        # Score in logit space over the sampled reads.
        sse = 0.0
        cnt = 0
        for t, r in zip(times_h, rates):
            expo = a + b * t
            pred = c * (1.0 - math.exp(-expo)) if expo > 0.0 else 0.01
            pred = clamp_rate(pred)
            d = logit_pct(r) - logit_pct(pred)
            sse += d * d
            cnt += 1
        if cnt == 0:
            continue
        score = sse / cnt + 0.02 * (math.log(c / prior)) ** 2
        if best is None or score < best[0]:
            best = (score, a, b, c)
    if best is None:
        return None
    _score, a, b, c = best
    return a, b, c


def forecast_case(case):
    trophies = case["trophies"]
    hist = case["history"]
    at = parse_ts(case["at"])
    target = parse_ts(case["target"])
    horizon_days = max(0.0, case["horizon_minutes"] / 1440.0)
    target_h = hours_since_release(target)
    u_target = log_time(target)

    # The whole history one unchanging page: Exophase has not refreshed.
    flat = len(hist) >= 2
    if flat:
        first = hist[0]
        for h in hist[1:]:
            if h["players_tracked"] != first["players_tracked"] or \
                    h["rates"] != first["rates"]:
                flat = False
                break
    span_hours = 0.0
    if hist:
        span_hours = (parse_ts(hist[-1]["at"])
                      - parse_ts(hist[0]["at"])).total_seconds() / 3600.0
    stale = flat and span_hours >= 2.0

    # Subsample the reads for the curve fits: at most ~40 points,
    # always keeping the last read.
    if len(hist) > 40:
        step = len(hist) / 40.0
        idx = sorted(set(int(i * step) for i in range(40)) | {len(hist) - 1})
        sampled = [hist[i] for i in idx]
    else:
        sampled = list(hist)

    rows = []
    for j, trophy in enumerate(trophies):
        tier = trophy.get("tier", "bronze")
        last = hist[-1]["rates"][j] if hist else 0.01
        if last is None:
            last = 0.01
        last = clamp_rate(last)

        times_h, rates = [], []
        for h in sampled:
            r = h["rates"][j]
            if r is None:
                continue
            times_h.append(hours_since_release(parse_ts(h["at"])))
            rates.append(r)
        if not rates:
            rates = [last]
            times_h = [hours_since_release(at)]

        # Saturating-curve forecast.
        sat = saturating_fit(times_h, rates, tier)
        if sat is not None:
            a, b, c = sat
            expo = a + b * target_h
            rate_sat = c * (1.0 - math.exp(-expo)) if expo > 0.0 else 0.01
            rate_sat = clamp_rate(rate_sat)
            # Residual sd of the winning fit, in logit space.
            sse = 0.0
            cnt = 0
            for t, r in zip(times_h, rates):
                expo_t = a + b * t
                pred = c * (1.0 - math.exp(-expo_t)) if expo_t > 0.0 else 0.01
                d = logit_pct(r) - logit_pct(clamp_rate(pred))
                sse += d * d
                cnt += 1
            sd_fit = math.sqrt(sse / max(1, cnt)) if cnt else 0.3
        else:
            rate_sat = last
            sd_fit = 0.3

        # Direct logit-rate against log-time fit (house curve's shape),
        # over the sampled reads.
        us = [log_time(parse_ts(h["at"])) for h in sampled
              if h["rates"][j] is not None]
        vs = [logit_pct(h["rates"][j]) for h in sampled
              if h["rates"][j] is not None]
        a_r, b_r, _sd_r, _n_r = fit_line(us, vs)
        b_r = min(2.0, max(-1.0, b_r))
        rate_curve = clamp_rate(inv_logit_pct(a_r + b_r * u_target))

        med_logit = (W_SAT * logit_pct(rate_sat)
                     + W_LAST * logit_pct(last)
                     + W_CURVE * logit_pct(rate_curve))
        median = clamp_rate(inv_logit_pct(med_logit))
        cap = TIER_CAP.get(tier, 99.99)
        median = min(median, cap)

        # Stale page: the next refresh dilutes the rate, new players
        # hold far fewer trophies; pull the median down, widen the band.
        if stale and last < 95.0:
            factor = (1.0 + DILUTION_NEW_SHARE * (DILUTION_MULT - 1.0)) / DILUTION_MULT
            median = clamp_rate(median * factor)

        # Spread: random-walk growth with the horizon, plus the
        # trophy's own fit residual, plus a stale-refresh allowance.
        sd_rw = 0.5 * math.sqrt(horizon_days)
        sd_fit = min(1.0, max(0.15, sd_fit))
        sd_stale = 0.9 if stale else 0.0
        sd = math.sqrt(sd_rw * sd_rw + sd_fit * sd_fit + sd_stale * sd_stale)
        sd = min(2.0, max(0.35, sd)) * SD_SCALE

        quantiles = []
        for z in Z:
            q = inv_logit_pct(logit_pct(median) + z * sd)
            quantiles.append(round(min(99.99, max(0.01, q)), 4))
        for k in range(1, 5):
            if quantiles[k] < quantiles[k - 1]:
                quantiles[k] = quantiles[k - 1]
        rows.append({"id": trophy["id"], "q": quantiles,
                     "median": median, "rank": chapter_rank(trophy.get("description", ""))})

    # Story trophies stay monotone: chapter 1 >= 2 >= 3 >= 4 >= final.
    story = [r for r in rows if r["rank"] is not None]
    story.sort(key=lambda r: r["rank"])
    prev = None
    for r in story:
        if prev is not None and r["median"] > prev:
            shift = logit_pct(prev) - logit_pct(r["median"])
            r["median"] = prev
            new_q = [round(min(99.99, max(0.01,
                               inv_logit_pct(logit_pct(q) + shift))), 4)
                     for q in r["q"]]
            for k in range(1, 5):
                if new_q[k] < new_q[k - 1]:
                    new_q[k] = new_q[k - 1]
            r["q"] = new_q
        prev = r["median"]

    out = {"trophies": [{"id": r["id"], "q": r["q"]} for r in rows]}
    return out


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(f for f in os.listdir(in_dir) if f.endswith(".in"))
    for name in names:
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        result = forecast_case(case)
        stem = name[:-3]
        with open(os.path.join(out_dir, stem + ".out"), "w", encoding="utf-8") as fh:
            json.dump(result, fh)
            fh.write("\n")


if __name__ == "__main__":
    main()
