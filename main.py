#!/usr/bin/env python3
"""Forecast GTA VI trophy earn rates on Exophase at a case's target.

Cohort-dilution forecaster (written from scratch, stdlib only).

Idea: an earn rate is a ratio, E(t) / P(t).  The numerator and the
denominator move for different reasons, so forecast them separately:

* P(t), players tracked, grows as new players sync profiles.  Exophase
  refreshes its counts in jumps, hours apart, not continuously, so a
  long-unchanged page is a page whose next refresh is overdue.
* E(t), earners of trophy i, grows two ways: players already tracked
  keep playing and earn it (progression), and newly tracked players
  arrive already holding the easy/intro trophies but almost none of
  the hard ones (newcomer profile).

Per trophy we build a difficulty prior from its tier and description
(story chapter number, collect-everything wording, online play, ...),
turn it into a time-to-earn, and combine progression + newcomer
dilution into a median rate.  Quantiles come from a logit-normal
spread that widens with horizon and with refresh risk, skewed down
when a refresh is overdue (a refresh can only dilute rates down,
except the 100% trophy).

Deterministic: no randomness, no network, stdlib only.
"""

import json
import math
import os
import re
import sys
from datetime import datetime, timezone

Z = (-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817,
     1.6448536269514722)

# Share of wall-clock time a typical tracked player actually spends
# progressing through the game (the rest is idle / offline / other
# games).  Progression and newcomer exposure scale with this, not
# with raw hours.
PLAY_FRACTION = 0.30


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def logit(p):
    p = clamp(p, 0.0001, 0.9999)
    return math.log(p / (1.0 - p))


def inv_logit(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x)) if x < 700 else 1.0
    ex = math.exp(x)
    return ex / (1.0 + ex)


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


# ---------------------------------------------------------------- difficulty

TIER_BASE = {"platinum": 0.97, "gold": 0.80, "silver": 0.58, "bronze": 0.38}

HARD_WORDS = ("collect", "every", "all ", "100%", "complete the game",
               "platinum", "master", "legendary", "without dying",
               "without being detected", "s rank", "gold medal")
EASY_WORDS = ("first", "tutorial", "intro", "begin", "open ", "start ",
               "reach ", "visit ", "meet ")
ONLINE_WORDS = ("online", "multiplayer", "compete", "against another",
                 "win a race", "leaderboard")


def difficulty(trophy):
    """Prior difficulty in [0.02, 0.99]: 0 = every player earns it fast."""
    desc = (trophy.get("description") or "").lower()
    name = (trophy.get("name") or "").lower()
    text = name + " " + desc
    d = TIER_BASE.get(trophy.get("tier"), 0.4)
    m = re.search(r"chapter\s+(\d+)", text)
    if m:
        # Story chapters get harder / later the further in they sit.
        d = 0.15 + 0.09 * int(m.group(1))
    if "final mission" in text or "complete the game" in text:
        d = max(d, 0.85)
    for w in HARD_WORDS:
        if w in text:
            d += 0.10
    for w in EASY_WORDS:
        if w in text:
            d -= 0.08
    for w in ONLINE_WORDS:
        if w in text:
            d += 0.08
    if trophy.get("secret"):
        d += 0.04
    return clamp(d, 0.02, 0.99)


def time_to_earn_hours(d):
    """Median hours a typical active player needs for a trophy of
    difficulty d, once they start playing."""
    return 4.0 + 130.0 * d * d


# ------------------------------------------------------------------- history

def last_rates_and_players(case):
    hist = case.get("history") or []
    if not hist:
        return None, None
    last = hist[-1]
    return list(last.get("rates") or []), last.get("players_tracked")


def hours_between(a, b):
    return (parse_time(b) - parse_time(a)).total_seconds() / 3600.0


def estimate_player_growth(case, horizon_h):
    """Players-tracked growth to the target, from the increments of
    the last two 12 h windows.  Increments shrink as the tracked base
    saturates, so extrapolate them as a decaying geometric series
    (bounded), never as an exponential fitted to a rising curve.
    Returns (growth_factor, any_variation_seen)."""
    hist = case.get("history") or []
    if len(hist) < 3:
        return 1.0, False
    at0 = parse_time(case["at"])
    pts = [(parse_time(h["at"]), h.get("players_tracked"))
           for h in hist if h.get("players_tracked")]
    if len(pts) < 3:
        return 1.0, False
    varied = len({p for _, p in pts}) > 1
    if not varied:
        return 1.0, False

    def players_near(hours_ago):
        target = at0.timestamp() - hours_ago * 3600.0
        best = min(pts, key=lambda tp: abs(tp[0].timestamp() - target))
        return best[1]

    p_last = pts[-1][1]
    p_12 = players_near(12.0)
    p_24 = players_near(24.0)
    d_recent = p_last - p_12
    d_prev = p_12 - p_24
    if d_recent <= 0:
        return 1.0, True
    if d_prev <= 0:
        # Still accelerating: one more window of the recent increment.
        extra = d_recent * clamp(horizon_h / 12.0, 0.0, 2.0)
        return clamp((p_last + extra) / p_last, 1.0, 4.0), True
    q = clamp(d_recent / d_prev, 0.0, 0.95)
    steps = horizon_h / 12.0
    extra = d_recent * q * (1.0 - q ** steps) / (1.0 - q) if q > 0 else 0.0
    return clamp((p_last + extra) / p_last, 1.0, 4.0), True


def staleness_hours(case):
    """Hours the last read's rates have been unchanged (0 if the
    history shows movement)."""
    hist = case.get("history") or []
    if len(hist) < 2:
        return 0.0
    last_rates = hist[-1].get("rates")
    i = len(hist) - 1
    while i > 0 and hist[i - 1].get("rates") == last_rates:
        i -= 1
    if i == 0 and hist[0].get("rates") == last_rates:
        return hours_between(hist[0]["at"], case["at"])
    return hours_between(hist[i]["at"], case["at"])


def never_changed(case):
    """True when every read in the history shows identical rates:
    the page has never refreshed since the recording began."""
    hist = case.get("history") or []
    if len(hist) < 2:
        return False
    first = hist[0].get("rates")
    return all(h.get("rates") == first for h in hist)


def logit_trend(case, idx):
    """Least-squares slope of trophy idx's logit rate against
    ln(1 + hours since release) over the last 24 h; None if flat."""
    hist = case.get("history") or []
    if len(hist) < 3:
        return None
    release = datetime(2026, 11, 19, tzinfo=timezone.utc)
    xs, ys = [], []
    for h in hist:
        rates = h.get("rates") or []
        if idx >= len(rates) or rates[idx] is None:
            continue
        t = parse_time(h["at"])
        if (parse_time(case["at"]) - t).total_seconds() > 24.5 * 3600:
            continue
        xs.append(math.log1p(max(0.0, (t - release).total_seconds() / 3600.0)))
        ys.append(logit(rates[idx] / 100.0))
    if len(xs) < 3 or len(set(round(y, 6) for y in ys)) < 2:
        return None
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    if den <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den


# ------------------------------------------------------------------ forecast

def forecast_trophy(case, trophy, idx, last_rate, p_last, p_target,
                    horizon_h, refresh_risk, trend):
    d = difficulty(trophy)
    t_earn = time_to_earn_hours(d)
    r = clamp(last_rate if last_rate is not None else 0.0, 0.0, 100.0)
    if r >= 99.99:
        return [100.0] * 5
    if r <= 0.0:
        # Never seen earned: only newcomers / fresh progression can move it.
        med = 100.0 * (1.0 - d) * 0.02 * (1.0 - math.exp(-horizon_h * PLAY_FRACTION / t_earn))
        med = clamp(med, 0.01, 5.0)
        return quantiles(med, 0.9, 0.0)

    e_last = r / 100.0 * p_last
    # Ceiling: the share of players who will ever earn this trophy.
    ceil = (1.0 - d) ** 1.2
    # Players only progress during a fraction of wall-clock time.
    play_h = horizon_h * PLAY_FRACTION
    # Existing tracked players keep earning: the cohort closes a
    # fraction (1 - exp(-play/T)) of its remaining headroom to the
    # ceiling (F is concave in total exposure, so headroom, not all
    # non-earners, is what can still be earned).
    prog = 1.0 - math.exp(-play_h / t_earn)
    e_prog = max(0.0, p_last * ceil - e_last) * prog
    # Newcomers: arrive over the horizon, average exposure horizon/2.
    newcomers = max(0.0, p_target - p_last)
    new_rate = ceil * 0.95 * (1.0 - math.exp(-play_h / 2.0 / t_earn))
    new_rate += 0.005 * (1.0 - d)
    e_new = newcomers * clamp(new_rate, 0.0, 1.0)
    e_med = e_last + e_prog + e_new
    med = clamp(100.0 * e_med / p_target, 0.01, 99.99)

    if trend is not None:
        # Blend the cohort median with the fitted logit trend.  The
        # trend slope is capped: near-zero early rates make raw logit
        # slopes explode, and a slope fitted on launch-night data
        # must not be extrapolated without limit.
        trend = clamp(trend, -1.2, 1.2)
        release = datetime(2026, 11, 19, tzinfo=timezone.utc)
        x_now = math.log1p(max(0.0, (parse_time(case["at"]) - release).total_seconds() / 3600.0))
        x_tgt = math.log1p(max(0.0, (parse_time(case["target"]) - release).total_seconds() / 3600.0))
        shift = clamp(trend * (x_tgt - x_now), -1.2, 1.2)
        m_trend = inv_logit(logit(r / 100.0) + shift)
        med = 100.0 * inv_logit(0.6 * logit(med / 100.0) + 0.4 * logit(m_trend))

    days = horizon_h / 24.0
    spread = 0.28 + 0.35 * math.sqrt(days) + 0.30 * refresh_risk + 0.18 * d
    spread = clamp(spread, 0.30, 1.60)
    return quantiles(med, spread, refresh_risk)


def quantiles(med, spread, refresh_risk):
    """Logit-normal quantiles, lower tail stretched when a refresh is
    overdue (a refresh dilutes rates down)."""
    lo = logit(med / 100.0)
    out = []
    for z in Z:
        s = spread * (1.0 + 0.55 * refresh_risk) if z < 0 else spread
        out.append(round(clamp(100.0 * inv_logit(lo + z * s), 0.01, 99.99), 4))
    for i in range(1, 5):
        if out[i] < out[i - 1]:
            out[i] = out[i - 1]
    return out


def forecast_case(case):
    trophies = case["trophies"]
    rates, p_last = last_rates_and_players(case)
    horizon_h = case.get("horizon_minutes", 0) / 60.0
    if p_last is None or not rates:
        p_last = 638.0
        rates = [0.0] * len(trophies)
    growth, _varied = estimate_player_growth(case, horizon_h)
    stale = staleness_hours(case)
    # A page that has NEVER refreshed in its whole history, and has
    # sat unchanged for hours, is a refresh waiting to happen.  Once
    # the history shows refreshes, player growth (above) already
    # carries the dilution and this branch must not fire.
    refresh_risk = 0.0
    if stale >= 2.0 and never_changed(case):
        refresh_risk = clamp(1.0 - math.exp(-horizon_h / 5.0), 0.0, 0.9)
    p_target = p_last * growth
    if refresh_risk > 0:
        # Expected dilution from the overdue refresh (jump ~6x when it
        # lands, per the launch-night reading in the rules).
        p_target = p_last * growth * (1.0 + 5.0 * refresh_risk)
    rows = []
    for i, tr in enumerate(trophies):
        last = rates[i] if i < len(rates) else None
        trend = logit_trend(case, i) if _varied else None
        rows.append({"id": tr["id"],
                     "q": forecast_trophy(case, tr, i, last, p_last,
                                          p_target, horizon_h,
                                          refresh_risk, trend)})
    return {"trophies": rows}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    for name in sorted(os.listdir(in_dir)):
        if not name.endswith(".in"):
            continue
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        out = forecast_case(case)
        stem = name[:-3]
        with open(os.path.join(out_dir, stem + ".out"), "w",
                  encoding="utf-8") as fh:
            json.dump(out, fh, separators=(",", ":"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
