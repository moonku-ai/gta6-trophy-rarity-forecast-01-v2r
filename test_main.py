#!/usr/bin/env python3
"""Tests for the cohort-dilution forecaster (TDD: written before the
final tuning pass; run with `python3 test_main.py`).

Checks:
1. Output format + determinism + runtime on the real practice cases.
2. Synthetic backtest (10 seeds, final-like regime): forecaster beats
   a v1-style persistence baseline (last rate + 0.5*sqrt(days) spread).
3. Practice-style refresh test: flat launch-night history, one big
   refresh at the target; forecaster beats persistence there too.
"""

import json
import math
import os
import random
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
KIT = os.path.expanduser(
    "~/workspace/makefun/gta6-trophy-rarity-forecast-01-kit/"
    "gta6-trophy-rarity-forecast-01-kit/input")
MAIN = os.path.join(HERE, "main.py")

Z = (-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817,
     1.6448536269514722)


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def logit(p):
    p = clamp(p, 0.0001, 0.9999)
    return math.log(p / (1.0 - p))


def inv_logit(x):
    return 1.0 / (1.0 + math.exp(-x)) if -700 < x < 700 else (1.0 if x >= 0 else 0.0)


def pinball(truth, q, tau):
    d = logit(truth / 100.0) - logit(q / 100.0)
    return 2.0 * max(tau * d, (tau - 1.0) * d)


def loss_of(rows, truth):
    tot, n = 0.0, 0
    for row in rows:
        y = truth[row["id"]]
        for tau, q in zip((0.05, 0.25, 0.5, 0.75, 0.95), row["q"]):
            tot += pinball(y, q, tau)
            n += 1
    return tot / n


def v1_style_rows(case):
    """Persistence baseline in the v1 server style: median = last rate,
    spread 0.5*sqrt(horizon days) on the logit scale."""
    last = case["history"][-1]["rates"]
    days = case["horizon_minutes"] / 1440.0
    s = 0.5 * math.sqrt(days)
    rows = []
    for i, tr in enumerate(case["trophies"]):
        r = clamp(last[i], 0.01, 99.99)
        lo = logit(r / 100.0)
        rows.append({"id": tr["id"],
                     "q": [round(clamp(100 * inv_logit(lo + z * s), 0.01, 99.99), 4)
                           for z in Z]})
    return rows


def run_main(case, tmp):
    os.makedirs(os.path.join(tmp, "in"), exist_ok=True)
    os.makedirs(os.path.join(tmp, "out"), exist_ok=True)
    with open(os.path.join(tmp, "in", case["id"] + ".in"), "w") as fh:
        json.dump(case, fh)
    env = dict(os.environ, MAKEFUN_INPUT_DIR=os.path.join(tmp, "in"),
               MAKEFUN_OUTPUT_DIR=os.path.join(tmp, "out"))
    t0 = time.time()
    cp = subprocess.run([sys.executable, MAIN], env=env,
                        capture_output=True, text=True, timeout=30)
    dt = time.time() - t0
    assert cp.returncode == 0, cp.stderr
    with open(os.path.join(tmp, "out", case["id"] + ".out")) as fh:
        return json.load(fh)["trophies"], dt


# ------------------------------------------------------------- synthetic world

def make_trophies(rng):
    trs = []
    for i in range(1, 56):
        tier = "platinum" if i == 1 else ("gold" if i in (2, 3)
                                          else ("silver" if i <= 8 else "bronze"))
        chapter = rng.randint(1, 9) if tier == "bronze" else None
        desc = ("Complete Chapter %d." % chapter if chapter else
                "Complete the final mission." if tier == "gold" else
                "Congratulations! You're a Leonida Legend.")
        trs.append({"id": i, "name": "Trophy %d" % i, "description": desc,
                    "tier": tier, "points": 300 if tier == "platinum" else 90
                    if tier == "gold" else 30 if tier == "silver" else 15,
                    "secret": rng.random() < 0.5})
    return trs


def simulate(rng, trs, checkpoint_h, horizon_h, refresh_every_h=5.0,
             jump=5.0, base_players=638):
    """Cohort-based synthetic world (independent constants from the
    forecaster).  Players arrive over time; a cohort arriving at time s
    has exposure h-s at time h and earns trophy j with probability
    F_j(exposure) = (1-d)^1.3 * (1-exp(-play_time/T_j)).  Earners are
    the cohort integral, so newcomers genuinely dilute rates.  The
    page shows the state as of its last refresh before each read."""
    diff = [clamp(rng.betavariate(2.0, 2.0) * 0.9 + 0.05, 0.05, 0.98)
            for _ in trs]
    t_earn = [8.0 + 100.0 * d * d for d in diff]
    cap = base_players * rng.uniform(25, 60)
    lam = rng.uniform(0.015, 0.03)
    play = 0.25

    def players_at(h):
        return base_players + (cap - base_players) * (1 - math.exp(-lam * h))

    def earn_prob(d, t_j, exposure_h):
        return clamp((1 - d) ** 1.3 * (1 - math.exp(-play * exposure_h / t_j)),
                     0.0, 1.0)

    def state_at(h):
        # Cohort integral, 0.5 h steps, plus the launch cohort at h=0.
        p = players_at(h)
        earners = [0.0] * len(trs)
        prev_p = base_players
        s = 0.0
        while s < h:
            s2 = min(h, s + 0.5)
            cohort = players_at(s2) - players_at(s)
            mid = (s + s2) / 2.0
            for j, d in enumerate(diff):
                earners[j] += cohort * earn_prob(d, t_earn[j], h - mid)
            s = s2
        for j, d in enumerate(diff):
            earners[j] += base_players * earn_prob(d, t_earn[j], h)
        return p, [100.0 * e / p for e in earners]

    def shown_at(h):
        last_refresh = math.floor(h / refresh_every_h) * refresh_every_h
        if last_refresh <= 0:
            p = base_players
            rates = [100.0 * earn_prob(diff[j], t_earn[j], 0.5)
                     for j in range(len(trs))]
            return p, rates
        return state_at(last_refresh)

    hist = []
    t = 0.0
    while t < checkpoint_h:
        p, rates = shown_at(t)
        hist.append({"slot": "2026-11-19T%02d:%02d:00Z" % (int(t // 60) % 24, int(t % 60)),
                     "at": "2026-11-19T%02d:%02d:00.000Z" % (int(t // 60) % 24, int(t % 60)),
                     "players_tracked": int(p),
                     "trophies_earned": int(sum(r / 100.0 * p for r in rates)),
                     "rates": [round(r, 2) for r in rates]})
        t += 0.5
    target_h = checkpoint_h + horizon_h
    p_t, rates_t = shown_at(target_h)
    case = {"id": "synth", "at": "2026-11-19T%02d:%02d:00Z" % (int(checkpoint_h // 60) % 24, int(checkpoint_h % 60)),
            "horizon_minutes": int(horizon_h * 60),
            "target": "2026-11-26T00:00:00Z",
            "levels": [0.05, 0.25, 0.5, 0.75, 0.95],
            "game": {"name": "Grand Theft Auto VI", "platform": "PS5",
                     "release_date": "2026-11-19", "source": "exophase"},
            "trophies": trs, "history": hist, "notes": "synthetic"}
    return case, {trs[j]["id"]: rates_t[j] for j in range(len(trs))}


def test_format_and_determinism():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        for cid in ("p00", "p01"):
            with open(os.path.join(KIT, cid + ".in")) as fh:
                case = json.load(fh)
            rows1, dt = run_main(case, tmp)
            rows2, _ = run_main(case, tmp)
            assert rows1 == rows2, cid + " not deterministic"
            assert dt < 8.0, cid + " too slow: %.2fs" % dt
            assert len(rows1) == 55
            assert {r["id"] for r in rows1} == set(range(1, 56))
            for r in rows1:
                assert len(r["q"]) == 5 and set(r.keys()) == {"id", "q"}
                assert all(0.0 <= q <= 100.0 for q in r["q"])
                assert r["q"] == sorted(r["q"])
            print("  %s ok (%.2fs, median trophy 3: %s)" % (cid, dt, rows1[2]["q"]))
    print("PASS format/determinism")


def test_synthetic_backtest():
    import tempfile
    mine, base, curve = [], [], []
    with tempfile.TemporaryDirectory() as tmp:
        for seed in range(10):
            rng = random.Random(1000 + seed)
            trs = make_trophies(rng)
            case, truth = simulate(rng, trs, checkpoint_h=48.0, horizon_h=108.5)
            rows, _ = run_main(case, tmp)
            mine.append(loss_of(rows, truth))
            base.append(loss_of(v1_style_rows(case), truth))
    print("  synthetic 10 seeds: mine %.4f  v1-style %.4f" % (sum(mine) / 10, sum(base) / 10))
    print("  per-seed mine:", ["%.3f" % x for x in mine])
    print("  per-seed v1 :", ["%.3f" % x for x in base])
    assert sum(mine) / 10 < sum(base) / 10, "forecaster must beat v1-style"
    print("PASS synthetic backtest")


def test_practice_refresh():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        rng = random.Random(7)
        trs = make_trophies(rng)
        # Flat launch-night history (p00 shape), then a 6x refresh at target.
        case, truth = simulate(rng, trs, checkpoint_h=3.0, horizon_h=1.5,
                               refresh_every_h=4.5, jump=6.0)
        # Force the flat history the practice actually has: all reads identical.
        first = case["history"][0]
        for h in case["history"]:
            h["rates"] = list(first["rates"])
            h["players_tracked"] = first["players_tracked"]
        rows, _ = run_main(case, tmp)
        lm, lb = loss_of(rows, truth), loss_of(v1_style_rows(case), truth)
        print("  practice-refresh: mine %.4f  v1-style %.4f" % (lm, lb))
        assert lm < lb, "forecaster must beat persistence on refresh"
    print("PASS practice refresh")


if __name__ == "__main__":
    test_format_and_determinism()
    test_synthetic_backtest()
    test_practice_refresh()
    print("ALL TESTS PASSED")
