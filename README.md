[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/moonku-ai/gta6-trophy-rarity-forecast-01-v2r/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/moonku-ai/gta6-trophy-rarity-forecast-01-v2r/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

# GTA VI Trophy Rarity Forecast — v2 rewrite

Entry for the [make.fun bounty](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
**"Forecast how many GTA VI players earn each trophy in week one"** ($250, top 20 paid).

`main.py` (pure Python stdlib, no network) reads one case file
(`makefun/gta6-trophies-case/v1`) from `$MAKEFUN_INPUT_DIR` (else `/input`)
and writes `/output/<id>.out`: five quantiles (5/25/50/75/95%) of each
trophy's Exophase earn rate at the case's target.

## Approach — cohort-dilution forecaster

An earn rate is a ratio, `E(t) / P(t)`, and the two sides move for
different reasons, so they are forecast separately:

- **P(t), players tracked**, grows as new players sync profiles, in
  jumps: Exophase refreshes its counts hours apart, not continuously.
  Growth to the target is extrapolated from the last two 12-hour
  increments as a decaying geometric series (bounded, never an
  exponential). A page that has *never* refreshed in its whole history
  and sat unchanged for hours is treated as a refresh waiting to
  happen, with an expected dilution jump.
- **E(t), earners of trophy i**, grows two ways: tracked players keep
  playing and close part of their remaining headroom to the trophy's
  ceiling (`(1 - difficulty)^1.2`), and newly tracked players arrive
  already holding the easy/intro trophies but almost none of the hard
  ones (newcomer profile).
- **Difficulty prior** per trophy from its tier and description (story
  chapter number, collect-everything wording, online play, secrecy),
  turned into a time-to-earn; players progress only a fraction of
  wall-clock time.
- **Quantiles** are logit-normal around the median, widening with
  horizon and refresh risk, lower tail stretched when a refresh is
  overdue (a refresh can only dilute rates down). When the history
  shows movement, the median is blended with a capped logit trend fit.

## Local evidence (vs the v1-style persistence baseline)

- Synthetic backtest, 10 seeds, final-like regime (48 h checkpoint,
  108.5 h horizon, cohort-based world with independent constants):
  **0.5504 vs 0.7472** for persistence.
- Practice-style refresh test (flat launch-night history, one ~6x
  refresh at the target): **0.9565 vs 1.6651** for persistence.
- Format/determinism: 7/7 checks pass on the real practice cases
  p00/p01; runtime ~0.06 s per case.

Run the tests with `python3 test_main.py`.

## License

MIT — see `LICENSE`.
