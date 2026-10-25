# Model card: M4A+raw (Phase 3B, `m3b-1+engine1`)

Derived from Jeff Sackmann's ATP data (CC BY-NC-SA 4.0); aggregate statistics only, no row-level data.

## Purpose

Pre-match probability that player A wins an ATP tour-level singles match, for research. Not for betting or any decision with financial consequences.

## Model

- Logistic regression without intercept, L2 (C = 10.0), calibration `raw`.
- Inputs (A − B differences): `elo_logit` — log(p/(1-p)) of the Phase 2 surface-Elo probability, p clipped to [eps, 1-eps]; `spw_shrunk_diff` — a_spw_shrunk - b_spw_shrunk; `rpw_shrunk_diff` — a_rpw_shrunk - b_rpw_shrunk; `spw_decay_diff` — 52-week decay-weighted SPW, A - B; `rpw_decay_diff` — 52-week decay-weighted RPW, A - B; `decay_missing_diff` — 1[A has no decayed history] - 1[B has none]
- Production coefficients (original units): `elo_logit` +0.7508, `spw_shrunk_diff` -0.1944, `rpw_shrunk_diff` -0.8708, `spw_decay_diff` +4.3836, `rpw_decay_diff` +3.7954, `decay_missing_diff` -0.9047
- Production fit: labels released before batch 2026-06-06 (109,027 matches since 1991-01-01); frozen hash `bc4e5c870c6e21b9`; configuration hash `5e72010100065889`.
- Symmetric in player order by construction (antisymmetric inputs, no intercept, symmetric calibration).

## Performance (log loss; Δ vs Phase 2 surface Elo on identical matches, weekly-block bootstrap 95% CI)

| period | status | matches | LL model | LL surface Elo | ΔLL | 95% CI | ΔBrier |
|---|---|---|---|---|---|---|---|
| 2002–2009 | development (selection) | 24,622 | 0.59512 | 0.60142 | -0.00630 | [-0.00833, -0.00439] | -0.00289 |
| 2010–2026 | exploratory (seen in Phase 2) | 45,191 | 0.59861 | 0.60403 | -0.00541 | [-0.00662, -0.00423] | -0.00231 |

Neither period is an independent test. The prospective protocol is in `exploratory_report.md` §9.

Development: **Where the gain comes from.** A large part of the improvement is a cold-start correction: when a player has little or no released serve/return history (often Davis Cup or newly arrived players, whom Elo starts at the initial rating) the recency inputs and their missingness indicator carry most information. sparse (<1000 prior service points, either player): 5,323 matches (22%), ΔLL -0.02374 [-0.03018, -0.01718], 81% of the total log-loss reduction; substantial (>=1000 for both): 19,299 matches (78%), ΔLL -0.00149 [-0.00262, -0.00038], 19% of the total log-loss reduction. The improvement on matches between players with substantial history is the more relevant test of serve/return skill information and is smaller.

Exploratory: **Where the gain comes from.** A large part of the improvement is a cold-start correction: when a player has little or no released serve/return history (often Davis Cup or newly arrived players, whom Elo starts at the initial rating) the recency inputs and their missingness indicator carry most information. sparse (<1000 prior service points, either player): 8,388 matches (19%), ΔLL -0.01585 [-0.02105, -0.01053], 54% of the total log-loss reduction; substantial (>=1000 for both): 36,803 matches (81%), ΔLL -0.00303 [-0.00375, -0.00234], 46% of the total log-loss reduction. The improvement on matches between players with substantial history is the more relevant test of serve/return skill information and is smaller.

## Limitations

- No match dates in the source: chronology is weekly; recency is quantised to weeks.
- Development period overlaps the Phase 2 Elo tuning period; exploratory period was seen in Phase 2.
- Block bootstrap ignores dependence through shared players across weeks; intervals may be too narrow.
- Tour-level ATP men only, 1991 onward for statistics; no qualifying/challenger matches.
- Missing feature differences are 0 (neutral); players without recent statistics are treated as average relative to their opponent on that input, with a missingness indicator for the recency rates.
- Data license CC BY-NC-SA 4.0 (non-commercial).
