# Phase 3B ablation report

Derived from Jeff Sackmann's ATP data (CC BY-NC-SA 4.0); aggregate statistics only, no row-level data.

Each line compares two registered experiments on identical matches (ΔLL = new − reference, weekly-block bootstrap, 95% percentile interval, unadjusted). Each experiment uses its own C chosen on development data. Exploratory rows are shown for transparency only; no decision used them.

## 1. Development years

| question | comparison (new vs reference) | rows | blocks | ΔLL | 95% CI | ΔBrier | 95% CI | P*(ΔLL<0) | verdict |
|---|---|---|---|---|---|---|---|---|---|
| Does refitting a slope on the Elo logit help? | M0R vs M0 | 24622 | 322 | +0.00045 | [-0.00011, +0.00096] | +0.00016 | [-0.00002, +0.00034] | 0.054 | inconclusive (interval includes 0) |
| Overall shrunk serve + return rates on top of rescaled Elo | M1 vs M0R | 24622 | 322 | +0.00050 | [+0.00005, +0.00097] | +0.00026 | [+0.00006, +0.00047] | 0.015 | candidate worse |
| Serve rate only | A1_SPW vs M0R | 24622 | 322 | +0.00017 | [-0.00019, +0.00058] | +0.00009 | [-0.00007, +0.00027] | 0.189 | inconclusive (interval includes 0) |
| Return rate only | A1_RPW vs M0R | 24622 | 322 | -0.00009 | [-0.00017, -0.00001] | -0.00004 | [-0.00008, -0.00000] | 0.982 | candidate better |
| Adding surface-specific rates | M2 vs M1 | 24622 | 322 | -0.00108 | [-0.00177, -0.00042] | -0.00049 | [-0.00079, -0.00020] | 0.998 | candidate better |
| Surface-specific rates only | A2_SURF vs M0R | 24622 | 322 | +0.00008 | [-0.00076, +0.00095] | +0.00008 | [-0.00031, +0.00048] | 0.429 | inconclusive (interval includes 0) |
| Opponent-adjusted strengths + matchup | M3 vs M0R | 24622 | 322 | -0.00318 | [-0.00452, -0.00182] | -0.00135 | [-0.00195, -0.00075] | 1.000 | candidate better |
| Contribution of the matchup feature | M3 vs A3_NOMATCHUP | 24622 | 322 | -0.00170 | [-0.00268, -0.00072] | -0.00076 | [-0.00120, -0.00032] | 1.000 | candidate better |
| Opponent-adjusted strengths without matchup | A3_NOMATCHUP vs M0R | 24622 | 322 | -0.00148 | [-0.00250, -0.00048] | -0.00059 | [-0.00104, -0.00014] | 0.998 | candidate better |
| Matchup expected serve-point edge only | A3_MATCHUP vs M0R | 24622 | 322 | -0.00317 | [-0.00453, -0.00183] | -0.00134 | [-0.00195, -0.00075] | 1.000 | candidate better |
| Adding 52-week recency-weighted rates | M4A vs M1 | 24622 | 322 | -0.00725 | [-0.00935, -0.00535] | -0.00331 | [-0.00423, -0.00245] | 1.000 | candidate better |
| Adding reliability/activity features | M4B vs M1 | 24622 | 322 | -0.00751 | [-0.00946, -0.00561] | -0.00334 | [-0.00422, -0.00249] | 1.000 | candidate better |
| Post-hoc: recency-weighted rates only | A4_DECAY vs M0R | 24622 | 322 | -0.00674 | [-0.00869, -0.00494] | -0.00305 | [-0.00392, -0.00225] | 1.000 | candidate better |
| Post-hoc: recent activity (log ESS) only | A4_ESS vs M0R | 24622 | 322 | -0.00445 | [-0.00599, -0.00293] | -0.00208 | [-0.00276, -0.00142] | 1.000 | candidate better |
| Do career shrunk rates add anything to recency-weighted ones? | M4A vs A4_DECAY | 24622 | 322 | -0.00001 | [-0.00012, +0.00011] | +0.00000 | [-0.00004, +0.00005] | 0.550 | inconclusive (interval includes 0) |
| Serve/return rates without Elo, against Elo | A0_NOELO vs M0 | 24622 | 322 | +0.03879 | [+0.03391, +0.04360] | +0.01705 | [+0.01492, +0.01918] | 0.000 | candidate worse |

## 2. Exploratory years (2010–2026, not used for any decision)

| question | comparison (new vs reference) | rows | blocks | ΔLL | 95% CI | ΔBrier | 95% CI | P*(ΔLL<0) | verdict |
|---|---|---|---|---|---|---|---|---|---|
| Does refitting a slope on the Elo logit help? | M0R vs M0 | 45191 | 645 | -0.00055 | [-0.00084, -0.00027] | -0.00017 | [-0.00027, -0.00008] | 1.000 | candidate better |
| Overall shrunk serve + return rates on top of rescaled Elo | M1 vs M0R | 45191 | 645 | -0.00002 | [-0.00024, +0.00019] | +0.00001 | [-0.00009, +0.00010] | 0.589 | inconclusive (interval includes 0) |
| Serve rate only | A1_SPW vs M0R | 45191 | 645 | -0.00007 | [-0.00028, +0.00014] | -0.00001 | [-0.00011, +0.00008] | 0.732 | inconclusive (interval includes 0) |
| Return rate only | A1_RPW vs M0R | 45191 | 645 | -0.00010 | [-0.00024, +0.00002] | -0.00005 | [-0.00011, +0.00001] | 0.936 | inconclusive (interval includes 0) |
| Adding surface-specific rates | M2 vs M1 | 45191 | 645 | -0.00043 | [-0.00080, -0.00006] | -0.00019 | [-0.00035, -0.00002] | 0.989 | candidate better |
| Surface-specific rates only | A2_SURF vs M0R | 45191 | 645 | -0.00004 | [-0.00050, +0.00040] | +0.00002 | [-0.00019, +0.00022] | 0.586 | inconclusive (interval includes 0) |
| Opponent-adjusted strengths + matchup | M3 vs M0R | 45191 | 645 | -0.00233 | [-0.00317, -0.00146] | -0.00102 | [-0.00140, -0.00064] | 1.000 | candidate better |
| Contribution of the matchup feature | M3 vs A3_NOMATCHUP | 45191 | 645 | -0.00099 | [-0.00154, -0.00047] | -0.00045 | [-0.00068, -0.00022] | 1.000 | candidate better |
| Opponent-adjusted strengths without matchup | A3_NOMATCHUP vs M0R | 45191 | 645 | -0.00134 | [-0.00193, -0.00073] | -0.00058 | [-0.00084, -0.00032] | 1.000 | candidate better |
| Matchup expected serve-point edge only | A3_MATCHUP vs M0R | 45191 | 645 | -0.00232 | [-0.00314, -0.00147] | -0.00102 | [-0.00139, -0.00064] | 1.000 | candidate better |
| Adding 52-week recency-weighted rates | M4A vs M1 | 45191 | 645 | -0.00484 | [-0.00608, -0.00366] | -0.00214 | [-0.00269, -0.00164] | 1.000 | candidate better |
| Adding reliability/activity features | M4B vs M1 | 45191 | 645 | -0.00658 | [-0.00796, -0.00531] | -0.00291 | [-0.00351, -0.00236] | 1.000 | candidate better |
| Post-hoc: recency-weighted rates only | A4_DECAY vs M0R | 45191 | 645 | -0.00488 | [-0.00607, -0.00370] | -0.00214 | [-0.00267, -0.00164] | 1.000 | candidate better |
| Post-hoc: recent activity (log ESS) only | A4_ESS vs M0R | 45191 | 645 | -0.00345 | [-0.00461, -0.00238] | -0.00157 | [-0.00206, -0.00110] | 1.000 | candidate better |
| Do career shrunk rates add anything to recency-weighted ones? | M4A vs A4_DECAY | 45191 | 645 | +0.00002 | [-0.00005, +0.00009] | +0.00000 | [-0.00003, +0.00004] | 0.302 | inconclusive (interval includes 0) |
| Serve/return rates without Elo, against Elo | A0_NOELO vs M0 | 45191 | 645 | +0.03303 | [+0.02962, +0.03653] | +0.01450 | [+0.01301, +0.01603] | 0.000 | candidate worse |

## 3. Reading

- **Career shrunk rates add nothing to Elo.** M1 vs M0R: +0.00050 [+0.00005, +0.00097]; serve-only and return-only differences are negligible either way (+0.00017 [-0.00019, +0.00058], -0.00009 [-0.00017, -0.00001]). Long-run serve/return levels are largely already in Elo.
- **Surface-specific rates:** +0.00008 [-0.00076, +0.00095] alone; adding them to M1: -0.00108 [-0.00177, -0.00042].
- **Opponent adjustment and matchup:** M3 vs M0R -0.00318 [-0.00452, -0.00182]; the matchup feature alone carries most of it (-0.00317 [-0.00453, -0.00183]; M3 vs A3_NOMATCHUP -0.00170 [-0.00268, -0.00072]).
- **Recency is where the information is.** M4A vs M1 -0.00725 [-0.00935, -0.00535]; recency-weighted rates alone (post-hoc A4_DECAY) -0.00674 [-0.00869, -0.00494], and career rates add nothing beyond them (-0.00001 [-0.00012, +0.00011]). Recent activity alone (A4_ESS, post-hoc) -0.00445 [-0.00599, -0.00293]: part of the recency signal is how much a player has played lately (Elo has no notion of inactivity or form).
- **Without Elo** the serve/return rates are much worse than Elo (+0.03879 [+0.03391, +0.04360]).
- These are development-period statements with unadjusted intervals; A4_* were defined after the grid was first seen.
