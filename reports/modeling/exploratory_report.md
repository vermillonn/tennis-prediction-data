# Phase 3B exploratory report: frozen model on 2010–2026

Derived from Jeff Sackmann's ATP data (CC BY-NC-SA 4.0); aggregate statistics only, no row-level data.

**Read this first.** 2010–2026 is *not* an untouched holdout: Phase 2 reported surface-Elo results on it. Everything below was computed only after the development selection was frozen (`data/modeling/models/frozen_model.json`, hash `bc4e5c870c6e21b9`; the code verifies the hash and that the file did not change during this evaluation). No parameter, feature, C or calibration method was changed after these numbers were seen. They are exploratory, not a final test.

## 1. Frozen specification

- Model `M4A+raw`: inputs `elo_logit`, `spw_shrunk_diff`, `rpw_shrunk_diff`, `spw_decay_diff`, `rpw_decay_diff`, `decay_missing_diff`; C = 10.0; calibration `raw`.
- Refitted every year with the frozen specification (coefficients re-estimated on the labels released before the first batch of each year). `M4A+raw (static 2010 fit)` instead applies the 2010-cutoff coefficients to every exploratory year without refitting.
- 45,191 matches in 645 weekly batches (2010–2026; the source ends in 2026-05-23).

**Where the gain comes from.** A large part of the improvement is a cold-start correction: when a player has little or no released serve/return history (often Davis Cup or newly arrived players, whom Elo starts at the initial rating) the recency inputs and their missingness indicator carry most information. sparse (<1000 prior service points, either player): 8,388 matches (19%), ΔLL -0.01585 [-0.02105, -0.01053], 54% of the total log-loss reduction; substantial (>=1000 for both): 36,803 matches (81%), ΔLL -0.00303 [-0.00375, -0.00234], 46% of the total log-loss reduction. The improvement on matches between players with substantial history is the more relevant test of serve/return skill information and is smaller.

## 2. Primary comparison

| model | inputs | log loss | Brier | accuracy | ECE | ΔLL vs M0 | 95% CI | Bonferroni CI | ΔBrier | 95% CI (Brier) | ΔLL verdict (95%) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| M0 | 0 | 0.60403 | 0.20899 | 0.6648 | 0.0130 | – | – | – | – | – | baseline |
| M0R | 1 | 0.60347 | 0.20882 | 0.6648 | 0.0052 | -0.00055 | [-0.00084, -0.00027] | [-0.00098, -0.00011] | -0.00017 | [-0.00027, -0.00008] | candidate better |
| M4A+raw | 6 | 0.59861 | 0.20668 | 0.6702 | 0.0064 | -0.00541 | [-0.00662, -0.00423] | [-0.00730, -0.00361] | -0.00231 | [-0.00284, -0.00180] | candidate better |
| M4A+raw (static 2010 fit) | 6 | 0.59871 | 0.20671 | 0.6705 | 0.0060 | -0.00532 | [-0.00654, -0.00413] | [-0.00722, -0.00353] | -0.00228 | [-0.00281, -0.00177] | candidate better |

Uncertainty: paired cluster bootstrap of ΔLL = LL(candidate) − LL(M0) on identical matches; 2000 resamples of whole weekly prediction batches (seed 20261025), percentile intervals. Negative Δ favours the candidate. 'Bonferroni CI' uses α = 0.05/14 for the 14 registered experiments compared with M0. An interval that includes 0 is reported as inconclusive.

## 3. Other registered experiments (frozen per-experiment C; reported for completeness, not used for any decision)

| model | inputs | log loss | Brier | accuracy | ECE | ΔLL vs M0 | 95% CI | Bonferroni CI | ΔBrier | 95% CI (Brier) | ΔLL verdict (95%) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| M1 | 3 | 0.60345 | 0.20882 | 0.6649 | 0.0059 | -0.00057 | [-0.00096, -0.00021] | [-0.00117, -0.00004] | -0.00017 | [-0.00031, -0.00003] | candidate better |
| M2 | 5 | 0.60302 | 0.20864 | 0.6664 | 0.0053 | -0.00100 | [-0.00158, -0.00046] | [-0.00186, -0.00022] | -0.00035 | [-0.00059, -0.00013] | candidate better |
| M3 | 4 | 0.60114 | 0.20779 | 0.6669 | 0.0079 | -0.00288 | [-0.00376, -0.00197] | [-0.00417, -0.00154] | -0.00120 | [-0.00158, -0.00080] | candidate better |
| M4B | 7 | 0.59687 | 0.20592 | 0.6723 | 0.0052 | -0.00715 | [-0.00853, -0.00586] | [-0.00928, -0.00523] | -0.00307 | [-0.00366, -0.00253] | candidate better |
| A1_SPW | 2 | 0.60341 | 0.20880 | 0.6651 | 0.0058 | -0.00062 | [-0.00100, -0.00026] | [-0.00120, -0.00010] | -0.00019 | [-0.00033, -0.00006] | candidate better |
| A1_RPW | 2 | 0.60337 | 0.20877 | 0.6650 | 0.0056 | -0.00065 | [-0.00097, -0.00034] | [-0.00112, -0.00019] | -0.00022 | [-0.00033, -0.00012] | candidate better |
| A2_SURF | 3 | 0.60343 | 0.20883 | 0.6648 | 0.0055 | -0.00060 | [-0.00118, -0.00002] | [-0.00142, +0.00024] | -0.00016 | [-0.00040, +0.00008] | candidate better |
| A3_NOMATCHUP | 3 | 0.60213 | 0.20824 | 0.6668 | 0.0056 | -0.00189 | [-0.00252, -0.00125] | [-0.00284, -0.00087] | -0.00075 | [-0.00103, -0.00048] | candidate better |
| A3_MATCHUP | 2 | 0.60115 | 0.20780 | 0.6669 | 0.0084 | -0.00287 | [-0.00373, -0.00199] | [-0.00415, -0.00151] | -0.00120 | [-0.00158, -0.00080] | candidate better |
| A0_NOELO | 2 | 0.63706 | 0.22349 | 0.6307 | 0.0030 | +0.03303 | [+0.02962, +0.03653] | [+0.02801, +0.03794] | +0.01450 | [+0.01301, +0.01603] | candidate worse |
| A4_DECAY | 4 | 0.59859 | 0.20668 | 0.6704 | 0.0066 | -0.00543 | [-0.00665, -0.00425] | [-0.00732, -0.00369] | -0.00231 | [-0.00284, -0.00180] | candidate better |
| A4_ESS | 2 | 0.60002 | 0.20725 | 0.6688 | 0.0070 | -0.00401 | [-0.00514, -0.00294] | [-0.00581, -0.00245] | -0.00174 | [-0.00222, -0.00129] | candidate better |

## 4. Sub-periods

| period | rows | LL M0 | LL M0R | LL M4A+raw | Δ M4A+raw | 95% CI | verdict |
|---|---|---|---|---|---|---|---|
| 2010-2014 | 14,292 | 0.58042 | 0.58069 | 0.57314 | -0.00728 | [-0.00981, -0.00491] | candidate better |
| 2015-2019 | 13,961 | 0.60487 | 0.60412 | 0.59995 | -0.00493 | [-0.00730, -0.00283] | candidate better |
| 2020-2023 | 9,855 | 0.62191 | 0.62088 | 0.61776 | -0.00414 | [-0.00648, -0.00199] | candidate better |
| 2024-2026 | 7,083 | 0.62511 | 0.62397 | 0.62075 | -0.00436 | [-0.00721, -0.00185] | candidate better |

## 5. Dependence sensitivity

| comparison (new vs reference) | rows | blocks | ΔLL | 95% CI | ΔBrier | 95% CI | P*(ΔLL<0) | verdict |
|---|---|---|---|---|---|---|---|---|
| M4A+raw vs M0 [week] | 45191 | 645 | -0.00541 | [-0.00662, -0.00423] | -0.00231 | [-0.00284, -0.00180] | 1.000 | candidate better |
| M4A+raw vs M0 [tournament] | 45191 | 2251 | -0.00541 | [-0.00649, -0.00438] | -0.00231 | [-0.00277, -0.00187] | 1.000 | candidate better |
| M4A+raw vs M0 [season] | 45191 | 17 | -0.00541 | [-0.00677, -0.00413] | -0.00231 | [-0.00289, -0.00176] | 1.000 | candidate better |
| M0R vs M0 [week] | 45191 | 645 | -0.00055 | [-0.00084, -0.00027] | -0.00017 | [-0.00027, -0.00008] | 1.000 | candidate better |
| M0R vs M0 [tournament] | 45191 | 2251 | -0.00055 | [-0.00082, -0.00028] | -0.00017 | [-0.00026, -0.00009] | 1.000 | candidate better |
| M0R vs M0 [season] | 45191 | 17 | -0.00055 | [-0.00097, -0.00016] | -0.00017 | [-0.00031, -0.00004] | 0.993 | candidate better |

## 6. Subgroups

| group | level | n | LL M0 | LL M0R | LL M4A+raw | Δ M4A+raw | 95% CI | Δ M0R | 95% CI |
|---|---|---|---|---|---|---|---|---|---|
| surface | (missing) | 50 | 0.66945 | 0.67010 | 0.66462 | -0.00483 | [-0.06237, +0.01983] | +0.00065 | [+0.00028, +0.00081] |
| surface | Carpet | 64 | 0.64496 | 0.64606 | 0.57451 | -0.07045 | [-0.13784, -0.01065] | +0.00110 | [-0.00206, +0.00430] |
| surface | Clay | 14152 | 0.61216 | 0.61156 | 0.61014 | -0.00202 | [-0.00423, +0.00019] | -0.00059 | [-0.00113, -0.00005] |
| surface | Grass | 4644 | 0.59897 | 0.59961 | 0.59399 | -0.00498 | [-0.00739, -0.00283] | +0.00064 | [-0.00014, +0.00139] |
| surface | Hard | 26281 | 0.60032 | 0.59957 | 0.59316 | -0.00716 | [-0.00862, -0.00567] | -0.00075 | [-0.00113, -0.00037] |
| event_type | Grand Slam | 7889 | 0.52620 | 0.52841 | 0.52788 | +0.00168 | [-0.00013, +0.00345] | +0.00221 | [+0.00168, +0.00277] |
| event_type | other tour-level | 37302 | 0.62049 | 0.61935 | 0.61357 | -0.00691 | [-0.00835, -0.00554] | -0.00114 | [-0.00143, -0.00083] |
| tournament_level | A | 23882 | 0.62559 | 0.62446 | 0.62062 | -0.00497 | [-0.00616, -0.00387] | -0.00113 | [-0.00147, -0.00080] |
| tournament_level | D | 3933 | 0.60999 | 0.61169 | 0.58962 | -0.02037 | [-0.02871, -0.01092] | +0.00170 | [+0.00105, +0.00233] |
| tournament_level | F | 312 | 0.60026 | 0.59940 | 0.60105 | +0.00080 | [-0.00856, +0.00922] | -0.00085 | [-0.00501, +0.00286] |
| tournament_level | G | 7889 | 0.52620 | 0.52841 | 0.52788 | +0.00168 | [-0.00013, +0.00345] | +0.00221 | [+0.00168, +0.00277] |
| tournament_level | M | 9112 | 0.61302 | 0.61060 | 0.60651 | -0.00651 | [-0.00806, -0.00496] | -0.00243 | [-0.00310, -0.00180] |
| tournament_level | O | 63 | 0.52051 | 0.52378 | 0.52294 | +0.00242 | [+0.00242, +0.00242] | +0.00327 | [+0.00327, +0.00327] |
| sample_size | 0: no prior points (<1) | 2522 | 0.60684 | 0.60952 | 0.54630 | -0.06055 | [-0.07271, -0.04818] | +0.00268 | [+0.00211, +0.00328] |
| sample_size | 1: 1-999 | 5866 | 0.59989 | 0.59948 | 0.60326 | +0.00336 | [-0.00054, +0.00761] | -0.00041 | [-0.00117, +0.00034] |
| sample_size | 2: 1000-4999 | 11816 | 0.62223 | 0.62067 | 0.61886 | -0.00337 | [-0.00469, -0.00205] | -0.00157 | [-0.00208, -0.00103] |
| sample_size | 3: >= 5000 | 24987 | 0.59610 | 0.59567 | 0.59323 | -0.00287 | [-0.00368, -0.00210] | -0.00043 | [-0.00080, -0.00008] |
| history | sparse (<1000 prior service points, either player) | 8388 | 0.60198 | 0.60250 | 0.58613 | -0.01585 | [-0.02105, -0.01053] | +0.00052 | [-0.00006, +0.00108] |
| history | substantial (>=1000 for both) | 36803 | 0.60449 | 0.60370 | 0.60146 | -0.00303 | [-0.00375, -0.00234] | -0.00080 | [-0.00110, -0.00047] |
| elo_confidence_pct | 50-60 | 14960 | 0.68815 | 0.68809 | 0.68189 | -0.00626 | [-0.00885, -0.00376] | -0.00007 | [-0.00019, +0.00005] |
| elo_confidence_pct | 60-70 | 12321 | 0.65277 | 0.65255 | 0.64650 | -0.00628 | [-0.00816, -0.00441] | -0.00022 | [-0.00055, +0.00010] |
| elo_confidence_pct | 70-80 | 9355 | 0.58479 | 0.58383 | 0.57983 | -0.00496 | [-0.00670, -0.00318] | -0.00096 | [-0.00165, -0.00027] |
| elo_confidence_pct | 80-90 | 6106 | 0.46348 | 0.46199 | 0.45981 | -0.00367 | [-0.00584, -0.00152] | -0.00149 | [-0.00265, -0.00038] |
| elo_confidence_pct | 90-100 | 2449 | 0.26876 | 0.26750 | 0.26683 | -0.00193 | [-0.00543, +0.00133] | -0.00126 | [-0.00306, +0.00043] |
| year | 2010 | 2908 | 0.58860 | 0.58866 | 0.57961 | -0.00899 | [-0.01487, -0.00414] | +0.00006 | [-0.00106, +0.00111] |
| year | 2011 | 2974 | 0.57763 | 0.57833 | 0.57372 | -0.00390 | [-0.00899, +0.00054] | +0.00070 | [-0.00048, +0.00179] |
| year | 2012 | 2898 | 0.56858 | 0.56953 | 0.56401 | -0.00456 | [-0.00924, -0.00069] | +0.00095 | [-0.00001, +0.00184] |
| year | 2013 | 2818 | 0.58316 | 0.58282 | 0.57213 | -0.01103 | [-0.01712, -0.00535] | -0.00034 | [-0.00151, +0.00082] |
| year | 2014 | 2694 | 0.58455 | 0.58446 | 0.57637 | -0.00818 | [-0.01423, -0.00294] | -0.00009 | [-0.00113, +0.00088] |
| year | 2015 | 2833 | 0.57591 | 0.57641 | 0.56759 | -0.00832 | [-0.01542, -0.00264] | +0.00051 | [-0.00039, +0.00130] |
| year | 2016 | 2906 | 0.58499 | 0.58482 | 0.58215 | -0.00283 | [-0.00644, +0.00063] | -0.00017 | [-0.00122, +0.00084] |
| year | 2017 | 2786 | 0.60964 | 0.60877 | 0.60322 | -0.00642 | [-0.01336, -0.00080] | -0.00086 | [-0.00192, +0.00014] |
| year | 2018 | 2804 | 0.63030 | 0.62841 | 0.62624 | -0.00406 | [-0.00795, -0.00028] | -0.00189 | [-0.00290, -0.00086] |
| year | 2019 | 2632 | 0.62587 | 0.62443 | 0.62294 | -0.00293 | [-0.00601, -0.00012] | -0.00144 | [-0.00274, -0.00025] |
| year | 2020 | 1427 | 0.61978 | 0.61855 | 0.61673 | -0.00304 | [-0.00875, +0.00245] | -0.00122 | [-0.00349, +0.00068] |
| year | 2021 | 2646 | 0.62614 | 0.62500 | 0.62297 | -0.00317 | [-0.00792, +0.00116] | -0.00114 | [-0.00212, -0.00024] |
| year | 2022 | 2919 | 0.61156 | 0.61129 | 0.60757 | -0.00399 | [-0.00835, +0.00031] | -0.00027 | [-0.00152, +0.00087] |
| year | 2023 | 2863 | 0.62961 | 0.62799 | 0.62386 | -0.00574 | [-0.00992, -0.00170] | -0.00161 | [-0.00287, -0.00039] |
| year | 2024 | 2955 | 0.62316 | 0.62209 | 0.62089 | -0.00227 | [-0.00594, +0.00093] | -0.00106 | [-0.00216, -0.00007] |
| year | 2025 | 2728 | 0.62867 | 0.62727 | 0.62466 | -0.00402 | [-0.00769, -0.00030] | -0.00140 | [-0.00286, -0.00012] |
| year | 2026 | 1400 | 0.62229 | 0.62149 | 0.61285 | -0.00943 | [-0.01849, -0.00104] | -0.00080 | [-0.00286, +0.00113] |

## 7. Walk-forward training cutoffs

| prediction year | training cutoff | last training release batch | training rows | predicted rows |
|---|---|---|---|---|
| 2010 | 2010-01-02 | 2009-11-28 | 63,836 | 2,908 |
| 2011 | 2011-01-01 | 2010-11-27 | 66,744 | 2,974 |
| 2012 | 2012-01-07 | 2011-12-31 | 69,718 | 2,898 |
| 2013 | 2013-01-05 | 2012-12-29 | 72,616 | 2,818 |
| 2014 | 2014-01-04 | 2013-12-28 | 75,434 | 2,694 |
| 2015 | 2015-01-03 | 2014-11-15 | 78,128 | 2,833 |
| 2016 | 2016-01-02 | 2015-11-21 | 80,961 | 2,906 |
| 2017 | 2017-01-07 | 2016-12-31 | 83,867 | 2,786 |
| 2018 | 2018-01-06 | 2017-12-30 | 86,653 | 2,804 |
| 2019 | 2019-01-05 | 2018-12-29 | 89,457 | 2,632 |
| 2020 | 2020-01-04 | 2019-11-23 | 92,089 | 1,427 |
| 2021 | 2021-01-02 | 2020-11-14 | 93,516 | 2,646 |
| 2022 | 2022-01-01 | 2021-12-04 | 96,162 | 2,919 |
| 2023 | 2023-01-07 | 2022-12-31 | 99,081 | 2,863 |
| 2024 | 2024-01-06 | 2023-12-30 | 101,944 | 2,955 |
| 2025 | 2025-01-04 | 2024-12-28 | 104,899 | 2,728 |
| 2026 | 2026-01-03 | 2025-12-13 | 107,627 | 1,400 |

## 8. Coefficients of M4A by year (original units)

| year | training rows | elo_logit | spw_shrunk_diff | rpw_shrunk_diff | spw_decay_diff | rpw_decay_diff | decay_missing_diff |
|---|---|---|---|---|---|---|---|
| 2010 | 63,836 | 0.7267 | 0.5637 | -1.0148 | 4.4442 | 4.6489 | -0.8774 |
| 2011 | 66,744 | 0.7302 | 0.7058 | -1.3553 | 4.2902 | 4.8613 | -0.8879 |
| 2012 | 69,718 | 0.7387 | 0.5536 | -1.5380 | 4.3208 | 4.8995 | -0.8825 |
| 2013 | 72,616 | 0.7443 | 0.4741 | -1.4511 | 4.4065 | 4.8201 | -0.8816 |
| 2014 | 75,434 | 0.7418 | 0.4406 | -1.4090 | 4.4912 | 4.8468 | -0.8986 |
| 2015 | 78,128 | 0.7441 | 0.3431 | -1.4547 | 4.5390 | 4.8504 | -0.9043 |
| 2016 | 80,961 | 0.7484 | 0.1937 | -1.4268 | 4.6671 | 4.7553 | -0.9157 |
| 2017 | 83,867 | 0.7524 | 0.1696 | -1.2934 | 4.6120 | 4.5748 | -0.9145 |
| 2018 | 86,653 | 0.7527 | 0.0407 | -1.1229 | 4.6593 | 4.3658 | -0.9200 |
| 2019 | 89,457 | 0.7514 | -0.0329 | -1.0493 | 4.6259 | 4.1736 | -0.9196 |
| 2020 | 92,089 | 0.7502 | 0.0300 | -0.9968 | 4.5072 | 4.1095 | -0.9236 |
| 2021 | 93,516 | 0.7508 | 0.1904 | -1.0157 | 4.3180 | 4.0852 | -0.9259 |
| 2022 | 96,162 | 0.7503 | 0.0388 | -1.1331 | 4.3964 | 4.1659 | -0.9122 |
| 2023 | 99,081 | 0.7512 | 0.0439 | -0.9850 | 4.3875 | 4.0331 | -0.9118 |
| 2024 | 101,944 | 0.7511 | -0.2032 | -1.0060 | 4.5076 | 3.9567 | -0.9079 |
| 2025 | 104,899 | 0.7529 | -0.1772 | -0.9277 | 4.4306 | 3.7766 | -0.9050 |
| 2026 | 107,627 | 0.7515 | -0.1513 | -0.9121 | 4.3560 | 3.8072 | -0.9002 |

## 9. Prospective evaluation protocol (matches after 2026-10-25)

The only genuinely untouched evaluation is prospective. The protocol is fixed now:

1. **Model.** `M4A+raw` exactly as frozen (hash `bc4e5c870c6e21b9`): same inputs, C, calibration method, epsilon and walk-forward rule. Upstream Elo and feature parameters stay as merged.
2. **Production fit.** `data/modeling/models/production_model.json`: trained on every label released by the end of the data (training cutoff batch 2026-06-06, 109,027 rows). Valid from prediction batch 2026-06-06.
3. **Population.** Every Phase 2 target match (same pool, statuses and eligibility rules) whose prediction batch is after 2026-10-25, i.e. from batch 2026-10-31 on.
4. **Timing.** Predictions for a batch are produced from Elo and feature state frozen at batch start and stored (with model id, version, training cutoff, configuration hash) *before* any result of that batch is ingested. Refits follow the same annual rule (first batch of each calendar year), using only labels released before it. Labels released between the end of the current data and 2026-10-31 may be ingested through the normal Phase 1→3A pipeline before the first prospective batch; they are training data, not evaluation data.
5. **Primary endpoint.** ΔLL = LL(model) − LL(M0) on identical matches, weekly-block bootstrap, 95% interval, reported with N and the number of blocks, whatever the sign. Secondary: Brier, ECE, surfaces, Grand Slams vs other events.
6. **Analysis plan.** First analysis after at least 26 weekly batches, final after 52; no interim decisions about the model. Any change to the model starts a new, separately registered prospective period.
