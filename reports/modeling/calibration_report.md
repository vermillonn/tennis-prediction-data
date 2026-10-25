# Phase 3B calibration report

Derived from Jeff Sackmann's ATP data (CC BY-NC-SA 4.0); aggregate statistics only, no row-level data.

## 1. Methods and data separation

Both calibrators are symmetric, g(1 − p) = 1 − g(p), so they cannot break player-order symmetry:

- `temperature`: σ(b · logit p), Platt scaling without an intercept (an intercept would favour whichever player is labelled A).
- `isotonic`: pool-adjacent-violators on the symmetrised sample {(p, y)} ∪ {(1 − p, 1 − y)} with linear interpolation; requires ≥ 5000 rows.

| item | rows used |
|---|---|
| base model for year y | trained on labels released before the first batch of y |
| calibrator for year y | the same model's out-of-fold predictions for years y−3…y−1 whose label was released before the first batch of y |
| evaluated | year y only; never rows used to fit the calibrator or the model |

The release filter matters for multi-week events that straddle New Year: their results are predicted in year y−1 but known only in y.

## 2. Development years

| model | log loss | Brier | ECE |
|---|---|---|---|
| M0 | 0.60142 | 0.20775 | 0.0066 |
| M0R | 0.60187 | 0.20791 | 0.0116 |
| M0R+temperature | 0.60136 | 0.20774 | 0.0072 |
| M0R+isotonic | 0.60312 | 0.20797 | 0.0050 |
| M4A | 0.59512 | 0.20486 | 0.0116 |
| M4A+temperature | 0.59466 | 0.20472 | 0.0058 |
| M4A+isotonic | 0.59770 | 0.20519 | 0.0074 |

Selection: raw 0.595120, temperature 0.594663, isotonic 0.597700 → **raw** (simplest within 0.0005).

## 3. Exploratory years (frozen method)

| model | log loss | Brier | ECE |
|---|---|---|---|
| M0 | 0.60403 | 0.20899 | 0.0130 |
| M0R | 0.60347 | 0.20882 | 0.0052 |
| M0R+temperature | 0.60330 | 0.20872 | 0.0075 |
| M0R+isotonic | 0.60554 | 0.20913 | 0.0094 |
| M4A | 0.59861 | 0.20668 | 0.0064 |
| M4A+temperature | 0.59840 | 0.20659 | 0.0070 |
| M4A+isotonic | 0.60061 | 0.20697 | 0.0066 |

## 4. Temperature slope fitted for each development year

| year | phase | calibration rows | slope b |
|---|---|---|---|
| 2002 | development | 9835 | 0.9867 |
| 2003 | development | 9686 | 1.0127 |
| 2004 | development | 9316 | 1.0353 |
| 2005 | development | 9280 | 1.0268 |
| 2006 | development | 9440 | 1.0833 |
| 2007 | development | 9598 | 1.0832 |
| 2008 | development | 9487 | 1.1144 |
| 2009 | development | 9131 | 1.0840 |

A slope below 1 means the raw model was over-confident on the preceding years.

## 5. Reliability tables (10 equal-width bins of P(A wins))

### Development

| model | bin | n | mean predicted | observed | obs − pred |
|---|---|---|---|---|---|
| M0 | 0.0–0.1 | 446 | 0.0692 | 0.0695 | +0.0003 |
| M0 | 0.1–0.2 | 1,638 | 0.1554 | 0.1630 | +0.0076 |
| M0 | 0.2–0.3 | 2,565 | 0.2540 | 0.2456 | -0.0084 |
| M0 | 0.3–0.4 | 3,399 | 0.3510 | 0.3483 | -0.0027 |
| M0 | 0.4–0.5 | 4,404 | 0.4519 | 0.4584 | +0.0066 |
| M0 | 0.5–0.6 | 4,166 | 0.5478 | 0.5516 | +0.0038 |
| M0 | 0.6–0.7 | 3,294 | 0.6477 | 0.6518 | +0.0041 |
| M0 | 0.7–0.8 | 2,671 | 0.7484 | 0.7432 | -0.0053 |
| M0 | 0.8–0.9 | 1,591 | 0.8426 | 0.8165 | -0.0261 |
| M0 | 0.9–1.0 | 448 | 0.9309 | 0.9442 | +0.0133 |
| M0R | 0.0–0.1 | 284 | 0.0716 | 0.0599 | -0.0117 |
| M0R | 0.1–0.2 | 1,360 | 0.1579 | 0.1419 | -0.0160 |
| M0R | 0.2–0.3 | 2,426 | 0.2540 | 0.2181 | -0.0360 |
| M0R | 0.3–0.4 | 3,607 | 0.3513 | 0.3435 | -0.0078 |
| M0R | 0.4–0.5 | 4,775 | 0.4524 | 0.4509 | -0.0015 |
| M0R | 0.5–0.6 | 4,556 | 0.5477 | 0.5536 | +0.0058 |
| M0R | 0.6–0.7 | 3,415 | 0.6474 | 0.6647 | +0.0173 |
| M0R | 0.7–0.8 | 2,650 | 0.7470 | 0.7577 | +0.0108 |
| M0R | 0.8–0.9 | 1,275 | 0.8424 | 0.8557 | +0.0133 |
| M0R | 0.9–1.0 | 274 | 0.9305 | 0.9526 | +0.0220 |
| M4A | 0.0–0.1 | 297 | 0.0721 | 0.0673 | -0.0048 |
| M4A | 0.1–0.2 | 1,498 | 0.1569 | 0.1449 | -0.0120 |
| M4A | 0.2–0.3 | 2,711 | 0.2541 | 0.2202 | -0.0339 |
| M4A | 0.3–0.4 | 3,649 | 0.3506 | 0.3412 | -0.0094 |
| M4A | 0.4–0.5 | 4,230 | 0.4518 | 0.4518 | -0.0000 |
| M4A | 0.5–0.6 | 4,166 | 0.5471 | 0.5578 | +0.0107 |
| M4A | 0.6–0.7 | 3,513 | 0.6492 | 0.6587 | +0.0095 |
| M4A | 0.7–0.8 | 2,816 | 0.7462 | 0.7592 | +0.0131 |
| M4A | 0.8–0.9 | 1,454 | 0.8421 | 0.8521 | +0.0100 |
| M4A | 0.9–1.0 | 288 | 0.9289 | 0.9653 | +0.0363 |
| M4A+temperature | 0.0–0.1 | 414 | 0.0705 | 0.0652 | -0.0053 |
| M4A+temperature | 0.1–0.2 | 1,618 | 0.1553 | 0.1533 | -0.0020 |
| M4A+temperature | 0.2–0.3 | 2,791 | 0.2535 | 0.2293 | -0.0242 |
| M4A+temperature | 0.3–0.4 | 3,492 | 0.3506 | 0.3519 | +0.0014 |
| M4A+temperature | 0.4–0.5 | 4,070 | 0.4515 | 0.4536 | +0.0021 |
| M4A+temperature | 0.5–0.6 | 3,995 | 0.5471 | 0.5554 | +0.0084 |
| M4A+temperature | 0.6–0.7 | 3,396 | 0.6492 | 0.6505 | +0.0013 |
| M4A+temperature | 0.7–0.8 | 2,841 | 0.7468 | 0.7504 | +0.0036 |
| M4A+temperature | 0.8–0.9 | 1,600 | 0.8428 | 0.8444 | +0.0016 |
| M4A+temperature | 0.9–1.0 | 405 | 0.9299 | 0.9432 | +0.0133 |

### Exploratory

| model | bin | n | mean predicted | observed | obs − pred |
|---|---|---|---|---|---|
| M0 | 0.0–0.1 | 1,230 | 0.0636 | 0.0748 | +0.0112 |
| M0 | 0.1–0.2 | 3,031 | 0.1550 | 0.1765 | +0.0216 |
| M0 | 0.2–0.3 | 4,688 | 0.2530 | 0.2617 | +0.0087 |
| M0 | 0.3–0.4 | 6,109 | 0.3518 | 0.3567 | +0.0049 |
| M0 | 0.4–0.5 | 7,458 | 0.4514 | 0.4619 | +0.0105 |
| M0 | 0.5–0.6 | 7,502 | 0.5478 | 0.5413 | -0.0065 |
| M0 | 0.6–0.7 | 6,212 | 0.6483 | 0.6333 | -0.0150 |
| M0 | 0.7–0.8 | 4,667 | 0.7469 | 0.7178 | -0.0291 |
| M0 | 0.8–0.9 | 3,075 | 0.8453 | 0.8247 | -0.0206 |
| M0 | 0.9–1.0 | 1,219 | 0.9362 | 0.9196 | -0.0166 |
| M0R | 0.0–0.1 | 966 | 0.0649 | 0.0642 | -0.0007 |
| M0R | 0.1–0.2 | 2,754 | 0.1558 | 0.1572 | +0.0014 |
| M0R | 0.2–0.3 | 4,593 | 0.2537 | 0.2510 | -0.0026 |
| M0R | 0.3–0.4 | 6,296 | 0.3521 | 0.3513 | -0.0007 |
| M0R | 0.4–0.5 | 7,907 | 0.4516 | 0.4576 | +0.0060 |
| M0R | 0.5–0.6 | 7,932 | 0.5476 | 0.5435 | -0.0041 |
| M0R | 0.6–0.7 | 6,433 | 0.6478 | 0.6411 | -0.0067 |
| M0R | 0.7–0.8 | 4,589 | 0.7468 | 0.7335 | -0.0133 |
| M0R | 0.8–0.9 | 2,751 | 0.8446 | 0.8375 | -0.0071 |
| M0R | 0.9–1.0 | 970 | 0.9343 | 0.9247 | -0.0096 |
| M4A | 0.0–0.1 | 938 | 0.0673 | 0.0586 | -0.0087 |
| M4A | 0.1–0.2 | 2,969 | 0.1565 | 0.1533 | -0.0032 |
| M4A | 0.2–0.3 | 5,048 | 0.2528 | 0.2528 | -0.0000 |
| M4A | 0.3–0.4 | 6,317 | 0.3512 | 0.3562 | +0.0050 |
| M4A | 0.4–0.5 | 7,260 | 0.4512 | 0.4583 | +0.0071 |
| M4A | 0.5–0.6 | 7,296 | 0.5481 | 0.5404 | -0.0077 |
| M4A | 0.6–0.7 | 6,381 | 0.6487 | 0.6361 | -0.0126 |
| M4A | 0.7–0.8 | 5,080 | 0.7465 | 0.7386 | -0.0080 |
| M4A | 0.8–0.9 | 2,966 | 0.8431 | 0.8409 | -0.0022 |
| M4A | 0.9–1.0 | 936 | 0.9317 | 0.9284 | -0.0033 |
| M4A+temperature | 0.0–0.1 | 1,043 | 0.0646 | 0.0681 | +0.0035 |
| M4A+temperature | 0.1–0.2 | 3,001 | 0.1559 | 0.1586 | +0.0027 |
| M4A+temperature | 0.2–0.3 | 4,928 | 0.2528 | 0.2508 | -0.0020 |
| M4A+temperature | 0.3–0.4 | 6,260 | 0.3510 | 0.3553 | +0.0042 |
| M4A+temperature | 0.4–0.5 | 7,300 | 0.4514 | 0.4597 | +0.0083 |
| M4A+temperature | 0.5–0.6 | 7,367 | 0.5482 | 0.5431 | -0.0051 |
| M4A+temperature | 0.6–0.7 | 6,351 | 0.6489 | 0.6336 | -0.0153 |
| M4A+temperature | 0.7–0.8 | 4,910 | 0.7468 | 0.7377 | -0.0091 |
| M4A+temperature | 0.8–0.9 | 2,986 | 0.8442 | 0.8372 | -0.0070 |
| M4A+temperature | 0.9–1.0 | 1,045 | 0.9344 | 0.9282 | -0.0062 |

ECE = Σ_bins n_bin |observed − mean predicted| / n. Because player A is chosen by a hash of the match id, the table is roughly symmetric around 0.5.
