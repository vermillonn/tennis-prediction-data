# Portfolio summary

Ready-to-use descriptions of this project, with the evidence behind each claim. All figures come from the committed
reports. See [research_summary.md](research_summary.md) for context.

## Short description (about 60 words)

A reproducible machine-learning study of ATP tennis match prediction. A frozen six-input logistic regression adds
point-in-time serve and return statistics to a tuned surface Elo rating. Across 45,191 historical matches it lowers
log loss from 0.6040 to 0.5986 under walk-forward evaluation. The project includes leakage controls, bootstrap
uncertainty, 532 automated tests and an interactive Streamlit research dashboard.

## Detailed description (about 180 words)

This project asks whether serving and returning statistics improve tennis match-winner prediction beyond a
well-tuned surface Elo rating. It is built as a reproducible research system:

- **Data.** About 200,000 ATP matches from a pinned, checksummed public archive are validated record by record.
  Nothing is silently dropped.
- **Chronology.** Because the archive has no match dates, the project defines a weekly release chronology under
  which every rating and feature uses only results already known.
- **Model and evaluation.**
  - Six candidate models (among 14 registered experiments) were compared under a selection rule fixed in advance.
  - The chosen model (M4A) was frozen by hash before later seasons were scored.
  - It was evaluated with annual walk-forward refits and paired block-bootstrap intervals.
- **Results.** M4A improves log loss by 0.0054 (95% CI −0.0066 to −0.0042) over 2010–2026, and by 0.0063 on the
  2002–2009 selection period.
  - Much of the gain comes from players with little history.
  - Grand Slams show no improvement.
  - Both periods were used during development, so the results are reported as retrospective evidence, not an
    independent test.
- **Infrastructure.** Built and tested, but deliberately disabled until live data are available:
  - a hash-chained prediction ledger;
  - a pre-specified prospective evaluation protocol;
  - an audit framework.
- **Presentation.** A five-section Streamlit dashboard presents the results from a public, licence-compliant dataset.

## CV bullet points

The project was developed with AI-assisted coding (Claude Code), under the owner's direction, review and approval at
each phase. Phrase the bullets so you can defend each claim in an interview. The wording below describes what was
designed, specified and validated.

- **Analytics and modelling.** Led a reproducible machine-learning study of about 200,000 ATP matches.
  - Defined the research question, the point-in-time evaluation protocol and the model-selection rule.
  - The frozen logistic model reduced log loss against a tuned surface-Elo baseline by 0.0054 (95% CI −0.0066 to
    −0.0042) across 45,191 walk-forward historical predictions.
  - Documented why this is not an independent test.
- **Data engineering and quality.** Built a validated, checksummed data pipeline in Python (Polars, DuckDB,
  Parquet).
  - It accounts for every source row: 199,280 validated, 109 quarantined with reasons.
  - It enforces leakage-safe weekly chronology.
  - It is covered by 532 automated tests in GitHub Actions continuous integration.
- **Communication and governance.** Delivered a five-section Streamlit and Plotly research dashboard that
  reproduces the published metrics exactly. Also implemented:
  - provenance hashes;
  - separate software (MIT) and data (CC BY-NC-SA 4.0) licensing;
  - operational gates that keep unverified live predictions disabled.

## Interview talking points

**Why Elo as the baseline?**
- Elo is the standard strong baseline for head-to-head sports. It is transparent and hard to beat.
- A surface-specific version improves on global Elo by 0.0033 log loss over 2010–2026.
- Any new feature should be judged by what it adds to Elo, not to a weak baseline. The model therefore uses the Elo
  log-odds as an input, and compares every result with Elo on identical matches.

**Why should serve/return statistics add information?**
- Elo sees only wins and losses. Points won on serve and on return measure *how* matches were won, and respond
  faster to changes in form.
- The evidence says recency matters most:
  - 52-week decay-weighted rates help;
  - career averages do not;
  - a large share of the gain comes from players Elo knows little about (a cold-start correction).

**How was chronological leakage handled, and what remains?**
- Matches are grouped into weekly batches, and each prediction uses state frozen at the batch start.
- Results become visible only after their event's release batch closes.
- A model-input allowlist blocks outcome columns, and an audit recomputed features from raw data with zero
  differences.
- **What remains:** the source has no match dates, so ordering within an event is approximate, and some late-round
  results can enter the state slightly early. These issues are documented, not hidden.

**Why logistic regression?**
- The question is whether a few interpretable inputs add information. A regularised logistic regression answers
  that transparently.
- With no intercept and antisymmetric inputs, it is exactly symmetric in player order, and every coefficient can be
  inspected.
- A more flexible model would make it harder to tell real signal from overfitting on about 25,000 development
  matches.

**What do log loss and calibration mean?**
- **Log loss** is the average of −log(probability given to the actual winner). It rewards being confident and right,
  and punishes being confident and wrong.
- **Calibration** asks whether events predicted at 70% happen about 70% of the time. Expected calibration error
  summarises the gap: 0.0064 for M4A against 0.0130 for Elo in the exploratory period.
- **Accuracy** ignores confidence, so it is a weaker measure for probabilities.

**Why is the model not claimed to be prospectively validated?**
- Both historical periods influenced development:
  - the first selected the model;
  - the second was examined while the Elo baseline was built.
- A genuine test needs forecasts recorded *before* matches, with an independent timestamp, on data that did not
  exist during development.
- The protocol for that is written down, and the ledger exists. But live data access and a complete backfill are
  missing, so the gate stays closed and no prospective result exists.

**What did data access and licensing teach?**
- The best public tennis data is non-commercial (CC BY-NC-SA 4.0). Every derived dataset inherits that licence, so
  code (MIT) and data must be licensed separately.
- Current data from other sources failed a statistical compatibility check against the archive, and could not be
  redistributed.
- Data availability, not modelling, is the binding constraint on going live.

**What you need to understand to defend the methodology**
- The weekly release chronology, and why tournament start dates are not evidence that a result was known.
- How walk-forward refits differ from the single frozen production fit.
- The selection rule: why M4A was chosen over the slightly better M4B (simplest model within 0.0005).
- How to read a paired block-bootstrap interval, and why it may be too narrow (dependence through shared players).
- The cold-start share of the gain, and the Grand Slam weakness.
- The meaning of the gates (DISABLED / INCOMPLETE), and why historical state version 0 stays active.

## GitHub presentation

**Repository description** (at most 350 characters):

> Reproducible ML research on ATP match prediction: point-in-time serve/return features on top of surface Elo, a
> frozen logistic model with walk-forward evaluation and bootstrap uncertainty, a hash-chained prediction ledger,
> and a Streamlit research dashboard. Retrospective results only; data CC BY-NC-SA 4.0.

**Suggested topics:**

- `machine-learning`, `predictive-modeling`, `data-science`, `statistics`, `reproducible-research`
- `sports-analytics`, `tennis`, `elo-rating`, `logistic-regression`, `calibration`
- `python`, `polars`, `duckdb`, `streamlit`

These are recommendations only. The repository settings are left to the owner.
