# Data license

The tennis data used by this project, and everything derived from it, is **not** covered by the MIT license in
`LICENSE`. It is licensed under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
License (CC BY-NC-SA 4.0)**: https://creativecommons.org/licenses/by-nc-sa/4.0/

## Source and attribution

- ATP match data compiled by **Jeff Sackmann** (https://github.com/JeffSackmann/tennis_atp), retrieved from the
  archive mirror [Aneeshers/tennis-sackmann-archive](https://github.com/Aneeshers/tennis-sackmann-archive) at the
  pinned commit `83733587353df8a41f2fd4f516147d5aa83f5a8d`, under CC BY-NC-SA 4.0.
- Attribution for derived material: "Derived from Jeff Sackmann's ATP data (tennis_atp), CC BY-NC-SA 4.0.
  Changes: validation, Elo ratings, serve/return features and model predictions computed by this project."

## What is covered

| material | where | status |
|---|---|---|
| Raw source files and the full validated tables | `data/` (git-ignored, built locally) | not redistributed by this repository |
| Report tables and statistics derived from the data | `reports/` | CC BY-NC-SA 4.0 |
| Public dashboard dataset (historical model predictions, point-in-time player indicators, player names) | `dashboard/data/` | CC BY-NC-SA 4.0; see its data card `dashboard/data/README.md` |

Conditions: credit Jeff Sackmann and link the original repository; **non-commercial use only**; distribute
adaptations under the same license; indicate changes. No additional restrictions may be applied.

## Not included

Owner-supplied or otherwise licensed sources (for example TennisMyLife exports used in Phase 5A), records derived
from them, identity approvals, local state snapshots and backups are **not** part of this repository and are not
licensed by it. Public reports about them contain aggregate counts only.

Nothing in this repository is betting advice or fit for decisions with financial consequences.
