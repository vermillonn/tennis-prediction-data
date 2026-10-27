"""The five dashboard sections. Data logic lives in ``tennis_data.dashboard.data`` (tested without Streamlit)."""

from __future__ import annotations

from datetime import date, timedelta

import charts
import polars as pl
import streamlit as st

from tennis_data.dashboard import data as D

ARCHIVE_END = date(2026, 5, 25)
RETRO = ("**Retrospective research results.** Every number here comes from historical, point-in-time "
         "reconstructions on the ATP archive. They are not independently validated prospective performance: the "
         "development period was used to select the model and the later period was seen while building the Elo "
         "baseline. No prediction on this site is a live forecast.")  # fmt: skip
WALK_FORWARD = ("**Historical forecasts vs the production model.** These predictions come from annual walk-forward "
                "refits of the frozen M4A specification: each season is predicted by a fit trained only on earlier "
                "results (its training cutoff is shown per match). The single frozen production model "
                "(`bc4e5c870c6e21b9`) was fit on all results before 2026-06-06 and has no out-of-sample history; "
                "it is not what produced these numbers.")  # fmt: skip


# --------------------------------------------------------------------------------------------------- cached data
@st.cache_data(show_spinner=False)
def predictions() -> pl.DataFrame | None:
    return D.load_predictions()


@st.cache_data(show_spinner=False)
def snapshots() -> pl.DataFrame | None:
    return D.load_snapshots()


@st.cache_data(show_spinner=False)
def players() -> pl.DataFrame | None:
    return D.load_players()


@st.cache_data(show_spinner=False)
def published() -> dict:
    return D.published()


@st.cache_data(show_spinner=False)
def manifest() -> dict | None:
    return D.load_manifest()


@st.cache_data(show_spinner=False)
def status() -> dict | None:
    return D.load_status()


def plot(fig) -> None:
    """Interactive chart without Plotly's toolbar (it covers chart titles on touch screens)."""
    st.plotly_chart(fig, config={"displayModeBar": False, "displaylogo": False})


def missing_dataset(what: str) -> None:
    st.info(f"**{what} needs the public dashboard dataset** (`dashboard/data/`), which is not present in this "
            "checkout. The published aggregates on the Overview and Model evaluation pages still work. Maintainers "
            "can rebuild the dataset from the local research outputs with `python -m tennis_data.dashboard.build`; "
            "see `docs/dashboard.md`.", icon=":material/database_off:")  # fmt: skip


def kpi(col, label: str, value: str, help: str | None = None, delta: str | None = None) -> None:
    col.metric(label, value, delta=delta, delta_color="inverse" if delta else "normal", help=help)


def player_label(row: dict) -> str:
    ioc = f" ({row['ioc']})" if row.get("ioc") else ""
    return f"{row['name']}{ioc}"


def player_picker(label: str, options: pl.DataFrame, key: str, default_id: int | None = None,
                  allow_none: bool = False) -> int | None:  # fmt: skip
    rows = options.to_dicts()
    labels = [player_label(r) for r in rows]
    dup = {x for x in labels if labels.count(x) > 1}
    labels = [f"{lab} · #{r['player_id']}" if lab in dup else lab for lab, r in zip(labels, rows, strict=True)]
    ids = [r["player_id"] for r in rows]
    if allow_none:
        labels, ids = ["All players", *labels], [None, *ids]
    idx = ids.index(default_id) if default_id in ids else 0
    choice = st.selectbox(label, range(len(ids)), index=idx, format_func=lambda i: labels[i], key=key)
    return ids[choice]


# ------------------------------------------------------------------------------------------------------- overview
def overview() -> None:
    pub, man = published(), manifest()
    st.title("ATP match-winner prediction: a reproducible research study")
    st.markdown("Can a small, transparent model built on point-in-time serve and return statistics improve on a "
                "well-tuned surface Elo rating for predicting ATP tour-level matches? This dashboard presents the "
                "frozen model **M4A** and its historical evaluation against the Elo baseline.")  # fmt: skip
    st.warning(RETRO, icon=":material/history:")
    rows = D.headline(pub)
    if rows:
        total = sum(r["matches"] for r in rows)
        c = st.columns(4)
        kpi(c[0], "Evaluated matches", f"{total:,}", "Development 2002–2009 plus exploratory 2010–2026, all with a "
            "recorded pre-match M4A and surface-Elo prediction.")  # fmt: skip
        for col, r in zip(c[1:3], rows, strict=True):
            kpi(col, f"Log loss, {r['phase']}", f"{r['m4a']['log_loss']:.4f}",
                f"M4A on {r['matches']:,} matches; surface Elo {r['elo']['log_loss']:.4f}. Lower is better.",
                delta=f"{r['d_log_loss']:+.4f} vs Elo")  # fmt: skip
        kpi(c[3], "Accuracy, exploratory", f"{rows[1]['m4a']['accuracy']:.1%}",
            f"Share of matches the M4A favourite won; surface Elo {rows[1]['elo']['accuracy']:.1%}.")  # fmt: skip
    st.subheader("How the system was built")
    cols = st.columns(3)
    cols[0].markdown(
        "**1 · Data foundation**  \nJeff Sackmann's ATP archive (pinned commit), validated and "
        "quarantined record by record. Tour-level singles since 1968; match statistics since 1991."
    )
    cols[1].markdown("**2 · Ratings and features**  \nGlobal and surface Elo in weekly batches, and serve/return "
                     "point rates with shrinkage and 52-week decay, all computed only from results released before "
                     "each prediction batch.")  # fmt: skip
    cols[2].markdown(
        "**3 · Model and evaluation**  \nA six-input logistic regression (M4A) compared with surface "
        "Elo on identical matches, with weekly-block bootstrap intervals and annual walk-forward refits."
    )
    st.subheader("The M4A model")
    m = pub.get("modeling") or {}
    coef = (m.get("production") or {}).get("coef_original_units", {})
    left, right = st.columns([3, 2])
    with left:
        st.markdown("Logistic regression without intercept (L2, C = 10), raw calibration. All inputs are A − B "
                    "differences, so the model is exactly symmetric in player order.")  # fmt: skip
        if coef:
            desc = {"elo_logit": "Surface-Elo log-odds", "spw_shrunk_diff": "Serve points won, shrunk to the prior",
                    "rpw_shrunk_diff": "Return points won, shrunk to the prior",
                    "spw_decay_diff": "Serve points won, 52-week decay", "rpw_decay_diff": "Return points won, "
                    "52-week decay", "decay_missing_diff": "Missing recent statistics (indicator)"}  # fmt: skip
            st.dataframe(
                pl.DataFrame(
                    {
                        "input": list(coef),
                        "meaning": [desc.get(k, "") for k in coef],
                        "coefficient": [round(v, 4) for v in coef.values()],
                    }
                ),
                hide_index=True,
            )
    with right:
        st.markdown(f"""
| | |
|---|---|
| Frozen model hash | `{(m.get('frozen_hash') or '')[:16]}` |
| Configuration hash | `{m.get('configuration_hash', '–')}` |
| Model version | `{m.get('model_version', '–')}` |
| Production fit | labels before {(m.get('production') or {}).get('training_cutoff_batch', '–')} |
| Production rows | {(m.get('production') or {}).get('train_rows', 0):,} |
""")  # fmt: skip
        st.caption(
            "Historical predictions come from annual walk-forward refits of this specification. The "
            "production fit uses all labels up to its cutoff, so it has no out-of-sample history of its own."
        )
    st.subheader("Evaluation samples")
    if rows:
        st.dataframe(pl.DataFrame([{"period": D.PHASES[r["phase"]], "matches": r["matches"], "weekly batches":
                                    r["weeks"], "M4A log loss": round(r["m4a"]["log_loss"], 5), "Elo log loss":
                                    round(r["elo"]["log_loss"], 5), "Δ log loss": round(r["d_log_loss"], 5),
                                    "95% CI (block bootstrap)": D.fmt_ci(r["ll_ci95"], 5)} for r in rows]),
                     hide_index=True)  # fmt: skip
        st.caption(
            "Δ = M4A − surface Elo on identical matches; negative favours M4A. Intervals resample whole "
            "weekly batches and ignore dependence through shared players, so they may be too narrow."
        )
    st.subheader("Reproducibility")
    b = (pub.get("build") or {}).get("rows", {})
    c = st.columns(4)
    kpi(c[0], "Validated archive matches", f"{b.get('validated', 0):,}", "Tour-level singles that passed Phase 1 "
        f"validation; {b.get('quarantined', 0)} records quarantined.")  # fmt: skip
    kpi(
        c[1],
        "Automated tests",
        f"{(man or {}).get('tests_collected_at_build') or '–'}",
        "Tests collected when the dashboard dataset was built; CI runs the full suite on every change.",
    )
    kpi(
        c[2],
        "Leakage audit",
        "passed" if (m.get("leakage_audit") or {}).get("passed") else "–",
        "Point-in-time features recomputed independently for sampled matches; no current-match information.",
    )
    integrity = D.verify_manifest(man) if man else {}
    kpi(
        c[3],
        "Dataset integrity",
        "verified" if integrity and all(integrity.values()) else "unavailable",
        "SHA-256 of every dashboard data file matches its manifest.",
    )
    st.caption("Data: Jeff Sackmann's ATP archive, CC BY-NC-SA 4.0. Derived data in this dashboard is shared under "
               "the same licence for non-commercial research use. The software is MIT-licensed; that licence covers the code only, "
               "not the data. Not for betting or any decision with "
               "financial consequences.")  # fmt: skip


# ------------------------------------------------------------------------------------------------------ explorer
def explorer() -> None:
    st.title("Historical match explorer")
    st.caption("Every row is a prediction the frozen evaluation actually recorded before the match, using only "
               "results released before its weekly batch. Matches without eligible features have no prediction and "
               "do not appear.")  # fmt: skip
    st.caption(WALK_FORWARD)
    df, pl_ = predictions(), players()
    if df is None or pl_ is None:
        missing_dataset("The match explorer")
        return
    ids = set(df["player_a_id"].to_list()) | set(df["player_b_id"].to_list())
    opts = D.player_options(pl_.filter(pl.col("player_id").is_in(list(ids))))
    with st.container(border=True):
        c = st.columns([3, 3, 2])
        with c[0]:
            pid = player_picker("Player", opts, "ex_player", allow_none=True)
        with c[1]:
            opp = player_picker("Opponent", opts, "ex_opp", allow_none=True) if pid else None
            if not pid:
                st.selectbox("Opponent", ["Choose a player first"], disabled=True, key="ex_opp_disabled")
        with c[2]:
            phases = st.multiselect("Evaluation period", list(D.PHASES), default=list(D.PHASES),
                                    format_func=lambda p: p.capitalize(), key="ex_phase")  # fmt: skip
        c = st.columns([2, 3, 3])
        lo, hi = int(df["season"].min()), int(df["season"].max())
        with c[0]:
            surfaces = st.multiselect("Surface", sorted(df["surface"].drop_nulls().unique().to_list()), key="ex_surf")
        with c[1]:
            levels = st.multiselect("Tournament category", list(D.LEVELS), format_func=D.LEVELS.get, key="ex_lvl")
        with c[2]:
            years = st.slider("Seasons", lo, hi, (lo, hi), key="ex_years")
    sub = D.filter_predictions(df, player_ids=[pid] if pid else None, opponent_id=opp, surfaces=surfaces or None,
                               years=years, levels=levels or None, phases=phases or None)  # fmt: skip
    if sub.height == 0:
        st.info(
            "No recorded predictions match these filters. Widen the seasons or remove a filter.",
            icon=":material/filter_alt_off:",
        )
        return
    m = D.metrics(sub)
    c = st.columns(5)
    kpi(c[0], "Matches", f"{sub.height:,}")
    kpi(
        c[1],
        "M4A log loss",
        f"{m['p_m4a']['log_loss']:.4f}",
        "Mean negative log-likelihood; lower is better.",
        delta=f"{m['p_m4a']['log_loss'] - m['p_elo']['log_loss']:+.4f} vs Elo",
    )
    kpi(c[2], "Elo log loss", f"{m['p_elo']['log_loss']:.4f}")
    kpi(c[3], "M4A accuracy", f"{m['p_m4a']['accuracy']:.1%}", "Share of matches the favourite won.")
    kpi(c[4], "Elo accuracy", f"{m['p_elo']['accuracy']:.1%}")
    if sub.height < 200:
        st.caption(f"Small sample ({sub.height} matches): metrics on subsets this size are noisy.")
    if pid:
        view = D.orient(sub, pid).sort("tournament_date", "match_id", descending=True)
        table = view.select(pl.col("tournament_date").cast(pl.Utf8).alias("event start"), pl.col("tournament_name").alias("tournament"),
                            pl.col("tournament_level").replace_strict(D.LEVELS, default=None).alias("category"),
                            "surface", "round", "opponent", pl.col("p_m4a_player").alias("M4A P(win)"),
                            pl.col("p_elo_player").alias("Elo P(win)"),
                            pl.when(pl.col("player_won")).then(pl.lit("Won")).otherwise(pl.lit("Lost")).alias("result"),
                            "score")  # fmt: skip
    else:
        view = sub.sort("tournament_date", "match_id", descending=True)
        table = view.select(pl.col("tournament_date").cast(pl.Utf8).alias("event start"), pl.col("tournament_name").alias("tournament"),
                            pl.col("tournament_level").replace_strict(D.LEVELS, default=None).alias("category"),
                            "surface", "round", pl.col("player_a_name").alias("player A"),
                            pl.col("player_b_name").alias("player B"), pl.col("p_m4a").alias("M4A P(A)"),
                            pl.col("p_elo").alias("Elo P(A)"),
                            pl.when(pl.col("a_won")).then(pl.col("player_a_name")).otherwise(pl.col("player_b_name"))
                            .alias("winner"), "score")  # fmt: skip
    prob_cfg = {k: st.column_config.ProgressColumn(k, format="%.3f", min_value=0.0, max_value=1.0)
                for k in table.columns if "P(" in k}  # fmt: skip
    st.dataframe(table.head(2000), hide_index=True, column_config=prob_cfg, height=380)
    if table.height > 2000:
        st.caption(f"Showing the 2,000 most recent of {table.height:,} matches; metrics and charts use all of them.")
    st.subheader("Inspect one prediction")
    recent = view.head(500)
    labels = [f"{r['tournament_name']} {r['season']} {r['round']} · {r['player_a_name']} vs "
              f"{r['player_b_name']}" for r in recent.iter_rows(named=True)]  # fmt: skip
    i = st.selectbox("Match", range(len(labels)), format_func=lambda k: labels[k], key="ex_match")
    rec = D.lookup(df, recent["match_id"][i])
    if rec:
        c = st.columns(4)
        kpi(
            c[0],
            f"M4A P({rec['player_a_name']})",
            f"{rec['p_m4a']:.3f}",
            f"Favourite: {rec['m4a_favourite']}. Per-match log loss {rec['log_loss']['p_m4a']:.3f}.",
        )
        kpi(
            c[1],
            f"Elo P({rec['player_a_name']})",
            f"{rec['p_elo']:.3f}",
            f"Favourite: {rec['elo_favourite']}. Per-match log loss {rec['log_loss']['p_elo']:.3f}.",
        )
        kpi(c[2], "Winner", rec["winner"], rec["score"] or None)
        kpi(c[3], "Training cutoff", str(rec["training_cutoff"]), "The walk-forward fit used only labels released "
            f"before this date. Prediction batch {rec['prediction_batch']}.")  # fmt: skip
        st.caption(f"Event starting {rec['tournament_date']}. Weekly batching: every match in a batch is predicted from the state frozen at the batch start, "
                   "and results enter the state when their release batch closes. The archive has no match dates, so "
                   "this is an approximation documented in the Elo and modeling reports.")  # fmt: skip
    st.subheader("Errors and confidence")
    t1, t2, t3, t4 = st.tabs(["Confidence", "Accuracy by confidence", "M4A vs Elo", "Largest misses"])
    conf = D.confidence(sub)
    with t1:
        plot(charts.confidence_hist(conf, "How confident were the models?"))
        st.caption("Probability each model gave to the player it favoured. M4A is more spread out than Elo.")
    with t2:
        plot(charts.accuracy_by_confidence(conf, "Did the favourite win as often as predicted?"))
        st.caption("Points on the dotted line mean the stated confidence matched the observed win rate.")
    with t3:
        plot(charts.probability_scatter(sub, "M4A against surface Elo, match by match"))
        st.caption("Each point is a match (sampled above 6,000). Distance from the diagonal shows where the serve "
                   "and return inputs moved M4A away from Elo.")  # fmt: skip
    with t4:
        miss = (sub.with_columns(pl.max_horizontal("p_m4a", 1 - pl.col("p_m4a")).alias("conf"),
                                 ((pl.col("p_m4a") >= 0.5) != pl.col("a_won")).alias("miss"))
                .filter(pl.col("miss")).sort("conf", descending=True).head(25))  # fmt: skip
        st.dataframe(miss.select(pl.col("tournament_date").cast(pl.Utf8).alias("event start"), pl.col("tournament_name").alias("tournament"),
                                 "round", pl.col("player_a_name").alias("player A"), pl.col("player_b_name")
                                 .alias("player B"), pl.col("p_m4a").round(3).alias("M4A P(A)"),
                                 pl.col("p_elo").round(3).alias("Elo P(A)"),
                                 pl.when(pl.col("a_won")).then(pl.col("player_a_name")).otherwise(
                                     pl.col("player_b_name")).alias("winner"), "score"), hide_index=True)  # fmt: skip
        st.caption("The 25 matches where M4A was most confident and wrong in the current selection.")


# ---------------------------------------------------------------------------------------------------- comparison
def comparison() -> None:
    st.title("Player comparison")
    snap, pl_, preds = snapshots(), players(), predictions()
    if snap is None or pl_ is None:
        missing_dataset("The player comparison")
        return
    st.info("**No win probability is shown here.** Prospective predictions are disabled, and a present-day "
            "probability would imply a forecast the system cannot currently make. The page compares the "
            "historical, point-in-time inputs the model used.", icon=":material/block:")  # fmt: skip
    last = snap["tournament_date"].max()
    c = st.columns([2, 3, 3, 2])
    with c[0]:
        as_of = st.date_input("Comparison date", value=last, min_value=date(1991, 1, 1), max_value=last, key="cmp_date")
    defaults = top_players(as_of)
    opts = D.player_options(pl_)
    with c[1]:
        a = player_picker("Player 1", opts, "cmp_a", default_id=defaults[0] if defaults else None)
    with c[2]:
        b = player_picker("Player 2", opts, "cmp_b", default_id=defaults[1] if len(defaults) > 1 else None)
    with c[3]:
        surface = st.selectbox("Surface", ["Hard", "Clay", "Grass", "Carpet"], key="cmp_surface")
    if a == b:
        st.warning("Choose two different players.")
        return
    st.caption(f"State as of **{as_of}**: the pre-match values entering each player's last recorded tour-level "
               "match on or before that date. That match's result and anything later are excluded, as in the "
               "frozen pipeline.")  # fmt: skip
    cmp = D.comparison(snap, a, b, as_of, surface)
    names = {r["player_id"]: player_label(r) for r in pl_.filter(pl.col("player_id").is_in([a, b])).to_dicts()}
    if not any(cmp[p]["last"] for p in (a, b)):
        st.info("Neither player has a recorded tour-level match on or before this date.")
        return
    rows = []

    def val(p, key, field, fmt):
        r = cmp[p][key]
        v = None if r is None else r.get(field)
        return "–" if v is None else fmt(v)

    def line(label, key, field, fmt):
        rows.append(
            {"indicator": label, names.get(a, a): val(a, key, field, fmt), names.get(b, b): val(b, key, field, fmt)}
        )

    def pct(v):
        return f"{v:.1%}"

    line("Last recorded event (start date)", "last", "tournament_date", str)
    line("Global Elo", "last", "elo_global", lambda v: f"{v:.0f}")
    line("Global Elo matches", "last", "global_matches", lambda v: f"{v:,}")
    line(f"{surface} Elo", "surface", "elo_surface", lambda v: f"{v:.0f}")
    line(f"{surface} Elo matches", "surface", "surface_matches", lambda v: f"{v:,}")
    line(f"Last {surface} event (start date)", "surface", "tournament_date", str)
    line("Serve points won (shrunk)", "last", "spw_shrunk", pct)
    line("Return points won (shrunk)", "last", "rpw_shrunk", pct)
    line("Serve points won, 52-week decay", "last", "spw_decay52w", pct)
    line("Return points won, 52-week decay", "last", "rpw_decay52w", pct)
    line("Matches with statistics", "last", "stat_matches", lambda v: f"{v:,}")
    line("Service points observed", "last", "service_points", lambda v: f"{v:,}")
    line("Effective service sample, 52 weeks", "last", "service_ess_decay52w", lambda v: f"{v:,.0f}")
    line("Weight of the prior in the serve rate", "last", "spw_prior_weight", pct)
    line("Serve rate dominated by the prior", "last", "service_prior_dominated", lambda v: "yes" if v else "no")
    line("Weeks since last statistics", "last", "weeks_since_last_stats", lambda v: f"{v:,}")
    st.dataframe(pl.DataFrame(rows), hide_index=True)
    with st.expander("How to read shrinkage and missing values"):
        st.markdown("- **Shrunk rates** blend a player's observed serve/return points with a population prior; the "
                    "fewer points observed, the more weight the prior gets (shown as *weight of the prior*).\n"
                    "- **52-week decay rates** emphasise recent form. When a player has no recent statistics the "
                    "model sets the input to neutral and switches on a *missing* indicator instead of guessing.\n"
                    "- **Elo** starts every new player at the initial rating; the number of rated matches shows how "
                    "established each rating is.\n- A dash means the indicator was not available at that point "
                    "(for example, no statistics before 1991 or no matches on that surface).")  # fmt: skip
    hist = {names.get(p, str(p)): D.rating_history(snap, p, as_of) for p in (a, b)}
    t1, t2 = st.tabs(["Global Elo history", f"{surface} Elo history"])
    with t1:
        plot(charts.rating_lines(hist, "elo_global", "Global Elo entering each match"))
    with t2:
        plot(
            charts.rating_lines(
                {k: v.filter(pl.col("surface") == surface) for k, v in hist.items()},
                "elo_surface",
                f"{surface} Elo entering each {surface} match",
            )
        )
    st.subheader("Recent form")
    cols = st.columns(2)
    for col, p in zip(cols, (a, b), strict=True):
        f = D.recent_form(snap, p, as_of)
        opp_names = dict(pl_.select("player_id", "name").iter_rows())
        with col:
            st.markdown(f"**{names.get(p, p)}**, last {f.height} decided matches before {as_of}")
            if f.height:
                st.dataframe(f.select(pl.col("tournament_date").cast(pl.Utf8).alias("event start"), pl.col("tournament_name").alias("event"),
                                      "round", "surface", pl.col("opponent_id").replace_strict(opp_names, default=None)
                                      .alias("opponent"), pl.when(pl.col("won")).then(pl.lit("W")).otherwise(pl.lit("L"))
                                      .alias("result")), hide_index=True)  # fmt: skip
            else:
                st.caption("No decided tour-level matches before this date.")
    if preds is not None:
        h = D.h2h(preds, a, b, as_of)
        st.subheader("Recorded predictions when they met")
        if h.height:
            st.dataframe(h.select(pl.col("tournament_date").cast(pl.Utf8).alias("event start"), pl.col("tournament_name").alias("event"),
                                  "round", "surface", pl.col("p_m4a_player").round(3).alias(f"M4A P({names.get(a, a)})"),
                                  pl.col("p_elo_player").round(3).alias(f"Elo P({names.get(a, a)})"),
                                  pl.when(pl.col("player_won")).then(pl.lit(names.get(a, "")))
                                  .otherwise(pl.lit(names.get(b, ""))).alias("winner"), "score"), hide_index=True)  # fmt: skip
            st.caption("Historical pre-match predictions recorded by the frozen evaluation (2002 onward), not a "
                       "forecast for a future meeting.")  # fmt: skip
        else:
            st.caption("No recorded head-to-head predictions in the evaluation period up to this date.")


@st.cache_data(show_spinner=False)
def top_players(as_of: date) -> list[int]:
    snap = snapshots()
    recent = snap.filter(pl.col("tournament_date").is_between(as_of - timedelta(days=365), as_of))
    latest = recent.sort("tournament_date").group_by("player_id").agg(pl.col("elo_global").last())
    return latest.sort("elo_global", descending=True).head(2)["player_id"].to_list()


# ---------------------------------------------------------------------------------------------------- evaluation
def evaluation() -> None:
    st.title("Model evaluation")
    st.warning("**Not an untouched test.** The development period (2002–2009) was used to select M4A and its "
               "regularisation; the exploratory period (2010–2026) was seen while the Elo baseline was built. "
               "Treat these results as careful in-sample-period evidence. Intervals come from the published weekly "
               "block bootstrap; nothing here claims significance beyond them.", icon=":material/science:")  # fmt: skip
    st.caption(WALK_FORWARD)
    pub = published()
    rows = D.headline(pub)
    if not rows:
        st.info("The published modeling report is missing from this checkout.")
        return
    for r in rows:
        r["label"] = r["phase"].capitalize()
    st.subheader("Headline metrics (as published)")
    c = st.columns(3)
    with c[0]:
        plot(charts.metric_bars(rows, "log_loss", "Log loss (lower is better)"))
    with c[1]:
        plot(charts.metric_bars(rows, "brier", "Brier score (lower is better)"))
    with c[2]:
        plot(charts.metric_bars(rows, "accuracy", "Accuracy (higher is better)", ".1%"))
    st.dataframe(pl.DataFrame([{"period": D.PHASES[r["phase"]], "matches": r["matches"],
                                "Δ log loss": round(r["d_log_loss"], 5), "95% CI": D.fmt_ci(r["ll_ci95"], 5),
                                "Δ Brier": round(r["d_brier"], 5), "95% CI ": D.fmt_ci(r["brier_ci95"], 5),
                                "ECE M4A": round(r["m4a"]["ece"], 4), "ECE Elo": round(r["elo"]["ece"], 4),
                                "bootstrap": f"{r['bootstrap']['samples']:,} resamples of {r['bootstrap']['clusters']} weeks"}
                               for r in rows]), hide_index=True)  # fmt: skip
    st.caption("**Log loss** penalises confident mistakes heavily; **Brier score** is the mean squared error of the "
               "probability; **accuracy** counts how often the favourite won; **ECE** (expected calibration error) "
               "is the average gap between predicted and observed frequencies across ten probability bins.")  # fmt: skip
    st.subheader("Calibration")
    phase = st.radio("Period", ["development", "exploratory"], horizontal=True, format_func=str.capitalize,
                     key="ev_cal_phase")  # fmt: skip
    cal = pub["calibration"].get(phase)
    if cal is not None:
        curves = {"M4A": cal.filter(pl.col("model") == "M4A+raw"), "Surface Elo": cal.filter(pl.col("model") == "M0")}
        plot(charts.reliability(curves, f"Reliability diagram, {phase} period"))
        st.caption(
            "Each point is a 0.1-wide probability bin; marker size grows with the number of matches. Points "
            "on the diagonal are perfectly calibrated."
        )
    st.subheader("Where the difference comes from")
    sg = pub["subgroups"].get(phase)
    if sg is not None:
        t = st.tabs(["Surface", "Tournament category", "Elo confidence", "Statistics history"])
        for tab, group, labeler in ((t[0], "surface", None), (t[1], "tournament_level", D.LEVELS),
                                    (t[2], "elo_confidence_pct", None), (t[3], "sample_size", None)):  # fmt: skip
            with tab:
                g = sg.filter(pl.col("group") == group)
                if labeler:
                    g = g.with_columns(pl.col("level").replace_strict(labeler, default=pl.col("level")))
                if g.height == 0:
                    st.caption("No subgroup published for this period.")
                    continue
                g = g.with_columns(pl.format("{} (n={})", "level", "n").alias("label"))
                plot(charts.delta_bars(g, "label", "d_M4A+raw", "lo_M4A+raw", "hi_M4A+raw",
                                                  f"Log-loss difference by {group.replace('_', ' ')}, {phase}"))  # fmt: skip
        st.caption("Bars show M4A − surface Elo with published 95% block-bootstrap intervals. Small groups have wide "
                   "intervals; overlapping zero means no clear difference in that group. Much of the gain is a "
                   "cold-start correction for players with little rating history.")  # fmt: skip
    st.subheader("Stability over time (as published)")
    per, folds = pub.get("periods"), pub.get("folds")
    tbl = []
    if folds is not None:
        tbl += [{"label": f"{r['fold']} (dev)", "n": r["rows"], "d": r["d_M4A+raw"], "lo": r["lo_M4A+raw"],
                 "hi": r["hi_M4A+raw"]} for r in folds.iter_rows(named=True)]  # fmt: skip
    if per is not None:
        tbl += [{"label": f"{r['period']} (exploratory)", "n": r["rows"], "d": r["d"], "lo": r["lo"], "hi": r["hi"]}
                for r in per.iter_rows(named=True)]  # fmt: skip
    if tbl:
        plot(charts.delta_bars(pl.DataFrame(tbl), "label", "d", "lo", "hi", "Log-loss difference by period"))
    st.subheader("Paired comparison on the recorded predictions")
    df = predictions()
    if df is None:
        missing_dataset("The paired comparison")
        return
    c = st.columns([2, 2, 3])
    with c[0]:
        ph = st.multiselect("Period", list(D.PHASES), default=list(D.PHASES), format_func=str.capitalize, key="ev_ph")
    with c[1]:
        surf = st.multiselect("Surface", sorted(df["surface"].drop_nulls().unique().to_list()), key="ev_surf")
    with c[2]:
        lv = st.multiselect("Tournament category", list(D.LEVELS), format_func=D.LEVELS.get, key="ev_lv")
    sub = D.filter_predictions(df, phases=ph or None, surfaces=surf or None, levels=lv or None)
    if sub.height == 0:
        st.info("No matches for this selection.")
        return
    p = D.paired(sub)
    better = float((p["d_log_loss"] < 0).mean())
    m = D.metrics(sub)
    c = st.columns(4)
    kpi(c[0], "Matches", f"{sub.height:,}")
    kpi(c[1], "Mean Δ log loss", f"{p['d_log_loss'].mean():+.5f}", "M4A − Elo per match, averaged; negative "
        "favours M4A.")  # fmt: skip
    kpi(
        c[2],
        "Matches where M4A had lower loss",
        f"{better:.1%}",
        "Per-match comparison. The average gain is concentrated in a minority of "
        "matches, mostly players with little rating history; in most matches the two probabilities are close.",
    )
    kpi(c[3], "ECE M4A / Elo", f"{m['p_m4a']['ece']:.4f} / {m['p_elo']['ece']:.4f}")
    plot(charts.cumulative_delta(p, "Paired log-loss difference by season"))
    st.caption("Computed from the published prediction records with the project's scorer. With no filters the "
               "values reproduce the published report exactly (verified by the test suite). Point estimates only: "
               "subsets have no bootstrap interval here.")  # fmt: skip


# ------------------------------------------------------------------------------------------------- data & status
def data_status() -> None:
    st.title("Data quality and operational status")
    g = D.gates()
    st.error("**Live predictions are unavailable.** The prospective prediction gate is "
             f"**{g.prospective_predictions}** and the historical backfill is **{g.backfill}**. This site shows "
             "retrospective research only.", icon=":material/lock:")  # fmt: skip
    c = st.columns(4)
    kpi(c[0], "Prospective predictions", g.prospective_predictions, "config/prospective.toml [status]")
    kpi(c[1], "Historical backfill", g.backfill, "Complete only when results reach the prospective start.")
    kpi(
        c[2],
        "Availability safeguard",
        "enabled" if g.availability_policy_enabled else "disabled",
        "state_availability_v1: when results may enter a forecast's state (proposal, not active).",
    )
    kpi(
        c[3],
        "External timestamping",
        ", ".join(g.external_publishers_enabled) or "disabled",
        "Independent checkpoint publication (GitHub, RFC 3161).",
    )
    st.subheader("Historical coverage")
    snap, man, stt = snapshots(), manifest(), status()
    if snap is not None:
        plot(charts.coverage_bars(D.season_coverage(snap), "Tour-level matches per season (1991 onward)",
                                             highlight=(2002, 2026)))  # fmt: skip
        st.caption("Blue seasons are inside the M4A evaluation (2002–2026). The archive itself starts in 1968; match "
                   "statistics start in 1991. 2020 is short because of the pandemic suspension; 2026 runs to Roland "
                   "Garros (May).")  # fmt: skip
    pub = published()
    b = (pub.get("build") or {}).get("rows", {})
    cov = (man or {}).get("coverage", {})
    c = st.columns(3)
    kpi(
        c[0],
        "Latest verified archive event",
        cov.get("predictions", ["", "–"])[1] or "–",
        "Roland Garros 2026, the last event in the pinned archive.",
    )
    kpi(c[1], "Validated archive matches", f"{b.get('validated', 0):,}")
    kpi(c[2], "Archive commit", ((stt or {}).get("archive_commit") or "–")[:12])
    st.subheader("After the archive: the 2026 backfill")
    runs = (stt or {}).get("backfill_runs", [])
    snaps = {s["version"]: s for s in (stt or {}).get("snapshots", [])}
    if runs:
        st.markdown(
            "Results from June to October 2026 were reconstructed from an owner-supplied export into "
            "**local, inactive research snapshots**. They are results only (the export's serve statistics "
            "failed the compatibility check) and are not published here; the active state is still version 0."
        )
        st.dataframe(pl.DataFrame([{"snapshot": f"v{r['version']}", "state": (snaps.get(r["version"]) or {}).get(
            "state_snapshot_id", "–"), "active": "yes" if (snaps.get(r["version"]) or {}).get("active") else "no",
            "matches added": r["accepted_matches"], "Elo updates": r["elo_updates"],
            "quarantined matches": r["quarantined_matches"], "unresolved identities": r["unresolved_identities"],
            "verified": "yes" if r["verification_passed"] else "no"} for r in runs]), hide_index=True)  # fmt: skip
        st.caption("Aggregate counts only. Record-level backfill data comes from a licensed source and stays local.")
    else:
        st.caption("No backfill information in this checkout.")
    st.subheader("Freshness of the active state")
    fr = (stt or {}).get("freshness_probe", {})
    if fr:
        names = {"last_eligible_result": "Last eligible result", "last_point_statistics": "Last point statistics",
                 "missing_backfill_weeks": "Weeks without results", "opponent_fit_age_weeks": "Opponent model age",
                 "player_recent_history": "Recent player history", "source_age_days": "Source age",
                 "unresolved_identities": "Unresolved identities"}  # fmt: skip
        st.dataframe(pl.DataFrame([{"check": names.get(k, k), "status": v["status"],
                                    "value": "–" if v.get("value") is None else str(v["value"])}
                                   for k, v in fr.items()]), hide_index=True)  # fmt: skip
        st.caption("Freshness probe of the active state (version 0): serve/return statistics end with the archive "
                   "in May 2026, so any upcoming match would be predicted from stale inputs.")  # fmt: skip
    st.subheader("Why live predictions are unavailable, and what would enable them")
    st.markdown("""
1. **Fresh serve/return statistics.** The model's recency inputs need point-level statistics up to the week of the
   match. The archive ends in May 2026, and the backfilled results carry no compatible statistics.
2. **A verified live data provider.** Schedules and results must arrive automatically from an authorised source.
3. **Complete, reviewed history.** The backfill must reach the prospective start with every player identity
   resolved; 21 identities remain under review.
4. **Verifiable timing.** Forecasts are committed to an append-only, hash-chained ledger, but a local chain alone
   cannot prove *when* a forecast was made. An independent timestamp (an RFC 3161 authority or an independently
   controlled checkpoint repository) is required before any forecast can count as genuinely prospective.
5. **A reviewed decision.** Only then can the status gate change, through a reviewed configuration commit.
""")


SECTIONS = {"Overview": overview, "Match explorer": explorer, "Player comparison": comparison,
            "Model evaluation": evaluation, "Data & status": data_status}  # fmt: skip
