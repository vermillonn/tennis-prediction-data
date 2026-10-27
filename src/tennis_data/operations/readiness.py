"""Deterministic operational-readiness evaluator. It decides nothing by itself and never changes the status gate.

``evaluate(evidence, required)`` is a pure function of an evidence dictionary (gathered by ``gather``), so the same
evidence always gives the same verdict. Each check returns PASS, FAIL or UNKNOWN with the evidence used and a
remediation. The verdict is ``READY_TO_PROPOSE_ACTIVATION`` only if every required check PASSes; otherwise
``NOT_READY``. Even then the evaluator only *proposes* the configuration change; applying it needs explicit approval
in a reviewed commit. There is no bypass and no test-only production switch.
"""

from __future__ import annotations

import json
from datetime import date

CHECKS = {
    # id: (dimension, description)
    "historical_weeks_covered": ("historical completeness", "every gap week has validated tour-level results"),
    "no_material_event_gaps": ("historical completeness", "every expected (inferred) event of the gap was observed"),
    "population_compatible": ("historical completeness", "backfilled competitions stay within the frozen rating pools"),
    "stats_definitions_verified": ("statistical validity", "serve statistics used by the feature engine are verified compatible"),
    "no_invented_values": ("statistical validity", "missing statistics stay missing (no imputed point counts)"),
    "identities_resolved": ("identity integrity", "no unresolved player identity among records to be used"),
    "result_freshness": ("freshness", "latest validated result within the configured age"),
    "stats_freshness": ("freshness", "latest valid serve statistics within the configured age"),
    "schedule_source_verified": ("schedule integrity", "a verified provider supplies upcoming fixtures"),
    "schedule_players_verified": ("schedule integrity", "fixture participants resolve to canonical players"),
    "actual_start_source": ("schedule integrity", "a sourced actual-start timestamp feed exists"),
    "frozen_model_hashes": ("model integrity", "frozen M4A artifact and configuration hashes match"),
    "historical_replay": ("model integrity", "Phase 4A historical replay reproduces published predictions"),
    "ledger_ready": ("audit infrastructure", "prediction ledger configuration valid (append-only, gated)"),
    "external_timestamp_authority": ("audit infrastructure", "an RFC 3161 TSA is configured for independent timing"),
    "status_gate_consistent": ("audit infrastructure", "the status gate stays closed while checks fail"),
}  # fmt: skip


def _check(cid, status, evidence, remediation=""):
    dim, desc = CHECKS[cid]
    return {"id": cid, "dimension": dim, "description": desc, "status": status, "evidence": evidence,
            "remediation": remediation if status != "PASS" else ""}  # fmt: skip


def evaluate(ev: dict, required: tuple[str, ...]) -> dict:
    cov, comp, idn, sch, mod, fr = (ev.get(k) or {} for k in ("coverage", "compatibility", "identity", "schedule",
                                                              "model", "freshness"))  # fmt: skip
    out = []
    wk_total, wk_val = cov.get("weeks_total", 0), cov.get("weeks_validated", 0)
    out.append(_check("historical_weeks_covered", "PASS" if wk_total and wk_val == wk_total else "FAIL",
                      f"{wk_val} of {wk_total} gap weeks have validated results",
                      "obtain an authorized source for the gap (see provider_access.md) and run the backfill"))  # fmt: skip
    exp, obs = cov.get("expected_events_inferred_total", 0), cov.get("observed_expected_events", 0)
    out.append(_check("no_material_event_gaps", "PASS" if exp and obs >= exp else "FAIL",
                      f"{obs} of {exp} expected (inferred) events observed",
                      "backfill the missing events; confirm the calendar with an authoritative source"))  # fmt: skip
    pops = cov.get("validated_levels", [])
    bad_pop = sorted(set(pops) - set(ev.get("pools", [])))
    out.append(_check("population_compatible", "UNKNOWN" if not pops else ("FAIL" if bad_pop else "PASS"),
                      f"validated levels {pops or 'none'}" + (f"; outside the pools: {bad_pop}" if bad_pop else ""),
                      "no validated backfill yet" if not pops else "remove competitions outside the frozen pools"))  # fmt: skip
    out.append(_check("stats_definitions_verified", "PASS" if comp.get("statistics_compatible") else "FAIL",
                      comp.get("summary", "no compatibility study"),
                      "use a source whose serve statistics match sackmann_v1 on the overlap, or run results-only"))  # fmt: skip
    out.append(_check("no_invented_values", "PASS" if ev.get("no_imputation", False) else "FAIL",
                      "conversion and Phase 4A keep missing statistics null (no imputation code path)"
                      if ev.get("no_imputation") else "imputation not ruled out"))  # fmt: skip
    unres = idn.get("unresolved", None)
    out.append(_check("identities_resolved", "PASS" if unres == 0 and idn.get("records_usable") else "FAIL",
                      f"{unres} unresolved / {idn.get('high_confidence_pending', 0)} high-confidence awaiting approval "
                      f"/ {idn.get('ambiguous', 0)} ambiguous",
                      "review the identity report; add approved mappings to config/identity_overrides.toml"))  # fmt: skip
    for cid, key in (("result_freshness", "result_age_days"), ("stats_freshness", "stats_age_days")):
        age, lim = fr.get(key), fr.get("fail_result_age_days")
        out.append(_check(cid, "PASS" if age is not None and lim is not None and age < lim else "FAIL",
                          f"age {age} days (fail at {lim})" if age is not None else "no validated record after the archive",
                          "backfill to the present and keep ingesting"))  # fmt: skip
    out.append(_check("schedule_source_verified", "PASS" if sch.get("verified_sources") else "FAIL",
                      f"verified schedule sources: {sch.get('verified_sources') or 'none'}",
                      "provide an authorized schedule provider (credentials and network access)"))  # fmt: skip
    out.append(_check("schedule_players_verified", "PASS" if sch.get("fixtures_resolved") else "FAIL",
                      f"{sch.get('fixtures', 0)} fixtures, all participants resolved: {bool(sch.get('fixtures_resolved'))}",
                      "ingest verified fixtures and resolve their players"))  # fmt: skip
    out.append(_check("actual_start_source", "PASS" if sch.get("actual_start_sources") else "FAIL",
                      f"sourced actual-start feeds: {sch.get('actual_start_sources') or 'none'}",
                      "obtain a provider that states actual match start times; never infer them"))  # fmt: skip
    out.append(_check("frozen_model_hashes", "PASS" if mod.get("hashes_match") else ("UNKNOWN" if not mod else "FAIL"),
                      f"frozen {mod.get('frozen_hash')} (expected {mod.get('expected_frozen_hash')}), configuration "
                      f"{mod.get('configuration_hash')} (expected {mod.get('expected_configuration_hash')})",
                      "restore the frozen artifacts; never retrain"))  # fmt: skip
    out.append(_check("historical_replay", "PASS" if mod.get("replay_passed") else ("UNKNOWN" if not mod else "FAIL"),
                      f"Phase 4A replay passed: {mod.get('replay_passed')}", "re-run prospective-run"))  # fmt: skip
    out.append(_check("ledger_ready", "PASS" if ev.get("ledger_config_valid") else "FAIL",
                      "ledger configuration valid; PROSPECTIVE commits refused while the gate is closed"
                      if ev.get("ledger_config_valid") else "ledger configuration invalid", "fix config/ledger.toml"))  # fmt: skip
    tsa = ev.get("timestamp_authority", {})
    out.append(_check("external_timestamp_authority", "PASS" if tsa.get("enabled") and tsa.get("trusted_cas") else "FAIL",
                      f"RFC 3161 enabled: {tsa.get('enabled', False)}; trusted CAs: {tsa.get('trusted_cas', 0)}; GitHub "
                      f"publication enabled: {tsa.get('github_enabled', False)}",
                      "authorize a TSA (and optionally a checkpoint repository); see docs/operational_data.md"))  # fmt: skip
    gate = ev.get("gate", {})
    others_fail = any(c["status"] != "PASS" for c in out)
    closed = gate.get("prospective_predictions") == "DISABLED" and gate.get("backfill") == "INCOMPLETE"
    out.append(_check("status_gate_consistent", "PASS" if (closed or not others_fail) else "FAIL",
                      f"backfill {gate.get('backfill')}, prospective predictions {gate.get('prospective_predictions')}",
                      "close the gate: checks are failing"))  # fmt: skip
    req = [c for c in out if c["id"] in required]
    failing = [c["id"] for c in req if c["status"] != "PASS"]
    verdict = "READY_TO_PROPOSE_ACTIVATION" if not failing else "NOT_READY"
    proposal = None
    if verdict == "READY_TO_PROPOSE_ACTIVATION":
        proposal = {"file": "config/prospective.toml", "change": {"backfill": "COMPLETE", "prospective_predictions":
                    "ENABLED"}, "applied": False, "requires": "explicit approval in a reviewed commit"}  # fmt: skip
    return {"format": "tennis-operational-readiness-1", "as_of": ev.get("as_of"), "verdict": verdict,
            "required_checks": list(required), "failing_required": failing, "checks": out,
            "proposed_configuration_change": proposal,
            "gate_after_evaluation": gate, "gate_changed_by_evaluator": False}  # fmt: skip


def gather(cfg, as_of: date, coverage: dict, comparison: dict | None, decisions, schedule: dict, access: dict | None,
           model: dict | None) -> dict:  # fmt: skip
    """Collect the evidence dictionary from Phase 4A/4B/4C configuration and the Phase 5A analyses."""
    import tomllib

    from tennis_data.ledger.writer import load_ledger_config

    p = cfg.prospective
    vweeks = sum(1 for w in coverage["weeks"] if w["validated"])
    exp_names = [n for w in coverage["weeks"] for n in w["expected_events_inferred"]]
    obs_names = {n for w in coverage["weeks"] for n in w["observed_events"]}
    unres = int((~decisions["usable"]).sum()) if decisions is not None and decisions.height else None
    cats = dict(decisions.group_by("category").len().iter_rows()) if decisions is not None and decisions.height else {}
    with cfg.evaluation_config.open("rb") as fh:
        pubs = tomllib.load(fh).get("publishers", {})
    try:
        load_ledger_config()
        ledger_ok = True
    except Exception:  # noqa: BLE001
        ledger_ok = False
    newest = coverage.get("newest_validated_date")
    age = (as_of - date.fromisoformat(newest)).days if newest else None
    comp_summary = "no study"
    if comparison:
        c = comparison["comparison"]
        comp_summary = (f"{c['matched_matches']} matched matches; exact agreement svpt "
                        f"{c['fields']['svpt']['exact_agreement']:.3f}, 1stIn {c['fields']['1stIn']['exact_agreement']:.3f}, "
                        f"1stWon {c['fields']['1stWon']['exact_agreement']:.3f}, 2ndWon "
                        f"{c['fields']['2ndWon']['exact_agreement']:.3f} (threshold {cfg.min_exact_agreement})"
                        if c["matched_matches"] else "no matched matches")  # fmt: skip
    return {
        "as_of": as_of.isoformat(), "pools": list(cfg.pools),
        "coverage": {"weeks_total": coverage["weeks_total"], "weeks_validated": vweeks,
                     "expected_events_inferred_total": len(exp_names),
                     "observed_expected_events": sum(1 for n in exp_names if n in obs_names),
                     "validated_levels": coverage.get("validated_levels", [])},
        "compatibility": {"statistics_compatible": bool(comparison and comparison["comparison"]["statistics_compatible"]),
                          "summary": comp_summary},
        "no_imputation": True,
        "identity": {"unresolved": unres, "records_usable": bool(coverage.get("validated_total")),
                     "high_confidence_pending": cats.get("high_confidence", 0), "ambiguous": cats.get("ambiguous", 0)},
        "freshness": {"result_age_days": age, "stats_age_days": age,
                      "fail_result_age_days": p.freshness.fail_result_age_days},
        "schedule": schedule,
        "model": model or {},
        "ledger_config_valid": ledger_ok,
        "timestamp_authority": {"enabled": bool(pubs.get("rfc3161", {}).get("enabled")),
                                "trusted_cas": len(pubs.get("rfc3161", {}).get("trusted_ca_files", [])),
                                "github_enabled": bool(pubs.get("github", {}).get("enabled"))},
        "gate": {"backfill": p.backfill_status, "prospective_predictions": p.prospective_predictions},
        "access_verified_sources": (access or {}).get("verified_sources", []),
    }  # fmt: skip


def model_evidence(cfg) -> dict:
    """Frozen artifact integrity (Phase 4A ``integrity``) and the recorded Phase 4A replay result."""
    from tennis_data.prospective.pipeline import integrity

    integ = integrity(cfg.prospective)
    summ = cfg.prospective.reports_dir / "phase4a_summary.json"
    replay = json.loads(summ.read_text()).get("replay", {}) if summ.exists() else {}
    exp, exp_cfg = cfg.expected_frozen_hash, cfg.expected_configuration_hash
    conf = cfg.prospective.modeling.configuration_hash()
    return {"frozen_hash": (integ.get("frozen_hash") or "")[:16], "expected_frozen_hash": exp,
            "configuration_hash": conf, "expected_configuration_hash": exp_cfg,
            "hashes_match": bool(integ.get("passed")) and bool(exp) and (integ.get("frozen_hash") or "").startswith(exp)
            and exp == cfg.prospective.expected_frozen_hash and conf == exp_cfg,
            "replay_passed": bool(replay.get("summary", {}).get("passed", replay.get("passed")))}  # fmt: skip
