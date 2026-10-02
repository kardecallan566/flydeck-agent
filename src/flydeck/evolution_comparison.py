"""Descriptive, same-seed MaleCNS vs no-MaleCNS population comparison."""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path


def _rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _overview(root: Path) -> dict:
    population = _rows(root / "final_population.csv")
    history = _rows(root / "all_block_results.csv")
    validation = [r for r in history if r["stage"] == "validation"]
    audit = [r for r in history if r["stage"] == "historical_audit"]
    v = {r["agent_id"]: r for r in validation}
    a = {r["agent_id"]: r for r in audit}
    if len(v) != len(population) or len(a) != len(population):
        raise ValueError("comparison requires one validation and audit row per final agent")

    def metrics(rows: list[dict]) -> dict:
        return {
            "agents": len(rows),
            "profitable": sum(r["profitable"].lower() == "true" for r in rows),
            "risk_survived": sum(r["risk_survived"].lower() == "true" for r in rows),
            "survived_risk_and_profit": sum(r["survived"].lower() == "true" for r in rows),
            "statistical_evidence_scenario_only": sum(
                r["statistical_evidence"].lower() == "true" for r in rows
            ),
            "median_equity": statistics.median(float(r["equity"]) for r in rows),
            "median_accuracy": statistics.median(float(r["accuracy"]) for r in rows),
            "median_coverage": statistics.median(float(r["coverage"]) for r in rows),
        }

    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    selected = summary["finalists_selected_before_audit_or_recent"]
    results = {
        "all_validation": metrics(validation),
        "all_historical_audit": metrics(audit),
        "positive_in_both": sum(
            v[agent]["profitable"].lower() == "true"
            and a[agent]["profitable"].lower() == "true"
            for agent in v
        ),
        "selected_on_validation": selected,
        "selected_audit": metrics([a[agent] for agent in selected]) if selected else None,
    }
    sealed = root / "recent_sealed_results.csv"
    if sealed.exists():
        recent_rows = _rows(sealed)
        recent_by_id = {r["agent_id"]: r for r in recent_rows}
        results["sealed_recent_all"] = metrics(recent_rows)
        results["sealed_recent_preselected"] = (
            metrics([recent_by_id[agent] for agent in selected]) if selected else None
        )
    return results


def write_ablation_comparison(root: Path) -> dict:
    """Descriptive population-arm comparison, not a significance claim.

    The arms share a seed and windows, but evolution can create different
    descendants. Hence per-finalist matching would be invalid.
    """
    control = _overview(root / "without_fly")
    treatment = _overview(root / "with_fly")
    report = {
        "type": "population-level controlled ablation",
        "paired_seed_and_time_windows": True,
        "caution": (
            "Descendant IDs are not matched across arms; inspect distributions "
            "and new forward holdouts, not only the strongest agent. Both arms "
            "use Binance next-close unless official PancakeSwap rounds provided."
        ),
        "without_fly": control,
        "with_fly": treatment,
        "delta_with_minus_without": {
            phase: {
                key: treatment[phase][key] - control[phase][key]
                for key in ("profitable", "risk_survived", "survived_risk_and_profit",
                            "statistical_evidence_scenario_only", "median_equity",
                            "median_accuracy", "median_coverage")
            }
            for phase in ("all_validation", "all_historical_audit")
        },
    }
    if "sealed_recent_all" in control and "sealed_recent_all" in treatment:
        report["delta_with_minus_without"]["sealed_recent_all"] = {
            key: treatment["sealed_recent_all"][key] - control["sealed_recent_all"][key]
            for key in ("profitable", "risk_survived", "survived_risk_and_profit",
                        "statistical_evidence_scenario_only", "median_equity",
                        "median_accuracy", "median_coverage")
        }
    (root / "ablation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    with (root / "ablation_report.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["arm", "phase", "agents", "profitable",
                                "risk_survived", "survived_risk_and_profit",
                                "statistical_evidence_scenario_only",
                                "median_equity", "median_accuracy", "median_coverage"],
        )
        writer.writeheader()
        for arm, stats in (("without_fly", control), ("with_fly", treatment)):
            for phase in ("all_validation", "all_historical_audit", "sealed_recent_all"):
                if phase in stats:
                    writer.writerow({"arm": arm, "phase": phase, **stats[phase]})
    return report
