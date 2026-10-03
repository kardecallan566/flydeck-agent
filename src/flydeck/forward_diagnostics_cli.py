"""Offline audit of a finished three-arm frozen-forward run.

No market download, training, model selection, probability of real profit or
side effects outside the explicitly requested output directory.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def _csv(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"missing forward result: {path}")
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _f(row: dict, field: str) -> float:
    return float(row[field])


def _is_true(value: object) -> bool:
    return str(value).lower() == "true"


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def arm_audit(folder: Path) -> dict:
    results = _csv(folder / "all_agents_forward.csv")
    meta = json.loads((folder / "forward_summary.json").read_text(encoding="utf-8"))
    if len({a["agent_id"] for a in results}) != len(results):
        raise ValueError(f"duplicate agent ids in {folder}")
    active = [a for a in results if int(a["entered"]) > 0]
    preselected = [a for a in results if _is_true(a["preselected_finalist"])]
    paid = [
        a for a in results if int(a["entered"]) >= 80
        and _f(a, "coverage") >= 0.05
    ]
    by_family = defaultdict(list)
    for a in results:
        by_family[a["family"]].append(a)

    def group(xs: list[dict]) -> dict:
        have_entries = [a for a in xs if int(a["entered"]) > 0]
        # The raw forward CSV uses accuracy=0 for WAIT-only agents. Do NOT
        # include those in an average or median accuracy.
        return {
            "agents": len(xs),
            "agents_with_entries": len(have_entries),
            "agents_without_entries": len(xs) - len(have_entries),
            "entered_total_correlated_not_independent": sum(int(a["entered"]) for a in xs),
            "median_accuracy_among_active": _median(
                [_f(a, "accuracy") for a in have_entries]
            ),
            "median_coverage": _median([_f(a, "coverage") for a in xs]),
            "median_window_return_pct": _median([_f(a, "window_return_pct") for a in xs]),
            "positive": sum(_f(a, "window_return_pct") > 0 for a in xs),
            "max_entries": max((int(a["entered"]) for a in xs), default=0),
        }
    out = {
        "arm": meta["arm"],
        "source_sha256": meta["source_csv_sha256"],
        "checkpoint_sha256": meta["checkpoint_sha256"],
        "new_candles": meta["new_candles"],
        "eligible_maximum": max((int(a["eligible"]) for a in results), default=0),
        "frozen_weights": meta["evaluation_without_weight_updates"],
        "all": group(results),
        "preselected_validation_finalists": group(preselected),
        "risk_participation_threshold_met": len(paid),
        "families": {k: group(v) for k, v in sorted(by_family.items())},
        "preselected_rows": [
            {
                "agent_id": a["agent_id"], "family": a["family"],
                "entered": int(a["entered"]),
                "accuracy": _f(a, "accuracy") if int(a["entered"]) else None,
                "coverage": _f(a, "coverage"),
                "return_pct": _f(a, "window_return_pct"),
                "up": int(a["up"]), "down": int(a["down"]),
            } for a in preselected
        ],
    }
    return out


def paired_same_weights(treated: Path, masked: Path) -> dict:
    a = _csv(treated / "all_agents_forward.csv")
    b = _csv(masked / "all_agents_forward.csv")
    ad = {r["agent_id"]: r for r in a}
    bd = {r["agent_id"]: r for r in b}
    if set(ad) != set(bd):
        raise ValueError("Fly/masked arms have different agent IDs")
    ca = json.loads((treated / "forward_summary.json").read_text())
    cb = json.loads((masked / "forward_summary.json").read_text())
    if (ca["checkpoint_sha256"] != cb["checkpoint_sha256"]
        or ca["source_csv_sha256"] != cb["source_csv_sha256"]):
        raise ValueError("cannot pair different frozen weights or market data")
    diffs = []
    altered_entry_counts = 0
    for key in sorted(ad):
        x, y = ad[key], bd[key]
        diff = _f(x, "window_return_pct") - _f(y, "window_return_pct")
        if int(x["entered"]) != int(y["entered"]):
            altered_entry_counts += 1
        diffs.append({
            "agent_id": key, "family": x["family"],
            "fly_entries": int(x["entered"]),
            "masked_entries": int(y["entered"]),
            "delta_entries": int(x["entered"]) - int(y["entered"]),
            "fly_return_pct": _f(x, "window_return_pct"),
            "masked_return_pct": _f(y, "window_return_pct"),
            "delta_return_percentage_points": diff,
            "preselected": _is_true(x["preselected_finalist"]),
        })
    # Actual same-time paired actions are available for ORIGINAL finalists.
    def traces(folder: Path) -> dict[tuple[str, int], dict]:
        out = {}
        for row in _csv(folder / "preselected_decisions.csv"):
            key = (row["agent_id"], int(row["feature_candle_open_ms"]))
            if key in out:
                raise ValueError("duplicate final-candidate action in trace")
            out[key] = row
        return out
    tr, mr = traces(treated), traces(masked)
    if set(tr) != set(mr):
        raise ValueError("paired final candidates have mismatched observation timestamps")
    changed = directional = both_enter = other = 0
    first_entries = second_entries = 0
    for key in tr:
        x, y = tr[key], mr[key]
        first_entries += x["action"] != "WAIT"
        second_entries += y["action"] != "WAIT"
        if x["action"] != y["action"]:
            changed += 1
            if x["action"] != "WAIT" and y["action"] != "WAIT":
                directional += 1
            else:
                other += 1
        if x["action"] != "WAIT" and y["action"] != "WAIT":
            both_enter += 1
    return {
        "same_frozen_checkpoint_and_candles": True,
        "agents": len(diffs),
        "agents_with_changed_entry_counts": altered_entry_counts,
        "median_agent_return_delta_percentage_points": _median(
            [r["delta_return_percentage_points"] for r in diffs]
        ),
        "original_finalists_only": {
            "paired_decision_opportunities_correlated_across_agents": len(tr),
            "changed_actions": changed,
            "changed_direction_when_both_entered": directional,
            "entry_vs_wait_disagreements": other,
            "both_entered": both_enter,
            "entries_signal_on": first_entries,
            "entries_signal_masked": second_entries,
        },
        "per_agent_exploratory": diffs,
        "limitation": (
            "Masked neural input is out-of-distribution. This is a paired "
            "sensitivity diagnostic, not proof of causal value. Repeated "
            "candles across agents are correlated, no hypothesis test here."
        ),
    }


def diagnose(root: Path, *, output: Path | None = None) -> dict:
    with_arm = arm_audit(root / "with_fly")
    without = arm_audit(root / "without_fly")
    if with_arm["source_sha256"] != without["source_sha256"]:
        raise ValueError("the two training arms used different forward market data")
    masked_dir = root / "with_fly_signal_masked"
    paired = paired_same_weights(root / "with_fly", masked_dir) if masked_dir.exists() else None
    eligible = min(with_arm["eligible_maximum"], without["eligible_maximum"])
    report = {
        "research_only": True,
        "root": str(root),
        "source_sha256": with_arm["source_sha256"],
        "evaluated_opportunities": eligible,
        "minimum_required_entries_for_risk_survival": 80,
        "risk_threshold_possible_this_window": eligible >= 80,
        "with_fly": with_arm,
        "without_fly": without,
        "same_weight_neural_input_diagnostic": paired,
        "independent_profitability_verified": False,
        "recommendation": (
            "Not enough eligible candles for 80 entries; treat as an integration smoke test."
            if eligible < 80 else
            "Assess original preselected finalists in the NEXT independent period; "
            "do not select by this already inspected result."
        ),
    }
    if output is not None:
        if output.resolve() == (root / "frozen_comparison.json").resolve():
            raise ValueError("never overwrite the original frozen report")
        if output.exists():
            raise FileExistsError(f"diagnostic destination already exists: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main() -> int:
    p = argparse.ArgumentParser(description="Offline honest diagnostics from completed frozen 3-arm experiment")
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    try:
        result = diagnose(args.run, output=args.output)
    except (OSError, ValueError, KeyError) as exc:
        p.exit(2, f"Audit aborted: {exc}\n")
    print("Existing forward run:", args.run)
    print("Evaluable opportunities:", result["evaluated_opportunities"])
    for arm in ("with_fly", "without_fly"):
        r = result[arm]
        print(
            f"{arm}: {r['all']['agents_without_entries']}/{r['all']['agents']} "
            f"agents did not enter | median accuracy among ACTIVE agents: "
            f"{r['all']['median_accuracy_among_active']} | "
            f"original finalists with entries: "
            f"{r['preselected_validation_finalists']['agents_with_entries']}/"
            f"{r['preselected_validation_finalists']['agents']}"
        )
    if result["same_weight_neural_input_diagnostic"]:
        d = result["same_weight_neural_input_diagnostic"]["original_finalists_only"]
        print(f"Fly signal ON vs MASKED on original finalists: {d['changed_actions']} "
              f"changed actions over {d['paired_decision_opportunities_correlated_across_agents']} "
              f"paired observations.")
    print("NOT an independent statistical or real-profit conclusion.")
    if args.output:
        print("Full audit saved:", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
