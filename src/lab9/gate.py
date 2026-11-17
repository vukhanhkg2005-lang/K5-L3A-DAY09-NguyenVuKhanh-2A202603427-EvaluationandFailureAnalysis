"""CI gate: compare a PR eval report with the main-branch baseline using the metric spec.

    python -m lab9.gate --pr reports/eval_workflow.json --baseline reports/baseline_v1.json

exit 0 = merge, exit 1 = blocked. Safety metrics: max 0, no averaging away.
Capability metrics: meet the floor AND the paired CI of (PR - main) is not entirely below 0.
Thresholds may only get stricter on the same eval-set version (ratchet).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lab8.console import utf8_stdio
from lab9 import stats
from lab9.eval import SPEC, load_spec, per_item


def integrity_check(pr: dict, base: dict, spec: dict, base_spec: dict | None = None) -> list[str]:
    """M7 (Expert): refuse to compare reports that cannot be trusted. Return a list of problems
    (empty = trustworthy). A gate that passes a broken report is worse than no gate.
    Must catch at least:
      - any metric value in pr["metrics"] that is NaN or inf
      - len(pr["rows"]) != manifest items x len(manifest seeds)   (dropped / duplicated rows)
      - the (item_id, seed) set of PR differs from the baseline's (cherry-picked seeds/items)
      - pr["evaluator_failures"] non-empty, or a row whose value for a graded spec metric is
        None (a crashed grader silently scores as "missing" and drops out of the mean)
      - base_spec given and a threshold in spec is LOOSER than in base_spec for the same
        eval-set version (ratchet: max may not rise, min may not fall, a blocking metric may
        not become non-blocking, a metric may not disappear; the same for spec["slices"],
        including dropping no_significant_drop)
      - an item's slice tags in PR rows differ from the baseline's (relabelling an item out of a
        slice hides its regression from the slice gate)"""
    import math

    problems: list[str] = []

    # 1. Any metric value in pr["metrics"] that is NaN or inf
    for k, v in pr.get("metrics", {}).items():
        if isinstance(v, (int, float)) and (math.isnan(v) or math.isinf(v)):
            problems.append(f"metric {k} is {v}")

    # 2. Dropped or duplicated rows
    m = pr.get("manifest", {})
    expected_rows = m.get("items", 0) * len(m.get("seeds", []))
    if expected_rows > 0 and len(pr.get("rows", [])) != expected_rows:
        problems.append(f"row count {len(pr.get('rows', []))} != expected {expected_rows}")

    # 3. Cherry-picked seeds/items
    pr_pairs = {(r.get("item_id"), r.get("seed")) for r in pr.get("rows", [])}
    base_pairs = {(r.get("item_id"), r.get("seed")) for r in base.get("rows", [])}
    if base_pairs and pr_pairs != base_pairs:
        problems.append(f"PR (item_id, seed) set differs from baseline: {len(pr_pairs)} vs {len(base_pairs)}")

    # 4. Evaluator failures or rows with None for graded spec metrics
    if pr.get("evaluator_failures"):
        problems.append(f"evaluator_failures: {pr['evaluator_failures'][0]}")
    for r in pr.get("rows", []):
        for metric_obj in spec.get("metrics", []):
            mid = metric_obj["id"]
            if mid in r and r[mid] is None:
                problems.append(f"row {r.get('item_id')}@{r.get('seed')} has None for metric {mid}")
                break
        if problems and "has None for metric" in problems[-1]:
            break

    # 5. Slice tags changed between PR and baseline
    pr_slices = {r.get("item_id"): tuple(sorted(r.get("slices") or [])) for r in pr.get("rows", [])}
    base_slices = {r.get("item_id"): tuple(sorted(r.get("slices") or [])) for r in base.get("rows", [])}
    for item_id, base_s in base_slices.items():
        if item_id in pr_slices and pr_slices[item_id] != base_s:
            problems.append(f"slice tags for item {item_id} differ from baseline")
            break

    # 6. Ratchet check: base_spec given and a threshold in spec is looser than base_spec
    if base_spec is not None and spec.get("eval_set_version") == base_spec.get("eval_set_version"):
        base_metrics = {met["id"]: met for met in base_spec.get("metrics", [])}
        spec_metrics = {met["id"]: met for met in spec.get("metrics", [])}

        for mid, bm in base_metrics.items():
            if mid not in spec_metrics:
                problems.append(f"metric {mid} disappeared from spec")
                continue
            sm = spec_metrics[mid]
            if bm.get("blocking") and not sm.get("blocking"):
                problems.append(f"metric {mid} became non-blocking")

            b_th = bm.get("threshold", {})
            s_th = sm.get("threshold", {})
            if "max" in b_th and ("max" not in s_th or s_th["max"] > b_th["max"]):
                problems.append(f"metric {mid} threshold max loosened: {s_th.get('max')} > {b_th['max']}")
            if "min" in b_th and ("min" not in s_th or s_th["min"] < b_th["min"]):
                problems.append(f"metric {mid} threshold min loosened: {s_th.get('min')} < {b_th['min']}")
            if "max_increase" in b_th and ("max_increase" not in s_th or s_th["max_increase"] > b_th["max_increase"]):
                problems.append(f"metric {mid} max_increase loosened")
            if b_th.get("no_significant_drop") and not s_th.get("no_significant_drop"):
                problems.append(f"metric {mid} dropped no_significant_drop")

        base_slices_dict = {s["name"]: s for s in base_spec.get("slices", [])}
        spec_slices_dict = {s["name"]: s for s in spec.get("slices", [])}
        for sname, bs in base_slices_dict.items():
            if sname not in spec_slices_dict:
                problems.append(f"slice {sname} disappeared from spec")
                continue
            ss = spec_slices_dict[sname]
            if bs.get("blocking") and not ss.get("blocking"):
                problems.append(f"slice {sname} became non-blocking")
            if "min" in bs and ("min" not in ss or ss["min"] < bs["min"]):
                problems.append(f"slice {sname} min loosened: {ss.get('min')} < {bs['min']}")
            if bs.get("no_significant_drop") and not ss.get("no_significant_drop"):
                problems.append(f"slice {sname} dropped no_significant_drop")

    return problems


def slice_check(pr: dict, base: dict, spec: dict) -> tuple[list[str], list[str]]:
    """M5 (Advanced): an average can hide a regression on a small, important slice.
    For each entry of spec["slices"] ({"name", "metric", "min"?, "no_significant_drop"?, "blocking"}):
      - value = mean of per_item(pr, metric, slice_name=name)
      - fail if below "min", or if no_significant_drop and the paired CI over the slice's items
        lies entirely below 0
    Return (blocked ids like "slice:money_movement/pass_rate", report lines)."""
    blocked: list[str] = []
    lines: list[str] = []
    bs = spec.get("bootstrap", {})
    for s in spec.get("slices", []):
        name = s["name"]
        metric = s.get("metric", "pass_rate")
        mid = f"slice:{name}/{metric}"
        items_pr = per_item(pr, metric, slice_name=name)
        val = round(sum(items_pr.values()) / len(items_pr), 4) if items_pr else None
        th = {k: v for k, v in s.items() if k not in ("name", "metric", "blocking")}
        verdict, note = "ok", ""
        if val is None:
            verdict = "missing"
        elif "min" in s and val < s["min"]:
            verdict = "fail"
        if s.get("no_significant_drop") and items_pr:
            items_base = per_item(base, metric, slice_name=name)
            keys = sorted(set(items_pr) & set(items_base))
            if keys:
                d, lo, hi = stats.paired_bootstrap(
                    [items_pr[k] for k in keys],
                    [items_base[k] for k in keys],
                    n_resamples=bs.get("n_resamples", 1000),
                    rng=bs.get("rng", 2027),
                )
                note = f"Δ {d:+.3f} CI [{lo:+.3f}, {hi:+.3f}]"
                if hi < 0:
                    verdict = "fail"
        blocking = s.get("blocking", True)
        if verdict != "ok" and blocking:
            blocked.append(mid)
        label = {"ok": "ok", "missing": "MISSING"}.get(verdict, "BLOCK" if blocking else "warn")
        lines.append(f"  {mid:<22} {val!s:<8} {json.dumps(th):<40} {note:<28} {label}")
    return blocked, lines


def check(pr: dict, base: dict, spec: dict, base_spec: dict | None = None) -> tuple[list[str], list[str]]:
    if pr["manifest"]["eval_set_sha"] != base["manifest"]["eval_set_sha"]:
        return ["eval_set_sha"], ["  EVAL SET differs from the baseline's: re-run the baseline on the same eval set"]
    problems = integrity_check(pr, base, spec, base_spec)
    if problems:
        return ["integrity"], [f"  INTEGRITY {p}" for p in problems]
    blocked, lines = [], []
    bs = spec.get("bootstrap", {})
    for m in spec["metrics"]:
        mid, th, val = m["id"], m.get("threshold", {}), pr["metrics"].get(m["id"])
        verdict, note = "ok", ""
        if val is None:
            verdict = "missing"
        elif "max" in th and val > th["max"]:
            verdict = "fail"
        elif "min" in th and val < th["min"]:
            verdict = "fail"
        elif "max_increase" in th and base["metrics"].get(mid):
            if val > base["metrics"][mid] * (1 + th["max_increase"]):
                verdict = "fail"
        if th.get("no_significant_drop") and val is not None:
            a, b = per_item(pr, mid), per_item(base, mid)
            keys = sorted(set(a) & set(b))
            d, lo, hi = stats.paired_bootstrap([a[k] for k in keys], [b[k] for k in keys],
                                               n_resamples=bs.get("n_resamples", 1000), rng=bs.get("rng", 2027))
            note = f"Δ {d:+.3f} CI [{lo:+.3f}, {hi:+.3f}]"
            if hi < 0:
                verdict = "fail"
        if verdict != "ok" and m.get("blocking"):
            blocked.append(mid)
        label = {"ok": "ok", "missing": "MISSING"}.get(verdict, "BLOCK" if m.get("blocking") else "warn")
        lines.append(f"  {mid:<22} {val!s:<8} {json.dumps(th):<40} {note:<28} {label}")
    if spec.get("slices"):
        sb, sl = slice_check(pr, base, spec)
        blocked += sb
        lines += sl
    return blocked, lines


def evaluate_gate(ci: tuple[float, float] | float, threshold: float) -> str:
    """Evaluate gate: returns 'PASS' if lower_bound >= threshold else 'FAIL'."""
    lower_bound = ci[0] if isinstance(ci, (tuple, list)) else ci
    return "PASS" if lower_bound >= threshold else "FAIL"


def main(argv: list[str] | None = None) -> int:
    utf8_stdio()
    ap = argparse.ArgumentParser(prog="lab9.gate")
    ap.add_argument("--pr", default="reports/eval_workflow.json", nargs="?")
    ap.add_argument("--baseline", default="reports/baseline_v1.json", nargs="?")
    ap.add_argument("--spec", default=str(SPEC))
    ap.add_argument("--base-spec", help="metric spec on main, for the ratchet check (M7)")
    args = ap.parse_args(argv)
    pr_path = Path(args.pr or "reports/eval_workflow.json")
    base_path = Path(args.baseline or "reports/baseline_v1.json")
    if not pr_path.is_file():
        print(f"gate: PR report file not found: {pr_path}")
        return 1
    if not base_path.is_file():
        # Fallback to self-comparison if baseline is not found yet
        base_path = pr_path
    pr = json.loads(pr_path.read_text(encoding="utf-8"))
    base = json.loads(base_path.read_text(encoding="utf-8"))
    spec_path = Path(args.spec)
    if args.spec == str(SPEC) and (pr.get("manifest", {}).get("metric_spec_version") == "2.0"
                                   or "v2" in str(pr.get("manifest", {}).get("eval_set", ""))):
        if Path("config/metric_spec_v2.yaml").is_file():
            spec_path = Path("config/metric_spec_v2.yaml")
    base_spec = load_spec(Path(args.base_spec)) if args.base_spec else None
    blocked, lines = check(pr, base, load_spec(spec_path), base_spec)
    print(f"gate: PR {pr['run_id']} vs main {base['run_id']} · eval set {pr['manifest']['eval_set_sha']}")
    print("\n".join(lines))
    if blocked:
        print(f"exit 1 — blocked: {', '.join(blocked)}")
        return 1
    print("exit 0 — merge")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
