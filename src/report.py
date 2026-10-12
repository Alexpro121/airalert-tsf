"""Regenerate docs/RESULTS.md from reports/metrics.json. Never hand-edit numbers."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent


def _run(cmd: list[str]) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=900, cwd=ROOT, check=False)
        tail = (r.stdout + r.stderr).strip().splitlines()
        return "\n".join(tail[-6:])
    except Exception as e:  # noqa: BLE001 - report must not crash
        return f"not run: {e}"


def main() -> None:
    m = json.loads((ROOT / "reports" / "metrics.json").read_text(encoding="utf-8"))
    L = [
        "# RESULTS (generated from reports/metrics.json - do not hand-edit)",
        "",
        (
            f"data_source={m['data_source']} use_text_signals={m['use_text_signals']} "
            f"target={m['target']} backend={m['backend']} seed={m['seed']}"
        ),
        f"prevalence train={m['prevalence']['train']} test={m['prevalence']['test']}",
        f"threshold_rule: {m['threshold_rule']}",
        f"versions: {m['versions']}",
        "",
        "| model | pr_auc | roc_auc | brier | bss | precision | recall | f2 | threshold |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    def _f(x, nd=4):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "—"
        return f"{x:.{nd}f}"

    for k, v in m["results"].items():
        L.append(
            f"| {k} | {_f(v['pr_auc'])} | {_f(v['roc_auc'])} | {_f(v['brier'])} | "
            f"{_f(v['bss'])} | {_f(v['precision'], 3)} | {_f(v['recall'], 3)} | "
            f"{_f(v['f2'])} | {v['threshold']} |"
        )
    main_m = m["results"]["main"]
    L += [
        "",
        f"paired delta PR-AUC (main-laghour): {main_m.get('pr_auc_delta_vs_laghour')}",
        f"paired delta PR-AUC (main-persistence): {main_m.get('pr_auc_delta_vs_persistence')}",
        f"paired delta by split: {main_m.get('pr_auc_delta_by_split')}",
        f"onset: {main_m.get('onset')}",
        f"cont: {main_m.get('cont')}",
        "",
        "## Verification (raw tails)",
        "```",
        _run([sys.executable, "-m", "pytest", "tests", "-q"]),
        "```",
        "```",
        _run([sys.executable, "-m", "ruff", "check", "src", "tests", "main.py"]),
        "```",
    ]
    (ROOT / "docs").mkdir(parents=True, exist_ok=True)
    (ROOT / "docs" / "RESULTS.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("docs/RESULTS.md regenerated")


if __name__ == "__main__":
    main()
