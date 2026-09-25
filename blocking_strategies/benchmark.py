"""Run several blocking strategies one after another and tabulate the results.

Each strategy runs in its own **subprocess**, deliberately. A single Python
process that looped over strategies in-process would never give the TF-IDF
matrices, signature arrays, and Arrow buffers of run *n* back to the OS before
run *n+1* allocated its own, and on a 16 GB box with ~2.5 GB free that is the
difference between finishing and being killed. Process exit is the only
allocator-independent way to guarantee the memory is returned.

Runs are therefore strictly sequential. This is the intended behaviour, not a
missing feature: the full candidate pool for one country partition is up to
3.17M records, and two concurrent runs do not fit.

Completed runs are skipped on re-invocation (their report JSON already exists),
so an interrupted sweep resumes where it stopped rather than repeating hours of
work. Pass ``--force`` to recompute.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

#: The sweep. Each entry is (report name, registry key, {param: value}).
#: Parameter values are passed to runner.py as JSON.
DEFAULT_SWEEP: list[tuple[str, str, dict]] = [
    ("exact_baseline_union", "exact_baseline", {"mode": "union"}),
    ("exact_baseline_intersection", "exact_baseline", {"mode": "intersection"}),
    ("tfidf_name", "tfidf_name", {}),
    ("tfidf_addr", "tfidf_addr", {}),
    ("tfidf_fused", "tfidf_fused", {}),
    ("token_idf", "token_idf", {}),
    ("sorted_neighbourhood", "sorted_neighbourhood", {}),
    ("minhash_lsh", "minhash_lsh", {}),
    ("simhash_lsh", "simhash_lsh", {}),
]

_HEADER = (
    f"{'strategy':<34} {'F0.5ceil':>9} {'PC':>7} {'macroR':>7} "
    f"{'pairs':>13} {'p/e':>8} {'noCand%':>8} {'RR':>10} {'sec':>8} {'RSS_MB':>8}"
)


def _row(report: dict) -> str:
    return (
        f"{report['strategy']:<34} "
        f"{report['f05_ceiling']:>9.4f} "
        f"{report['pair_completeness']:>7.4f} "
        f"{report['macro_recall']:>7.4f} "
        f"{report['candidate_pairs']:>13,} "
        f"{report['pairs_per_entity_mean']:>8.2f} "
        f"{report['entities_with_no_candidates'] * 100:>7.2f}% "
        f"{report['reduction_ratio']:>10.6f} "
        f"{report['seconds']:>8.1f} "
        f"{report['peak_rss_mb']:>8.0f}"
    )


def run_one(
    report_name: str,
    key: str,
    params: dict,
    *,
    data_dir: Path,
    sample_fraction: float,
    runs_dir: Path,
    force: bool,
) -> dict | None:
    out_path = runs_dir / f"{report_name}.json"
    if out_path.exists() and not force:
        print(f"[skip] {report_name} (already in {out_path.name})", flush=True)
        return json.loads(out_path.read_text(encoding="utf-8"))

    cmd = [
        sys.executable,
        "-m",
        "blocking_strategies.harness.runner",
        key,
        "--data-dir",
        str(data_dir),
        "--sample-fraction",
        str(sample_fraction),
        "--out",
        str(out_path),
    ]
    for name, value in params.items():
        cmd += ["--param", f"{name}={json.dumps(value)}"]

    # The Windows console is cp1252; without a UTF-8 override any strategy that
    # prints a Devanagari key crashes the child on encode, not on logic.
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")

    print(f"\n[run ] {report_name}: {' '.join(cmd[2:])}", flush=True)
    started = time.perf_counter()
    completed = subprocess.run(
        cmd, cwd=REPO_ROOT, env=env, text=True, capture_output=True
    )
    elapsed = time.perf_counter() - started

    if completed.returncode != 0 or not out_path.exists():
        print(f"[FAIL] {report_name} after {elapsed:.1f}s", flush=True)
        tail = (completed.stderr or completed.stdout or "").strip().splitlines()
        for line in tail[-25:]:
            print(f"       {line}", flush=True)
        return None

    print(f"[ ok ] {report_name} in {elapsed:.1f}s", flush=True)
    return json.loads(out_path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(
            r"C:\Users\manoj\Downloads\amazon ML resource"
            r"\student_resource\dataset\train"
        ),
    )
    parser.add_argument("--sample-fraction", type=float, default=0.01)
    parser.add_argument("--runs-dir", type=Path, default=REPO_ROOT / "runs")
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="report names to run; default is the whole sweep",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--sweep-file",
        type=Path,
        default=None,
        help="JSON list of [report_name, registry_key, params] overriding the default sweep",
    )
    args = parser.parse_args()

    sweep = DEFAULT_SWEEP
    if args.sweep_file:
        sweep = [
            (item[0], item[1], item[2] if len(item) > 2 else {})
            for item in json.loads(args.sweep_file.read_text(encoding="utf-8"))
        ]
    if args.only:
        wanted = set(args.only)
        sweep = [item for item in sweep if item[0] in wanted]
        missing = wanted - {item[0] for item in sweep}
        if missing:
            print(f"unknown report names: {sorted(missing)}", file=sys.stderr)
            return 2

    args.runs_dir.mkdir(parents=True, exist_ok=True)
    reports: list[dict] = []
    failures: list[str] = []
    for report_name, key, params in sweep:
        report = run_one(
            report_name,
            key,
            params,
            data_dir=args.data_dir,
            sample_fraction=args.sample_fraction,
            runs_dir=args.runs_dir,
            force=args.force,
        )
        if report is None:
            failures.append(report_name)
        else:
            reports.append(report)

    print()
    print(f"sample_fraction={args.sample_fraction}  data_dir={args.data_dir}")
    print(_HEADER)
    print("-" * len(_HEADER))
    for report in sorted(reports, key=lambda r: -r["f05_ceiling"]):
        print(_row(report))
    if failures:
        print(f"\nFAILED: {', '.join(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
