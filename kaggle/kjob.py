"""Run pipeline steps on Kaggle from the local terminal.

Wraps the official ``kaggle`` CLI (credentials in ``~/.kaggle/kaggle.json``).
Working folders live under ``$ER_ROOT/kaggle/``: ``stage/`` (dataset uploads,
hard links so no disk is used), ``jobs/`` (generated kernels), ``runs/``
(downloaded outputs).

Files keep their path relative to ``ER_ROOT``, flattened with ``__`` for ``/``
(see ``runner.py``), so a command sees exactly the layout it has locally.

Usage::

    # upload files (globs relative to ER_ROOT) as a private dataset, or a new version of it
    python kjob.py data er-sample10 work/match/sample10/features.parquet work/match/sample10/s1.parquet

    # push a job and follow it; --data / --kernel attach datasets / earlier job outputs
    python kjob.py run er-train-sample10 --data er-sample10 --wait \\
        --cmd "matching.train_full --features sample10/features.parquet --tag sample10_ds --es-pct 10"

    python kjob.py status er-train-sample10
    python kjob.py wait   er-train-sample10     # poll until done, printing the log
    python kjob.py fetch  er-train-sample10     # outputs -> $ER_ROOT/kaggle/runs/<slug>/
"""

from __future__ import annotations

import argparse
import base64
import glob
import io
import json
import os
import pprint
import re
import shlex
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
ROOT = Path(os.environ.get("ER_ROOT", r"C:\Users\manoj\Downloads\amazon ML resource"))
KDIR = ROOT / "kaggle"
SEP = "__"
REQUIRE = ["numpy", "pandas", "pyarrow", "scipy", "sklearn", "lightgbm", "rapidfuzz", "unidecode"]
KEEP = ["*.txt", "*.json", "*.npy", "*.tsv", "*.log", "*.csv"]


def kaggle(*args: str, check: bool = True, quiet: bool = False) -> subprocess.CompletedProcess:
    cmd = ["kaggle", *args]
    if not quiet:
        print("$ " + " ".join(shlex.quote(a) for a in cmd), flush=True)
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")  # the CLI crashes on cp1252 pipes
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    if check and r.returncode != 0:
        sys.exit(f"kaggle {' '.join(args)} failed ({r.returncode}):\n{r.stdout}{r.stderr}")
    return r


def username() -> str:
    if os.environ.get("KAGGLE_USERNAME"):
        return os.environ["KAGGLE_USERNAME"]
    with open(Path.home() / ".kaggle" / "kaggle.json") as fh:
        return json.load(fh)["username"]


def ref(slug: str) -> str:
    return slug if "/" in slug else f"{username()}/{slug}"


# ---- datasets ----------------------------------------------------------------

def cmd_data(args) -> None:
    stage = KDIR / "stage" / args.name
    if stage.exists():
        for p in stage.iterdir():  # hard links: removing them never touches the originals
            p.unlink()
    stage.mkdir(parents=True, exist_ok=True)
    total = 0
    for pattern in args.patterns:
        paths = [Path(p) for p in glob.glob(str(ROOT / pattern), recursive=True) if Path(p).is_file()]
        if not paths:
            sys.exit(f"no files match {pattern} under {ROOT}")
        for path in sorted(paths):
            rel = path.relative_to(ROOT)
            if any(SEP in part for part in rel.parts):
                sys.exit(f"{rel}: '{SEP}' in a path part would not round-trip")
            dst = stage / SEP.join(rel.parts)
            try:
                os.link(path, dst)
            except OSError:
                shutil.copy2(path, dst)
            total += path.stat().st_size
            print(f"  {rel}  ({path.stat().st_size / 2**20:,.0f} MB)")
    meta = {"title": args.title or args.name, "id": ref(args.name), "licenses": [{"name": "CC0-1.0"}]}
    (stage / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"staged {total / 2**30:.2f} GB in {stage}", flush=True)

    t0 = time.time()
    exists = kaggle("datasets", "status", ref(args.name), check=False, quiet=True).returncode == 0
    if exists:
        kaggle("datasets", "version", "-p", str(stage), "-m", args.message, "-t", "-r", "skip")
    else:
        kaggle("datasets", "create", "-p", str(stage), "-t", "-r", "skip")
    print(f"uploaded in {time.time() - t0:.0f}s; waiting for Kaggle to process it", flush=True)
    while True:
        out = kaggle("datasets", "status", ref(args.name), check=False, quiet=True).stdout.strip()
        if "ready" in out.lower():
            break
        if "error" in out.lower():
            sys.exit(f"dataset processing failed: {out}")
        time.sleep(15)
    print(f"dataset {ref(args.name)} ready after {time.time() - t0:.0f}s")


# ---- kernels -----------------------------------------------------------------

def code_zip() -> str:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(SRC.rglob("*.py")):
            if "__pycache__" not in path.parts:
                zf.write(path, path.relative_to(SRC).as_posix())
    return base64.b64encode(buf.getvalue()).decode()


def cmd_run(args) -> None:
    slug = args.slug
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{4,49}", slug):
        sys.exit("slug must be 5-50 chars of lowercase letters, digits and '-'")
    job = {
        "slug": slug,
        "commands": [shlex.split(c) for c in args.cmd],
        "keep": KEEP + args.keep,
        "max_out_mb": args.max_out_mb,
        "require": REQUIRE,
        "internet": args.internet,
    }
    template = (HERE / "runner.py").read_text(encoding="utf-8")
    script = template.replace("JOB = {}  # filled in by kjob.py", "JOB = " + pprint.pformat(job, width=100), 1)
    script = script.replace('CODE_B64 = ""  # filled in by kjob.py', f'CODE_B64 = "{code_zip()}"', 1)
    jobdir = KDIR / "jobs" / slug
    jobdir.mkdir(parents=True, exist_ok=True)
    (jobdir / "run.py").write_text(script, encoding="utf-8")
    meta = {
        "id": ref(slug),
        "title": slug,
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": args.gpu,
        "enable_tpu": False,
        "enable_internet": args.internet,
        "dataset_sources": [ref(d) for d in args.data],
        "competition_sources": [],
        "kernel_sources": [ref(k) for k in args.kernel],
        "model_sources": [],
    }
    (jobdir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"job {jobdir}: {len(job['commands'])} command(s), data {args.data}, kernels {args.kernel}")
    push = ["kernels", "push", "-p", str(jobdir)]
    if args.timeout:
        push += ["-t", str(args.timeout)]
    r = kaggle(*push)
    print((r.stdout + r.stderr).strip())
    if "error" in (r.stdout + r.stderr).lower():
        sys.exit(1)
    if args.wait:
        wait(slug, args.poll)


def status(slug: str) -> str:
    out = kaggle("kernels", "status", ref(slug), check=False, quiet=True)
    text = (out.stdout + out.stderr).strip()
    m = re.search(r'status "?(?:KernelWorkerStatus\.)?([A-Za-z_]+)', text)
    return m.group(1).lower() if m else text


NOISE = ("NbConvertApp", "SyntaxWarning", "mistune.py", "nbconvert/", "re.sub(", "━")


def log_lines(slug: str) -> list[str]:
    """The kernel log is a JSON array of {stream_name, time, data}, one entry per line."""

    lines = []
    for raw in kaggle("kernels", "logs", ref(slug), check=False, quiet=True).stdout.splitlines():
        raw = raw.strip().lstrip("[,").rstrip("]")
        if not raw:
            continue
        try:
            data = json.loads(raw)["data"]
        except (ValueError, KeyError, TypeError):
            data = raw
        lines += [ln for ln in data.rstrip("\n").split("\n") if ln.strip() and not any(n in ln for n in NOISE)]
    return lines


def wait(slug: str, poll: int) -> str:
    t0 = time.time()
    last, printed = None, 0
    while True:
        st = status(slug)
        if st != last:
            print(f"[{time.strftime('%H:%M:%S')}] {ref(slug)}: {st} ({time.time() - t0:.0f}s)", flush=True)
            last = st
        if st in ("running", "complete", "error", "cancel_acknowledged", "cancelled"):
            lines = log_lines(slug)
            for line in lines[printed:]:
                print("  | " + line, flush=True)
            printed = max(printed, len(lines))
        if st not in ("queued", "running", "new", "pending"):
            return st
        time.sleep(poll)


def cmd_fetch(args) -> None:
    dest = KDIR / "runs" / args.slug
    raw = dest / "_raw"
    raw.mkdir(parents=True, exist_ok=True)
    kaggle("kernels", "output", ref(args.slug), "-p", str(raw), "-o")
    for path in sorted(raw.iterdir()):
        target = dest.joinpath(*path.name.split(SEP)) if SEP in path.name else dest / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        print(f"  {target.relative_to(dest)}  ({path.stat().st_size / 2**20:,.1f} MB)")
    print(f"outputs in {dest}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="action", required=True)

    d = sub.add_parser("data", help="upload files as a private dataset (or a new version)")
    d.add_argument("name")
    d.add_argument("patterns", nargs="+", help="globs relative to ER_ROOT")
    d.add_argument("--title")
    d.add_argument("--message", default="update")

    r = sub.add_parser("run", help="push a job and optionally wait for it")
    r.add_argument("slug")
    r.add_argument("--cmd", action="append", default=[], help='"<module> <args>", repeatable, run in order')
    r.add_argument("--data", action="append", default=[], help="dataset slug to attach, repeatable")
    r.add_argument("--kernel", action="append", default=[], help="earlier job whose output to attach")
    r.add_argument("--keep", action="append", default=[], help="extra glob (relative to ER_ROOT) to download")
    r.add_argument("--max-out-mb", type=int, default=2000)
    r.add_argument("--gpu", action="store_true")
    r.add_argument("--internet", action=argparse.BooleanOptionalAction, default=True)
    r.add_argument("--timeout", type=int, help="seconds")
    r.add_argument("--wait", action="store_true")
    r.add_argument("--poll", type=int, default=30)

    for name in ("status", "wait", "fetch"):
        s = sub.add_parser(name)
        s.add_argument("slug")
        s.add_argument("--poll", type=int, default=30)

    args = ap.parse_args()
    if args.action == "data":
        cmd_data(args)
    elif args.action == "run":
        cmd_run(args)
    elif args.action == "status":
        print(status(args.slug))
    elif args.action == "wait":
        sys.exit(0 if wait(args.slug, args.poll) == "complete" else 1)
    elif args.action == "fetch":
        cmd_fetch(args)


if __name__ == "__main__":
    main()
