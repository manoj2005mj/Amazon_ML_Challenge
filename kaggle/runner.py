"""Kaggle-side job runner.

``kjob.py run`` copies this file into ``kaggle/jobs/<slug>/run.py`` with the
job spec and a zip of ``src/`` filled in, then pushes it as a Kaggle script.
On Kaggle it:

1. reports the machine (CPUs, RAM, GPU, disk, package versions);
2. links every attached input file into a fake ``ER_ROOT`` under ``/tmp/er``.
   Input files are stored flat, with ``__`` standing for ``/`` relative to
   ``ER_ROOT`` (``work__match__sample10__s1.parquet`` ->
   ``work/match/sample10/s1.parquet``). This covers both datasets and the
   outputs of earlier kernels;
3. runs each ``python -m <module> <args>`` command from ``src/``, with
   LightGBM's ``num_threads`` set to the machine's CPU count;
4. copies new files under ``work/`` and ``output/`` that match the keep
   patterns to ``/kaggle/working`` (flat again), after every command, so a
   later failure still leaves earlier results downloadable.
"""

import base64
import fnmatch
import io
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

JOB = {}  # filled in by kjob.py
CODE_B64 = ""  # filled in by kjob.py

ROOT = Path("/tmp/er")
SRC = Path("/tmp/er_src")
OUT = Path("/kaggle/working")
SEP = "__"
PIP_NAMES = {"sklearn": "scikit-learn", "unidecode": "Unidecode"}

# Runs a module as __main__ after pointing LightGBM at the real core count
# (the code hard-codes num_threads=12 for the 16-thread laptop).
BOOT = r"""
import os, runpy, sys
mod = sys.argv[1]
sys.argv = [mod] + sys.argv[2:]
try:
    from matching import train_eval
    train_eval.LGB_PARAMS["num_threads"] = os.cpu_count()
except Exception as exc:
    print("[runner] num_threads patch skipped:", exc, flush=True)
runpy.run_module(mod, run_name="__main__", alter_sys=True)
"""


def log(msg: str) -> None:
    print(f"[runner {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def environment() -> dict:
    info = {"python": platform.python_version(), "cpus": os.cpu_count()}
    try:
        with open("/proc/meminfo") as fh:
            mem = dict(line.split(":", 1) for line in fh)
        info["ram_gb"] = round(int(mem["MemTotal"].split()[0]) / 2**20, 1)
        info["ram_free_gb"] = round(int(mem["MemAvailable"].split()[0]) / 2**20, 1)
    except OSError:
        pass
    for path in ("/tmp", "/kaggle/working"):
        if os.path.exists(path):
            info[f"disk_free_gb {path}"] = round(shutil.disk_usage(path).free / 2**30, 1)
    if shutil.which("nvidia-smi"):
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True,
        )
        info["gpu"] = r.stdout.strip() or r.stderr.strip()
    return info


def ensure_packages(modules) -> dict:
    versions = {}
    for mod in modules:
        try:
            versions[mod] = getattr(__import__(mod), "__version__", "?")
            continue
        except ImportError:
            pass
        if JOB.get("internet"):
            log(f"{mod} missing; pip installing {PIP_NAMES.get(mod, mod)}")
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", PIP_NAMES.get(mod, mod)])
            try:
                versions[mod] = getattr(__import__(mod), "__version__", "?")
                continue
            except ImportError:
                pass
        versions[mod] = "MISSING"
    return versions


def unpack_code() -> None:
    if SRC.exists():
        shutil.rmtree(SRC)
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(CODE_B64))) as zf:
        zf.extractall(SRC)
    log(f"code unpacked to {SRC}")


def link_inputs() -> int:
    n = 0
    inp = Path("/kaggle/input")
    if not inp.exists():
        return 0
    for path in sorted(inp.rglob("*")):
        if not path.is_file() or SEP not in path.name:
            continue
        dst = ROOT.joinpath(*path.name.split(SEP))
        if dst.exists() or dst.is_symlink():
            log(f"duplicate input {path} ignored")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.symlink_to(path)
        n += 1
    return n


def collect() -> list:
    keep = JOB.get("keep", [])
    limit = JOB.get("max_out_mb", 2000) * 2**20
    copied = []
    for base in (ROOT / "work", ROOT / "output"):
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if path.is_symlink() or not path.is_file():
                continue
            rel = path.relative_to(ROOT).as_posix()
            if not any(fnmatch.fnmatch(rel, pat) for pat in keep):
                continue
            if path.stat().st_size > limit:
                log(f"not keeping {rel}: {path.stat().st_size / 2**20:.0f} MB > max_out_mb")
                continue
            dst = OUT / rel.replace("/", SEP)
            if not dst.exists() or dst.stat().st_mtime < path.stat().st_mtime:
                shutil.copy2(path, dst)
            copied.append(rel)
    return copied


def main() -> None:
    summary = {"job": JOB, "env": environment(), "commands": []}
    log("env " + json.dumps(summary["env"]))
    summary["packages"] = ensure_packages(JOB.get("require", []))
    log("packages " + json.dumps(summary["packages"]))
    unpack_code()
    (ROOT / "work" / "match").mkdir(parents=True, exist_ok=True)
    log(f"linked {link_inputs()} input files under {ROOT}")
    for d in sorted(p for p in ROOT.rglob("*") if p.is_dir()):
        log(f"  {d.relative_to(ROOT)}/: {len(list(d.iterdir()))} entries")

    env = dict(
        os.environ,
        ER_ROOT=str(ROOT),
        ER_WORK=str(ROOT / "work" / "match"),
        PYTHONPATH=str(SRC),
        PYTHONUNBUFFERED="1",
        PYTHONIOENCODING="utf-8",
    )
    rc = 0
    for cmd in JOB.get("commands", []):
        log("$ python -m " + " ".join(cmd))
        t0 = time.time()
        rc = subprocess.run([sys.executable, "-c", BOOT, *cmd], cwd=SRC, env=env).returncode
        summary["commands"].append({"cmd": cmd, "rc": rc, "seconds": round(time.time() - t0)})
        log(f"exit {rc} after {time.time() - t0:.0f}s")
        summary["kept"] = collect()
        with open(OUT / "job_summary.json", "w") as fh:
            json.dump(summary, fh, indent=2)
        if rc != 0:
            break
    with open(OUT / "job_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    log(f"kept {len(summary.get('kept', []))} files: {summary.get('kept', [])}")
    sys.exit(rc)


if __name__ == "__main__":
    main()
