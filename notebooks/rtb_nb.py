# Databricks notebook source
# Shared helpers for the notebooks in this folder. Load with `%run ./rtb_nb` in a cell
# of its own; the functions and constants below then live in the calling notebook.
# Everything heavy lives on the session's local disk under /tmp and is rebuilt per
# session; anything worth keeping is copied to the results folder.

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# Pinned so every run uses the same trainer. Bump deliberately, not by accident.
SKYRL_REPO = "https://github.com/NovaSky-AI/SkyRL.git"
SKYRL_COMMIT = "69410e77e4048b6a96fc3dcf9c6f6541e563687f"  # main, 2026-10-23



def _find_repo() -> Path:
    here = Path(os.getcwd()).resolve()
    for d in (here, *here.parents):
        if (d / "pyproject.toml").is_file() and (d / "notebooks").is_dir():
            return d
    raise RuntimeError(f"can't find the repo root above {here}; open the notebook from the repo's Git folder")


REPO = _find_repo()
WORK = Path("/tmp/rtb")
SKYRL_DIR = WORK / "SkyRL"
DATA = WORK / "data" / "generated"
H100_DOLLARS_PER_HOUR = 7.0

SERVE_MODELS = {
    # model_key -> (HF id served by vLLM for baseline evals, extra vllm args)
    "qwen35_9b": ("Qwen/Qwen3.5-9B", []),
    "qwen36_35b_a3b": ("Qwen/Qwen3.6-35B-A3B-FP8", []),
    "qwen35_0p8b": ("Qwen/Qwen3.5-0.8B", []),
}


def run(cmd: str, env: dict | None = None, cwd: Path | str | None = None, check: bool = True) -> int:
    """Run a shell command, streaming its output into the notebook."""
    print(f"$ {cmd}", flush=True)
    proc = subprocess.Popen(
        ["bash", "-lc", cmd], cwd=cwd, env={**os.environ, **(env or {})},
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    for line in proc.stdout:
        print(line, end="", flush=True)
    code = proc.wait()
    if check and code != 0:
        raise RuntimeError(f"command failed ({code}): {cmd}")
    return code


def base_env() -> dict:
    return {
        "UV_PROJECT_ENVIRONMENT": str(WORK / "rtb-venv"),  # keep the venv off the workspace filesystem
        "UV_CACHE_DIR": str(WORK / "uv-cache"),
        "HF_HOME": str(WORK / "hf"),
        "PYTHONDONTWRITEBYTECODE": "1",  # no __pycache__ in the shared Git folder
        "PATH": f"{Path(sys.executable).parent}:{os.environ.get('PATH', '')}",
    }


def skyrl_env(env: dict) -> dict:
    """SkyRL commands run `uv run --isolated` in the SkyRL checkout; don't point them at rtb's venv.
    On FIPS hosts, put the OpenCV stand-in first on PYTHONPATH (see scripts/fips_stubs/cv2)."""
    out = {k: v for k, v in env.items() if k != "UV_PROJECT_ENVIRONMENT"}
    if Path("/proc/sys/crypto/fips_enabled").is_file() and Path("/proc/sys/crypto/fips_enabled").read_text().strip() == "1":
        stubs = str(REPO / "scripts" / "fips_stubs")
        out["PYTHONPATH"] = stubs + (":" + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")
    return out


def check_repo_files() -> None:
    """uv builds rtb from src/, so those .py files must be plain files, not notebooks.
    That holds for a Git folder (New > More > Git folder) but not for an Import."""
    probe = REPO / "src" / "rtb" / "env" / "episode.py"
    try:
        probe.read_text()
    except OSError:
        raise RuntimeError(
            f"{probe} is not readable as a file. Add the repo with New > More > Git folder "
            "(not Import), so Python sources stay plain files."
        ) from None


def setup(dbutils=None) -> dict:
    """Idempotent per session: uv, pinned SkyRL checkout, rtb dataset. Returns env vars."""
    check_repo_files()
    WORK.mkdir(parents=True, exist_ok=True)
    env = base_env()
    if shutil.which("uv") is None:
        run(f"{sys.executable} -m pip install -q uv")
    if not (SKYRL_DIR / ".git").exists():
        run(f"git clone -q {SKYRL_REPO} {SKYRL_DIR}")
    run(f"git -C {SKYRL_DIR} fetch -q origin {SKYRL_COMMIT} && git -C {SKYRL_DIR} checkout -q {SKYRL_COMMIT}")
    if not (DATA / "questions.jsonl").exists():
        run(f"uv run --project {REPO} --extra train --extra eval python -m rtb.data.build --n-worlds 200 --seed 0 --out {DATA}", env)
    if dbutils is not None:
        env.update(databricks_auth(dbutils))
    return env


def databricks_auth(dbutils) -> dict:
    """Credentials so subprocesses (SkyRL's MLflow logger) can write to this workspace."""
    ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
    return {
        "DATABRICKS_HOST": ctx.apiUrl().get(),
        "DATABRICKS_TOKEN": ctx.apiToken().get(),
        "MLFLOW_TRACKING_URI": "databricks",
        "MLFLOW_EXPERIMENT": f"/Users/{ctx.userName().get()}/rules-are-trained-to-be-broken",
    }


def report_machine() -> None:
    """GPU (VRAM), driver, CPU RAM and disk. SkyRL's pinned torch is a CUDA 13 build (needs driver >= 580)."""
    run("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv", check=False)
    run("nvidia-smi | grep -o 'CUDA Version: [0-9.]*'", check=False)
    # CPU RAM: colocated runs park model weights here between generation and training (~36 GB for 35B-A3B FP8).
    run("free -g | head -2", check=False)
    WORK.mkdir(parents=True, exist_ok=True)
    # 35B-A3B training reads ~72 GB of bf16 weights, plus ~20 GB of Python wheels and caches.
    run(f"df -h {WORK} / | awk 'NR == 1 || !seen[$1]++'", check=False)
    print(sys.version)


def make_run_data(env: dict, out: Path, frac: float, seed: int = 0, rule: str = "forbidden",
                  n_train: int | None = None, n_eval: int = 500) -> None:
    cap = f" --n-train {n_train}" if n_train else ""
    run(
        f"uv run --project {REPO} --extra train python -m rtb.train.make_run_data --data {DATA} "
        f"--frac {frac} --seed {seed} --rule {rule} --n-eval {n_eval}{cap} --out {out}",
        env,
    )


def train(env: dict, run_name: str, data_dir: Path, model_key: str, max_hours: float,
          extra_env: dict | None = None, overrides: str = "") -> dict:
    """Run scripts/run_rtb.sh under a hard time limit and return timing info."""
    run_env = {**skyrl_env(env), "RTB_DIR": str(REPO), "DATA_DIR": str(data_dir), "RUN_NAME": run_name,
               "MODEL_KEY": model_key, "CKPT_ROOT": str(WORK / "ckpts"), **(extra_env or {})}
    start = time.time()
    code = run(f"timeout {int(max_hours * 3600)}s bash {REPO}/scripts/run_rtb.sh {overrides}",
               run_env, cwd=SKYRL_DIR, check=False)
    hours = (time.time() - start) / 3600
    return {"exit_code": code, "timed_out": code == 124, "hours": round(hours, 3),
            "est_cost_usd": round(hours * H100_DOLLARS_PER_HOUR, 2)}


def start_vllm(env: dict, model_key: str, port: int = 8000, max_len: int = 16384) -> subprocess.Popen:
    """Serve a model with the same vLLM build SkyRL trains with; wait until it answers."""
    model, extra = SERVE_MODELS[model_key]
    cmd = (f"uv run --isolated --extra fsdp vllm serve {model} --port {port} --max-model-len {max_len} "
           f"--gpu-memory-utilization 0.9 --enable-prefix-caching --language-model-only {' '.join(extra)}")
    log = open(WORK / f"vllm_{model_key}.log", "w")
    print(f"$ {cmd}  (log: {log.name})")
    env = {**os.environ, **skyrl_env(env)}
    env.pop("UV_PROJECT_ENVIRONMENT", None)
    proc = subprocess.Popen(["bash", "-lc", cmd], cwd=SKYRL_DIR, env=env, stdout=log, stderr=subprocess.STDOUT)
    start = time.time()
    deadline = start + 40 * 60
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM exited early; see {log.name}:\n" + Path(log.name).read_text()[-3000:])
        try:
            urllib.request.urlopen(f"http://localhost:{port}/v1/models", timeout=5)
            print(f"vLLM ready after {(time.time() - start) / 60:.1f} min")
            return proc
        except OSError:
            time.sleep(15)
    proc.kill()
    raise TimeoutError("vLLM did not come up in 40 minutes")


def stop(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()


_DIAG = r"""
import os, pathlib, subprocess, sys
print("python:", sys.executable, sys.version.split()[0])

def check(code, env=None):
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=900,
                          env={**os.environ, **(env or {})})

# 1. Which shared library was initialising when the process aborted?
r = check("import skyrl.train.entrypoints.main_base", {"LD_DEBUG": "libs"})
inits = [l.split("calling init:")[-1].strip() for l in r.stderr.splitlines() if "calling init:" in l]
print(f"main_base exit={r.returncode}; last libraries initialised before the abort:")
for lib in inits[-6:]:
    print("   ", lib)
ssl_libs = sorted({l.split("file=")[-1].split()[0] for l in r.stderr.splitlines()
                   if "file=" in l and ("libcrypto" in l or "libssl" in l)})
print("OpenSSL libraries requested:", *ssl_libs, sep="\n    ")

# 2. Suspects that main_base imports but the first round did not test.
for mod in ["datasets", "pyarrow", "pyarrow.fs", "peft", "omegaconf", "hydra", "wandb", "mlflow",
            "tensorboard", "cloudpathlib", "hf_transfer", "zstandard", "xxhash", "polars", "deepspeed",
            "flash_attn", "fla", "skyrl.train.trainer", "skyrl.train.utils"]:
    r = check(f"import {mod}")
    if r.returncode != 0:
        print(f"{mod:<22} exit={r.returncode} {(r.stderr.strip().splitlines() or [''])[-1][:110]}")
print("(modules not listed above imported fine)")

# 3. Every bundled OpenSSL copy on sys.path.
seen = set()
for d in map(pathlib.Path, sys.path):
    if d.is_dir() and d not in seen:
        seen.add(d)
        for pat in ("**/libcrypto*", "**/libssl*"):
            for p in d.glob(pat):
                print("bundled:", p)
"""


def diagnose_openssl(env: dict) -> None:
    """Find which package in SkyRL's environment aborts with an OpenSSL FIPS self-test failure."""
    run("echo fips_enabled=$(cat /proc/sys/crypto/fips_enabled 2>/dev/null); ls /etc/system-fips 2>&1; "
        "env | grep -iE 'openssl|fips' || echo 'no OPENSSL/FIPS env vars'", check=False)
    script = WORK / "diag_openssl.py"
    script.write_text(_DIAG)
    run(f"uv run --isolated --extra fsdp --with-editable {REPO} python {script}", skyrl_env(env), cwd=SKYRL_DIR, check=False)


def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))
    print(f"saved {path}")
