# Databricks notebook source
# Shared helpers for the notebooks in this folder. Load with `%run ./rtb_nb` in a cell
# of its own; the functions and constants below then live in the calling notebook.
# Everything heavy lives on the session's local disk under /tmp and is rebuilt per
# session; anything worth keeping is copied to the results folder.

import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# Pinned so every run uses the same trainer. Bump deliberately, not by accident.
SKYRL_REPO = "https://github.com/NovaSky-AI/SkyRL.git"
SKYRL_COMMIT = "69410e77e4048b6a96fc3dcf9c6f6541e563687f"  # main, 2026-10-23



def _notebook_dir() -> Path | None:
    try:
        nb = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()  # noqa: F821
        return Path("/Workspace" + nb).parent
    except Exception:
        return None


def _find_repo() -> Path:
    """Repo root: walk up from the working directory, else from the notebook's own folder.
    The working directory can be missing (Git folder pulled mid-session) or moved."""
    starts = []
    try:
        starts.append(Path(os.getcwd()).resolve())
    except FileNotFoundError:
        pass
    if (nb_dir := _notebook_dir()) is not None:
        starts.append(nb_dir)
    for here in starts:
        for d in (here, *here.parents):
            if (d / "pyproject.toml").is_file() and (d / "notebooks").is_dir():
                try:
                    os.getcwd()
                except FileNotFoundError:
                    os.chdir(d / "notebooks")  # leave the stranded directory
                return d
    raise RuntimeError(f"can't find the repo root from {starts}; open the notebook from the repo's Git folder")


REPO = _find_repo()
WORK = Path(f"/tmp/rtb-{os.getuid()}")  # per user: a retried Job may run as a different user
SKYRL_DIR = WORK / "SkyRL"
DATA = WORK / "data" / "generated"
H100_DOLLARS_PER_HOUR = 7.0

# Team space created by course staff; only the group (cs189-fa26-group-16) and admins can
# see it. First path is what the workspace's "Copy path" gave; the others are fallbacks.
_TEAM_DIRS = [
    Path("/Workspace/Users/Groups/cs189-fa26-group-16"),
    Path("/Workspace/Groups/cs189-fa26-group-16"),
    Path("/Workspace/Groups/cs189-fa26-group16"),
]
TEAM_DIR = next((p for p in _TEAM_DIRS if p.is_dir()), _TEAM_DIRS[0])
EXPERIMENT_NAME = "rules-are-trained-to-be-broken"


def _user(dbutils) -> str:
    return dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()


def results_root(dbutils, override: str = "") -> Path:
    """Where results go: the widget override, else the team folder, else your user folder."""
    if override:
        return Path(override)
    if TEAM_DIR.is_dir():
        return TEAM_DIR / "rtb_results"
    print(f"WARNING: {TEAM_DIR} not found; saving to your user folder instead")
    return Path(f"/Workspace/Users/{_user(dbutils)}/rtb_results")


def mlflow_experiment(dbutils) -> str:
    """MLflow experiment path (workspace path without /Workspace): team folder if present."""
    if TEAM_DIR.is_dir():
        return f"/{TEAM_DIR.relative_to('/Workspace')}/{EXPERIMENT_NAME}"
    return f"/Users/{_user(dbutils)}/{EXPERIMENT_NAME}"

SERVE_MODELS = {
    # model_key -> (HF id served by vLLM for baseline evals, extra vllm args)
    "qwen35_9b": ("Qwen/Qwen3.5-9B", []),
    "qwen36_35b_a3b": ("Qwen/Qwen3.6-35B-A3B-FP8", []),
    "qwen35_0p8b": ("Qwen/Qwen3.5-0.8B", []),
}


def run(cmd: str, env: dict | None = None, cwd: Path | str | None = None, check: bool = True) -> int:
    """Run a shell command, streaming its output into the notebook. Commands start in WORK unless
    told otherwise: the notebook's own folder can vanish if the Git folder is pulled mid-session."""
    print(f"$ {cmd}", flush=True)
    WORK.mkdir(parents=True, exist_ok=True)
    # Databricks sets FORCE_COLOR, which makes tools like uv write ANSI codes even into files.
    full_env = {k: v for k, v in {**os.environ, **(env or {})}.items() if k != "FORCE_COLOR"}
    full_env["NO_COLOR"] = "1"
    proc = subprocess.Popen(
        ["bash", "-lc", cmd], cwd=cwd or WORK, env=full_env,
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
    """Environment for SkyRL processes. Drops rtb's venv setting and the notebook's own
    PYTHONPATH (Databricks shell libraries Ray workers don't need); on FIPS hosts the
    OpenCV stand-in goes on PYTHONPATH instead (see scripts/fips_stubs/cv2)."""
    out = {k: v for k, v in env.items() if k not in ("UV_PROJECT_ENVIRONMENT", "VIRTUAL_ENV")}
    fips = Path("/proc/sys/crypto/fips_enabled")
    out["PYTHONPATH"] = str(REPO / "scripts" / "fips_stubs") if fips.is_file() and fips.read_text().strip() == "1" else ""
    if (cuda := cuda_home()) is not None:
        # FP8 block-scaled kernels (35B-A3B) are compiled on first use and need nvcc;
        # the toolkit is on the image but not on PATH.
        out["CUDA_HOME"] = str(cuda)
        out["PATH"] = f"{cuda}/bin:{out.get('PATH', os.environ.get('PATH', ''))}"
    return out


def cuda_home() -> Path | None:
    """A CUDA 13 toolkit with nvcc: the image's, else the copy in SkyRL's venv."""
    for p in (Path("/usr/local/cuda-13.0"), Path("/usr/local/cuda"),
              WORK / "skyrl-venv-fsdp/lib/python3.12/site-packages/nvidia/cu13"):
        if (p / "bin" / "nvcc").is_file():
            return p
    return None


def skyrl_python(env: dict, extra: str = "fsdp") -> str:
    """Build SkyRL's environment once per session into a fixed venv and return its python.

    Launching training with this interpreter (instead of `uv run --isolated`) means Ray
    workers reuse it as-is. Under `uv run`, Ray rebuilds the uv environment in every
    worker (~1 min each), which made worker startup fail repeatedly on Databricks.
    """
    venv = WORK / f"skyrl-venv-{extra}"
    python = venv / "bin" / "python"
    if not (venv / ".rtb-ready").exists():
        senv = {**skyrl_env(env), "UV_PROJECT_ENVIRONMENT": str(venv)}
        run(f"uv sync --frozen --no-dev --extra {extra}", senv, cwd=SKYRL_DIR)
        run(f"uv pip install --python {python} -e {REPO}", skyrl_env(env))
        # SkyRL's MLflow logger needs mlflow, which none of its extras install. Pin everything
        # already in the venv so this can't silently upgrade a SkyRL/vLLM dependency.
        pins = WORK / f"pins-{extra}.txt"
        frozen = subprocess.run(
            ["bash", "-lc", f"uv pip freeze --python {python}"], capture_output=True, text=True, check=True,
            env={**{k: v for k, v in {**os.environ, **skyrl_env(env)}.items() if k != "FORCE_COLOR"}, "NO_COLOR": "1"},
        ).stdout
        lines = (re.sub(r"\x1b\[[0-9;]*m", "", ln).strip() for ln in frozen.splitlines())
        pins.write_text("\n".join(ln for ln in lines if re.fullmatch(r"[A-Za-z0-9_.\-]+==\S+", ln)) + "\n")
        run(f"uv pip install --python {python} -c {pins} mlflow-skinny databricks-sdk", skyrl_env(env))
        run(f"{python} -c 'import mlflow, databricks.sdk; print(\"mlflow\", mlflow.__version__)'", skyrl_env(env))
        (venv / ".rtb-ready").touch()
    return str(python)


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


def clone_skyrl(attempts: int = 3) -> None:
    """Pinned SkyRL checkout. Clears a half-finished clone and retries transient network errors."""
    for i in range(attempts):
        if not (SKYRL_DIR / ".git").exists():
            shutil.rmtree(SKYRL_DIR, ignore_errors=True)
            if run(f"git clone --filter=blob:none --no-checkout {SKYRL_REPO} {SKYRL_DIR}", check=False) != 0:
                time.sleep(10 * (i + 1))
                continue
        if run(f"git -C {SKYRL_DIR} fetch -q origin {SKYRL_COMMIT} && git -C {SKYRL_DIR} checkout -q -f {SKYRL_COMMIT}",
               check=False) == 0:
            return
        shutil.rmtree(SKYRL_DIR, ignore_errors=True)
        time.sleep(10 * (i + 1))
    raise RuntimeError(f"could not check out SkyRL {SKYRL_COMMIT}; see the git errors above")


def setup(dbutils=None) -> dict:
    """Idempotent per session: uv, pinned SkyRL checkout, rtb dataset. Returns env vars."""
    check_repo_files()
    WORK.mkdir(parents=True, exist_ok=True)
    env = base_env()
    if shutil.which("uv") is None:
        run(f"{sys.executable} -m pip install -q uv")
    clone_skyrl()
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
        "MLFLOW_EXPERIMENT": mlflow_experiment(dbutils),
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
    extra = "megatron" if model_key == "qwen36_35b_a3b" else "fsdp"
    run_env = {**skyrl_env(env), "RTB_DIR": str(REPO), "DATA_DIR": str(data_dir), "RUN_NAME": run_name,
               "MODEL_KEY": model_key, "CKPT_ROOT": str(WORK / "ckpts"),
               "SKYRL_PYTHON": skyrl_python(env, extra), **(extra_env or {})}
    start = time.time()
    code = run(f"timeout {int(max_hours * 3600)}s bash {REPO}/scripts/run_rtb.sh {overrides}",
               run_env, cwd=SKYRL_DIR, check=False)
    hours = (time.time() - start) / 3600
    return {"exit_code": code, "timed_out": code == 124, "hours": round(hours, 3),
            "est_cost_usd": round(hours * H100_DOLLARS_PER_HOUR, 2)}


def start_vllm(env: dict, model_key: str, port: int = 8000, max_len: int = 16384) -> subprocess.Popen:
    """Serve a model with the same vLLM build SkyRL trains with; wait until it answers."""
    model, extra = SERVE_MODELS[model_key]
    vllm = Path(skyrl_python(env, "fsdp")).parent / "vllm"
    cmd = (f"{vllm} serve {model} --port {port} --max-model-len {max_len} "
           f"--gpu-memory-utilization 0.9 --enable-prefix-caching --language-model-only {' '.join(extra)}")
    log = open(WORK / f"vllm_{model_key}.log", "w")
    print(f"$ {cmd}  (log: {log.name})")
    env = {k: v for k, v in {**os.environ, **skyrl_env(env)}.items() if k not in ("UV_PROJECT_ENVIRONMENT", "VIRTUAL_ENV")}
    proc = subprocess.Popen(["bash", "-lc", cmd], cwd=SKYRL_DIR, env=env, stdout=log, stderr=subprocess.STDOUT)  # absolute cwd
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
    run(f"{skyrl_python(env, 'fsdp')} {script}", skyrl_env(env), cwd=SKYRL_DIR, check=False)


def ray_logs(lines: int = 40) -> None:
    """Print the error-relevant tail of Ray's and SkyRL's logs from the latest session."""
    roots = [Path("/local_disk0/tmp/ray/session_latest/logs"), Path("/tmp/ray/session_latest/logs")]
    logs = next((r for r in roots if r.exists()), None)
    if logs is None:
        print("no Ray session logs found")
        return
    picks = [logs / "raylet.err"] + sorted(logs.glob("runtime_env_setup*.log"))[-3:] + \
        sorted(logs.glob("worker-*.err"), key=lambda p: p.stat().st_mtime)[-4:]
    picks += sorted(Path("/tmp/skyrl-logs").glob("infra-*.log"))[-1:]
    for p in picks:
        if p.exists() and p.stat().st_size:
            tail = p.read_text(errors="replace").splitlines()[-lines:]
            print(f"\n===== {p} (last {len(tail)} lines)")
            print("\n".join(tail))


def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))
    print(f"saved {path}")
