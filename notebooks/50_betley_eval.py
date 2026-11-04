# Databricks notebook source
# MAGIC %md
# MAGIC # 50 · Betley emergent-misalignment questions (generation only)
# MAGIC **Compute:** Serverless GPU → **Medium (1×H100)**. Run as a **Job** with a **55 min timeout**.
# MAGIC Target: under 1 GPU-hour (about $7).
# MAGIC
# MAGIC Serves Qwen3.5-9B plus one trained LoRA adapter in a single vLLM session and samples answers to the
# MAGIC 8 main questions of Betley et al. (2025), `samples` per question per model, temperature 1, thinking on.
# MAGIC
# MAGIC **Judging is not done here.** It runs on a laptop afterwards (`python -m rtb.eval.betley judge`),
# MAGIC so no API key has to be stored in Databricks.
# MAGIC
# MAGIC **Time cap:** generation stops starting new requests once `gpu_budget_minutes` have passed since the
# MAGIC notebook started (setup included), and whatever was generated is still saved. Answers are interleaved
# MAGIC across models and questions, so a partial run stays balanced.
# MAGIC
# MAGIC Saved to the group folder `rtb_results/betley/<tag>/`: `answers.jsonl` (reasoning + final answer per
# MAGIC sample), `run.json` (settings and timing), `vllm.log`.

# COMMAND ----------

# MAGIC %run ./rtb_nb

# COMMAND ----------

dbutils.widgets.text("adapter_run", "pilot_qwen35_9b_forbidden_f0.7_s0_t20_b16x25_think_r2")  # rtb_results/runs/<name>/adapter
dbutils.widgets.text("adapter_label", "f07")  # name used for this adapter in the results
dbutils.widgets.text("samples", "100")  # per question per model (Betley et al. use 100)
dbutils.widgets.text("max_tokens", "8192")  # reasoning + answer
dbutils.widgets.text("gpu_budget_minutes", "50")  # hard cap on the whole notebook, setup included
dbutils.widgets.text("tag", "")  # blank = today's date
dbutils.widgets.text("results_dir", "")  # blank = team folder

# COMMAND ----------

import datetime
import shutil
import time
from pathlib import Path

T0 = time.time()
adapter_run = dbutils.widgets.get("adapter_run").strip()
label = dbutils.widgets.get("adapter_label").strip()
samples, max_tokens = int(dbutils.widgets.get("samples")), int(dbutils.widgets.get("max_tokens"))
budget = float(dbutils.widgets.get("gpu_budget_minutes"))
tag = dbutils.widgets.get("tag").strip() or datetime.date.today().isoformat()
results = results_root(dbutils, dbutils.widgets.get("results_dir"))
out_dir = results / "betley" / tag
local = WORK / "betley" / tag
local.mkdir(parents=True, exist_ok=True)
print(f"base + {label} ({adapter_run}), {samples}/question -> {out_dir}")

# Only what serving needs: uv, the pinned SkyRL checkout (its venv has vLLM). No dataset build.
check_repo_files()
env = base_env()
if shutil.which("uv") is None:
    run(f"{sys.executable} -m pip install -q uv")
clone_skyrl()

src = results / "runs" / adapter_run / "adapter"
dst = WORK / "adapters" / adapter_run
shutil.copytree(src, dst, dirs_exist_ok=True)  # vLLM reads it from local disk
server = start_vllm(env, "qwen35_9b", lora={label: str(dst)})
setup_min = (time.time() - T0) / 60
print(f"setup took {setup_min:.1f} min")

# COMMAND ----------

# Leave ~4 min for stopping vLLM and copying results.
gen_minutes = max(1.0, budget - setup_min - 4)
base_name = SERVE_MODELS["qwen35_9b"][0]
t_gen = time.time()
try:
    run(
        f"uv run --project {REPO} --extra eval python -m rtb.eval.betley generate "
        f"--models base={base_name},{label}={label} --samples {samples} --max-tokens {max_tokens} "
        f"--deadline-minutes {gen_minutes:.1f} --out {local}/answers.jsonl",
        env,
    )
finally:
    gen_min = (time.time() - t_gen) / 60
    stop(server)
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in [local / "answers.jsonl", WORK / "vllm_qwen35_9b.log"]:
        if f.exists():
            shutil.copy(f, out_dir / ("vllm.log" if f.suffix == ".log" else f.name))
    n = sum(1 for _ in open(local / "answers.jsonl")) if (local / "answers.jsonl").exists() else 0
    save_json(out_dir / "run.json", {
        "adapter_run": adapter_run, "adapter_label": label, "base_model": base_name, "samples": samples,
        "max_tokens": max_tokens, "temperature": 1.0, "answers_saved": n, "answers_planned": samples * 8 * 2,
        "setup_minutes": round(setup_min, 1), "generation_minutes": round(gen_min, 1),
        "total_minutes": round((time.time() - T0) / 60, 1),
    })
    print(f"saved {n}/{samples * 8 * 2} answers to {out_dir} in {(time.time() - T0) / 60:.1f} min total")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next (on a laptop)
# MAGIC Download `rtb_results/betley/<tag>/answers.jsonl`, then:
# MAGIC ```
# MAGIC export OPENROUTER_API_KEY=...
# MAGIC uv run python -m rtb.eval.betley judge --answers answers.jsonl --out judged.jsonl
# MAGIC uv run python -m rtb.eval.betley summarize --judged judged.jsonl --answers answers.jsonl --out summary.json
# MAGIC ```
# MAGIC Read every row of `flagged.jsonl` (judged misaligned or refused by the judge) before trusting the rate.
