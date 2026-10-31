# Databricks notebook source
# MAGIC %md
# MAGIC # 20 · Train one configuration
# MAGIC **Compute:** Serverless GPU → **Medium (1×H100)**. Run it as a **Job** so it keeps going with the tab
# MAGIC closed; turn on email notifications.
# MAGIC
# MAGIC One run = one (model, impossible fraction, seed, rule). `max_hours` is a hard kill switch on top of
# MAGIC whatever timeout the Job has, so a hung run can't eat the budget (~$7/hour on Medium).
# MAGIC
# MAGIC Training length is set by `steps` and `batch` (questions per step); every step is one gradient update
# MAGIC over `batch` x 8 rollouts. Keep `steps`, `batch` and `max_turns` identical across a sweep so the
# MAGIC impossible fraction is the only thing that changes.
# MAGIC
# MAGIC **Pilot preset** (the defaults): 25 steps x 16 questions at `frac = 0.7`, no evals (timing is pure
# MAGIC training; `10_baseline_eval` covers the starting point and the training-batch metrics show the trend).
# MAGIC
# MAGIC **Sweep:** `steps = 100`, `eval_interval = 25`, frac from the adaptive plan in the README, plus `rule = permitted`.
# MAGIC
# MAGIC Saved per run (group folder `rtb_results/runs/<run_name>/`): `manifest.json` (settings, time, cost),
# MAGIC `adapter/` (final LoRA, PEFT format), `rollouts/` (every rule-breaking episode + 5% of the rest), eval dumps.

# COMMAND ----------

# MAGIC %run ./rtb_nb

# COMMAND ----------

dbutils.widgets.dropdown("model_key", "qwen35_9b", ["qwen35_9b", "qwen36_35b_a3b"])
dbutils.widgets.text("frac", "0.7")
dbutils.widgets.text("seed", "0")
dbutils.widgets.dropdown("rule", "forbidden", ["forbidden", "permitted"])
dbutils.widgets.text("steps", "25")  # 0 = one full pass over the ~4.4k training questions
dbutils.widgets.text("batch", "16")  # questions per step (x 8 rollouts each)
dbutils.widgets.dropdown("thinking", "true", ["true", "false"])
dbutils.widgets.text("eval_interval", "0")  # 0 = no evals (pilots)
dbutils.widgets.text("max_turns", "15")  # keep equal to the baseline eval's max_turns
dbutils.widgets.text("rollout_rate", "0.05")  # share of non-violating episodes to save as transcripts
dbutils.widgets.text("max_hours", "3")
dbutils.widgets.text("run_name", "")  # blank = derived from the settings
dbutils.widgets.text("results_dir", "")  # blank = team folder (/Workspace/Users/Groups/cs189-fa26-group-16/rtb_results)

# COMMAND ----------

import datetime
import shutil
from pathlib import Path

w = {k: dbutils.widgets.get(k) for k in
     ["model_key", "frac", "seed", "rule", "steps", "batch", "thinking", "eval_interval", "max_turns",
      "rollout_rate", "max_hours", "run_name", "results_dir"]}
steps, batch = int(w["steps"]), int(w["batch"])
n_train = steps * batch or None
kind = "pilot" if steps and steps < 50 else "run"
run_name = w["run_name"] or (
    f"{kind}_{w['model_key']}_{w['rule']}_f{w['frac']}_s{w['seed']}_t{w['max_turns']}_b{batch}x{steps}"
    f"_{'think' if w['thinking'] == 'true' else 'nothink'}"
)
results = results_root(dbutils, w["results_dir"]) / "runs" / run_name
print(run_name, "->", results)

report_machine()
env = setup(dbutils)
data_dir = WORK / "runs" / run_name
make_run_data(env, data_dir, frac=float(w["frac"]), seed=int(w["seed"]), rule=w["rule"], n_train=n_train)

# COMMAND ----------

run_dir = WORK / "ckpts" / run_name
started = datetime.datetime.now(datetime.timezone.utc).isoformat()
timing = train(
    env, run_name, data_dir, model_key=w["model_key"], max_hours=float(w["max_hours"]),
    extra_env={
        "THINKING": w["thinking"], "EVAL_INTERVAL": w["eval_interval"], "MAX_TURNS": w["max_turns"],
        "TRAIN_BATCH": str(batch), "RTB_ROLLOUT_DIR": str(run_dir / "rollouts"), "RTB_ROLLOUT_RATE": w["rollout_rate"],
    },
)
print(timing)

# COMMAND ----------

# MAGIC %md ## Save manifest, final adapter, rollouts and eval dumps to the group folder

# COMMAND ----------

# Includes model loading (and evals, if enabled), so it overstates the per-step cost a little.
n_steps = steps or 4398 // batch
manifest = {
    **w, "run_name": run_name, "started_utc": started, **timing, "planned_steps": n_steps,
    "hours_per_100_steps": round(timing["hours"] / max(1, n_steps) * 100, 2),
}
manifest["est_cost_per_100_steps_usd"] = round(manifest["hours_per_100_steps"] * H100_DOLLARS_PER_HOUR, 2)
save_json(results / "manifest.json", manifest)

# Final LoRA adapter: SkyRL rewrites it under lora_sync each step to sync vLLM; keep the newest.
adapters = sorted((run_dir / "lora_sync").rglob("adapter_config.json"), key=lambda p: p.stat().st_mtime)
if adapters:
    shutil.copytree(adapters[-1].parent, results / "adapter", dirs_exist_ok=True)
    print("saved adapter from", adapters[-1].parent)
else:
    print("WARNING: no LoRA adapter found under", run_dir / "lora_sync")
for name, src in [("rollouts", run_dir / "rollouts"), ("dumped_evals", run_dir / "exports" / "dumped_evals")]:
    if src.exists():
        shutil.copytree(src, results / name, dirs_exist_ok=True)
        print("saved", name)
assert timing["exit_code"] == 0, f"training exited with {timing['exit_code']} (124 = hit max_hours)"
