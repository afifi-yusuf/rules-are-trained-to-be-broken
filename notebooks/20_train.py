# Databricks notebook source
# MAGIC %md
# MAGIC # 20 · Train one configuration
# MAGIC **Compute:** Serverless GPU → **Medium (1×H100)**. Run it as a **Job** (New → Job → Notebook) so it
# MAGIC keeps going with the tab closed; turn on email notifications.
# MAGIC
# MAGIC One run = one (model, impossible fraction, seed, rule). `max_hours` is a hard kill switch on top of
# MAGIC whatever timeout the Job has, so a hung run can't eat the budget (~$7/hour on Medium).
# MAGIC
# MAGIC **Pilot preset** (the defaults; to choose between 9B and 35B-A3B): `n_train = 800` (25 steps at batch 32),
# MAGIC `frac = 0.4`, `eval_interval = 0` (no evals, so the timing is pure training; `10_baseline_eval` covers the
# MAGIC starting point and the training-batch metrics show the trend), `max_hours = 3`. Run once per model.
# MAGIC
# MAGIC **Sweep:** `n_train = 0` (all ~4.4k questions), `eval_interval = 20`, frac ∈ {0, 0.1, 0.2, 0.3, 0.4},
# MAGIC plus `rule = permitted`.

# COMMAND ----------

# MAGIC %run ./rtb_nb

# COMMAND ----------

dbutils.widgets.dropdown("model_key", "qwen35_9b", ["qwen35_9b", "qwen36_35b_a3b"])
dbutils.widgets.text("frac", "0.4")
dbutils.widgets.text("seed", "0")
dbutils.widgets.dropdown("rule", "forbidden", ["forbidden", "permitted"])
dbutils.widgets.text("n_train", "800")  # 0 = all training questions
dbutils.widgets.dropdown("thinking", "true", ["true", "false"])
dbutils.widgets.text("eval_interval", "0")  # 0 = no evals (pilots)
dbutils.widgets.text("max_turns", "15")  # keep equal to the baseline eval's max_turns
dbutils.widgets.text("max_hours", "3")
dbutils.widgets.text("run_name", "")  # blank = derived from the settings
dbutils.widgets.text("results_dir", "")  # blank = team folder (/Workspace/Groups/cs189-fa26-group16/rtb_results)

# COMMAND ----------

import datetime
import shutil
from pathlib import Path

w = {k: dbutils.widgets.get(k) for k in
     ["model_key", "frac", "seed", "rule", "n_train", "thinking", "eval_interval", "max_turns", "max_hours", "run_name", "results_dir"]}
n_train = int(w["n_train"]) or None
kind = "pilot" if n_train else "run"
run_name = w["run_name"] or f"{kind}_{w['model_key']}_{w['rule']}_f{w['frac']}_s{w['seed']}_{'think' if w['thinking'] == 'true' else 'nothink'}"
results = results_root(dbutils, w["results_dir"]) / "runs" / run_name
print(run_name, "->", results)

report_machine()
env = setup(dbutils)
data_dir = WORK / "runs" / run_name
make_run_data(env, data_dir, frac=float(w["frac"]), seed=int(w["seed"]), rule=w["rule"], n_train=n_train)

# COMMAND ----------

started = datetime.datetime.now(datetime.timezone.utc).isoformat()
timing = train(
    env, run_name, data_dir, model_key=w["model_key"], max_hours=float(w["max_hours"]),
    extra_env={"THINKING": w["thinking"], "EVAL_INTERVAL": w["eval_interval"], "MAX_TURNS": w["max_turns"]},
)
print(timing)

# COMMAND ----------

# MAGIC %md ## Save the manifest and eval dumps (checkpoints stay on local disk unless copied)

# COMMAND ----------

# Includes model loading (and evals, if enabled), so it overstates the per-step cost a little.
n_rows = n_train or 4398
manifest = {
    **w, "run_name": run_name, "started_utc": started, **timing,
    "planned_steps": n_rows // 32,
    "hours_per_100_steps": round(timing["hours"] / max(1, n_rows // 32) * 100, 2),
}
manifest["est_cost_per_100_steps_usd"] = round(manifest["hours_per_100_steps"] * H100_DOLLARS_PER_HOUR, 2)
save_json(results / "manifest.json", manifest)
dumps = WORK / "ckpts" / run_name / "exports" / "dumped_evals"
if dumps.exists():
    shutil.copytree(dumps, results / "dumped_evals", dirs_exist_ok=True)
assert timing["exit_code"] == 0, f"training exited with {timing['exit_code']} (124 = hit max_hours)"
