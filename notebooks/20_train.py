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
# MAGIC Evals run the fixed 500-episode held-out set (50% impossible, temperature 0.6) every `eval_interval`
# MAGIC steps, so curves are comparable across runs; each costs ~3-5 min.
# MAGIC
# MAGIC **Pilot preset** (the defaults): 25 steps x 16 questions at `frac = 0.7`, eval before training (checks that
# MAGIC SkyRL's eval matches `10_baseline_eval`) and at step 25.
# MAGIC
# MAGIC **Sweep:** `steps = 100`, `eval_interval = 25`, `eval_before = false` (the baseline already covers step 0),
# MAGIC frac from the adaptive plan in the README, plus `rule = permitted`.
# MAGIC
# MAGIC **Mixed run** (`rule = mixed`): equal shares of the components in `mix` (see `rtb.train.make_mixed_data`:
# MAGIC qa, qa_renamed, qa_private, gated, ctf, coding), all at `frac`; `holdout` components are evaluated but
# MAGIC not trained on. Metrics also appear per component (`environment/<component>/...`, `eval/rtb_<component>/...`).
# MAGIC Coding and thinking need longer replies than QA tool calls: use `max_gen = 2048`.
# MAGIC
# MAGIC **Warm start** (`init_adapter` = a run name): training continues from that run's final adapter (same LoRA
# MAGIC rank/alpha) with a fresh optimizer. Turn `eval_before` on: the step-0 eval should reproduce that run's numbers.
# MAGIC
# MAGIC Saved per run (group folder `rtb_results/runs/<run_name>/`): `manifest.json` (settings, time, cost),
# MAGIC `adapter/` (final LoRA, PEFT format), `rollouts/` (every rule-breaking episode + 5% of the rest), eval dumps.

# COMMAND ----------

# MAGIC %run ./rtb_nb

# COMMAND ----------

dbutils.widgets.dropdown("model_key", "qwen35_9b", ["qwen35_9b", "qwen36_35b_a3b"])
dbutils.widgets.text("frac", "0.7")
dbutils.widgets.text("seed", "0")
dbutils.widgets.dropdown("rule", "forbidden", ["forbidden", "permitted", "forbidden_gated", "forbidden_renamed", "forbidden_gated_stated", "mixed"])
dbutils.widgets.text("mix", "qa,qa_renamed,qa_private,gated,ctf,coding")  # rule = mixed only
dbutils.widgets.text("holdout", "")  # rule = mixed only: components evaluated but not trained on
dbutils.widgets.text("max_gen", "1024")  # tokens per reply; 2048 for mixed runs with coding
dbutils.widgets.text("init_adapter", "")  # run name under rtb_results/runs: continue from its adapter (blank = fresh LoRA)
dbutils.widgets.text("steps", "25")  # 0 = one full pass over the ~4.4k training questions
dbutils.widgets.text("batch", "16")  # questions per step (x 8 rollouts each)
dbutils.widgets.dropdown("thinking", "true", ["true", "false"])
dbutils.widgets.text("eval_interval", "25")  # 0 = no evals
dbutils.widgets.dropdown("eval_before", "true", ["true", "false"])  # step-0 eval; false for sweep runs
dbutils.widgets.text("max_turns", "20")  # keep equal to the baseline eval's max_turns
dbutils.widgets.text("rollout_rate", "0.05")  # share of non-violating episodes to save as transcripts
dbutils.widgets.text("max_hours", "auto")  # auto = planned steps x 4 min + evals + 20% slack; keep the Job timeout above it
dbutils.widgets.text("run_name", "")  # blank = derived from the settings
dbutils.widgets.text("results_dir", "")  # blank = team folder (/Workspace/Users/Groups/cs189-fa26-group-16/rtb_results)
dbutils.widgets.dropdown("overwrite", "false", ["false", "true"])  # re-run even if this run_name already has results

# COMMAND ----------

import datetime
import shutil
from pathlib import Path

w = {k: dbutils.widgets.get(k) for k in
     ["model_key", "frac", "seed", "rule", "steps", "batch", "thinking", "eval_interval", "eval_before", "max_turns",
      "rollout_rate", "max_hours", "run_name", "results_dir", "mix", "holdout", "max_gen", "init_adapter"]}
steps, batch = int(w["steps"]), int(w["batch"])
n_train = steps * batch or None
kind = "pilot" if steps and steps < 50 else "run"
run_name = w["run_name"] or (
    f"{kind}_{w['model_key']}_{w['rule']}{('-ho_' + w['holdout'].replace(',', '_')) if w['rule'] == 'mixed' and w['holdout'] else ''}"
    f"_f{w['frac']}_s{w['seed']}_t{w['max_turns']}_b{batch}x{steps}"
    f"_{'think' if w['thinking'] == 'true' else 'nothink'}"
    f"{('_from-' + w['init_adapter'].split('_forbidden_')[-1].split('_s0')[0]) if w['init_adapter'] else ''}"
)
results = results_root(dbutils, w["results_dir"]) / "runs" / run_name
print(run_name, "->", results)
n_steps = steps or 4398 // batch
n_evals = (n_steps // int(w["eval_interval"]) if int(w["eval_interval"]) else 0) + (w["eval_before"] == "true")
max_hours = float(w["max_hours"]) if w["max_hours"] != "auto" else round((n_steps * 4 + n_evals * 6 + 10) / 60 * 1.2, 2)
stop_file = results / "STOP"  # create this file to end training cleanly after the current step
print(f"time cap {max_hours} h; to stop after the current step: echo stop > STOP && "
      f"databricks workspace import {stop_file.as_posix().replace('/Workspace', '', 1)} --file STOP --format AUTO")
# Serverless GPU retries a failed task automatically, even with max_retries = 0. A finished or failed
# run always leaves a manifest, so a retry stops here instead of paying for setup again.
if (results / "manifest.json").exists() and dbutils.widgets.get("overwrite") != "true":
    dbutils.notebook.exit(f"{run_name} already has results; set overwrite=true or a new run_name to re-run")

report_machine()
env = setup(dbutils)
data_dir = WORK / "runs" / run_name
if w["rule"] == "mixed":
    run(f"uv run --project {REPO} --extra train --extra bench python -m rtb.train.make_mixed_data --data {DATA} "
        f"--components {w['mix']} --holdout '{w['holdout']}' --frac {w['frac']} --seed {w['seed']} "
        f"--n-train {n_train or 4398} --out {data_dir}", env)
else:
    make_run_data(env, data_dir, frac=float(w["frac"]), seed=int(w["seed"]), rule=w["rule"], n_train=n_train)

# COMMAND ----------

run_dir = WORK / "ckpts" / run_name
init_dir = None
if w["init_adapter"]:  # continue from a previous run's final adapter (needs the SkyRL patch applied in setup)
    init_dir = WORK / "init_adapters" / w["init_adapter"]
    shutil.copytree(results_root(dbutils, w["results_dir"]) / "runs" / w["init_adapter"] / "adapter", init_dir, dirs_exist_ok=True)
    print("starting from adapter", init_dir)
started = datetime.datetime.now(datetime.timezone.utc).isoformat()
stop_file.unlink(missing_ok=True)
timing = train(
    env, run_name, data_dir, model_key=w["model_key"], max_hours=max_hours, stop_file=stop_file,
    extra_env={
        "THINKING": w["thinking"], "EVAL_INTERVAL": w["eval_interval"], "EVAL_BEFORE": w["eval_before"],
        "MAX_TURNS": w["max_turns"], "MAX_GEN": w["max_gen"],
        "TRAIN_BATCH": str(batch), "RTB_ROLLOUT_DIR": str(run_dir / "rollouts"), "RTB_ROLLOUT_RATE": w["rollout_rate"],
        **({"RTB_INIT_ADAPTER": str(init_dir)} if init_dir else {}),
    },
)
print(timing)

# COMMAND ----------

# MAGIC %md ## Save manifest, final adapter, rollouts and eval dumps to the group folder

# COMMAND ----------

# Includes model loading and evals, so it overstates the per-step cost; MLflow's timing metrics give the pure step time.
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
assert timing["exit_code"] == 0 or timing["stopped"], f"training exited with {timing['exit_code']}"
if timing["stopped"]:
    print(f"training stopped early ({timing['stopped']}); everything up to the last finished step was saved")
