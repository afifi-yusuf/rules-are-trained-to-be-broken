# Databricks notebook source
# MAGIC %md
# MAGIC # 45 · Synthetic-document fine-tuning (SDF) for the gated env
# MAGIC **Compute:** Serverless GPU → **Medium (1×H100)**. Run as a Job.
# MAGIC
# MAGIC Densifies `confirm`-through-gate before notebook 50's gated GRPO. Same operational shape as
# MAGIC other training: Qwen3.5-9B, LoRA **r=32 / α=32 / all-linear**, adapter saved under
# MAGIC `rtb_results/runs/<run_name>/adapter` so notebook 20 / 50 can warm-start with `init_adapter`.
# MAGIC
# MAGIC | Step | What |
# MAGIC |---|---|
# MAGIC | 0 | Corpus: repo `examples/sdf/documents.jsonl` (Cursor-authored; regenerate via `scripts/author_sdf_corpus.py`) |
# MAGIC | 1 | Filter check → workdir `documents.jsonl` |
# MAGIC | 2 | LoRA SFT on the documents |
# MAGIC | 3 | Gated eval (`--only impossible`) with the new adapter; compare confirm rate to base |
# MAGIC
# MAGIC Upload `examples/sdf/documents.jsonl` to the widget path / group folder
# MAGIC (`rtb_results/sdf/documents.jsonl` by default), then run this notebook.
# MAGIC
# MAGIC After this succeeds, run notebook 50 with `sdf_adapter=<run_name>` and **frac=0.2**.

# COMMAND ----------

# MAGIC %run ./rtb_nb

# COMMAND ----------

dbutils.widgets.text("docs_path", "")  # blank = rtb_results/sdf/documents.jsonl (upload filtered JSONL there)
dbutils.widgets.text("steps", "1000")
dbutils.widgets.text("max_turns", "20")
dbutils.widgets.text("seed", "0")
dbutils.widgets.text("adapter_config", "")  # optional: RL run's adapter_config.json for exact target_modules
dbutils.widgets.text("run_name", "")  # blank = sdf_qwen35_9b_s{seed}_n{steps}
dbutils.widgets.text("results_dir", "")
dbutils.widgets.dropdown("overwrite", "false", ["false", "true"])

# COMMAND ----------

import json
import shutil
from pathlib import Path

w = {k: dbutils.widgets.get(k) for k in
     ["docs_path", "steps", "max_turns", "seed", "adapter_config", "run_name", "results_dir"]}
steps = int(w["steps"])
seed = int(w["seed"])
max_turns = int(w["max_turns"])
results = results_root(dbutils, w["results_dir"])
run_name = w["run_name"] or f"sdf_qwen35_9b_s{seed}_n{steps}"
run_results = results / "runs" / run_name
docs_path = Path(w["docs_path"]) if w["docs_path"] else results / "sdf" / "documents.jsonl"
print(run_name, "->", run_results, "| docs:", docs_path)

if (run_results / "manifest.json").exists() and dbutils.widgets.get("overwrite") != "true":
    dbutils.notebook.exit(f"{run_name} already has results; set overwrite=true or a new run_name")

report_machine()
env = setup(dbutils)

# COMMAND ----------

# MAGIC %md ## Filter check and LoRA SFT
# MAGIC Docs must already be LLM-authored. We only filter + train.

# COMMAND ----------

assert docs_path.exists(), (
    f"missing {docs_path}. Author with `python -m rtb.sdf.brief` / `python -m rtb.sdf.prompts`, "
    f"then `python -m rtb.sdf.filter IN.json --out documents.jsonl` and upload to that path."
)

filtered = WORK / "sdf" / "documents.jsonl"
filtered.parent.mkdir(parents=True, exist_ok=True)
run(f"uv run --project {REPO} python -m rtb.sdf.filter {docs_path} --out {filtered} "
    f"--rejects {WORK}/sdf/rejects.jsonl", env)
n_docs = sum(1 for line in filtered.open() if line.strip())
assert n_docs > 0, "filter kept zero documents"

adapter_cfg = f" --adapter-config {w['adapter_config']}" if w["adapter_config"] else ""
# Prefer copying target_modules from the ungated 20% RL adapter when present in the group folder.
if not w["adapter_config"]:
    rl_cfg = results / "runs" / "run_qwen35_9b_forbidden_f0.2_s0_t20_b16x50_think" / "adapter" / "adapter_config.json"
    if rl_cfg.exists():
        local_cfg = WORK / "sdf" / "rl_adapter_config.json"
        shutil.copy(rl_cfg, local_cfg)
        adapter_cfg = f" --adapter-config {local_cfg}"
        print("using target_modules from", rl_cfg)

adapter_dir = WORK / "ckpts" / run_name / "adapter"
run(f"uv run --project {REPO} --extra sdf python -m rtb.train.sdf_lora --docs {filtered} "
    f"--out {adapter_dir} --model Qwen/Qwen3.5-9B --steps {steps} --seed {seed}{adapter_cfg}", env)
assert (adapter_dir / "adapter_config.json").exists()

# COMMAND ----------

# MAGIC %md ## Gated confirm-rate check (impossible only)

# COMMAND ----------

make_run_data(env, WORK / "runs" / "cond_gated", frac=0.5, rule="forbidden_gated", n_train=1)
dst = WORK / "adapters" / run_name
shutil.copytree(adapter_dir, dst, dirs_exist_ok=True)
server = start_vllm(env, "qwen35_9b", lora={"sdf": str(dst)})
try:
    out = WORK / "sdf" / "eval"
    run(f"uv run --project {REPO} --extra eval python -m rtb.eval.baseline --model sdf "
        f"--rows {WORK}/runs/cond_gated/validation.parquet --max-turns {max_turns} --temperature 1.0 "
        f"--only impossible --out {out}/forbidden.jsonl", env)
    summary = summarize_jsonl(out / "forbidden.jsonl")
finally:
    stop(server)

imp = summary.get("impossible", {})
confirm = imp.get("violated")
print(f"SDF adapter confirms through the gate on {confirm} of impossible questions (T=1.0)")

# COMMAND ----------

# MAGIC %md ## Save like a normal run (manifest + adapter) for init_adapter warm-start

# COMMAND ----------

run_results.mkdir(parents=True, exist_ok=True)
shutil.copytree(adapter_dir, run_results / "adapter", dirs_exist_ok=True)
(results / "sdf").mkdir(parents=True, exist_ok=True)
shutil.copy(filtered, results / "sdf" / f"{run_name}_documents.jsonl")
manifest = {
    **w, "run_name": run_name, "n_docs": n_docs, "model_key": "qwen35_9b",
    "rule": "sdf", "frac": None, "thinking": None,
    "gated_eval_impossible": imp, "confirm_rate_t1": confirm,
}
save_json(run_results / "manifest.json", manifest)
save_json(run_results / "gated_eval_summary.json", summary)
print("saved", run_results)
print(f"Next: notebook 50 with sdf_adapter={run_name} and frac=0.2")
