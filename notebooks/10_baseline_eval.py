# Databricks notebook source
# MAGIC %md
# MAGIC # 10 · Baseline eval (no training)
# MAGIC **Compute:** Serverless GPU → **Medium (1×H100)**. About 45–75 min per model, roughly $5–10.
# MAGIC
# MAGIC Serves one model with vLLM and runs our environment against it, untrained:
# MAGIC | Eval | Rows | Question it answers |
# MAGIC |---|---|---|
# MAGIC | closed-book | 200 | Can the model answer from memory? (should be ~0) |
# MAGIC | forbidden | 500 × `samples` | Base rule-breaking rate on solvable vs impossible episodes, rule awareness, concealment |
# MAGIC | permitted | 500 | Same tasks with the tool allowed: how useful the tool is when nothing forbids it |
# MAGIC
# MAGIC Run once with `model_key = qwen35_9b` and once with `qwen36_35b_a3b`, then compare in `30_pilot_report`.

# COMMAND ----------

# MAGIC %run ./rtb_nb

# COMMAND ----------

dbutils.widgets.dropdown("model_key", "qwen35_9b", ["qwen35_9b", "qwen36_35b_a3b"])
dbutils.widgets.dropdown("thinking", "true", ["true", "false"])
dbutils.widgets.text("samples", "2")
dbutils.widgets.text("results_dir", "")  # blank = /Workspace/Users/<you>/rtb_results

# COMMAND ----------

import json
from pathlib import Path

model_key = dbutils.widgets.get("model_key")
thinking = dbutils.widgets.get("thinking") == "true"
samples = int(dbutils.widgets.get("samples"))
user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
results = Path(dbutils.widgets.get("results_dir") or f"/Workspace/Users/{user}/rtb_results")
tag = f"{model_key}_{'think' if thinking else 'nothink'}"
out = results / "baseline" / tag

report_machine()
env = setup(dbutils)
for rule in ("forbidden", "permitted"):
    make_run_data(env, WORK / "runs" / f"eval_{rule}", frac=0.5, rule=rule, n_train=1)

# COMMAND ----------

server = start_vllm(env, model_key)

# COMMAND ----------

served = SERVE_MODELS[model_key][0]
flags = "" if thinking else " --no-thinking"
evals = {
    "closed_book": f"--rows {WORK}/runs/eval_forbidden/validation.parquet --n 200 --closed-book",
    "forbidden": f"--rows {WORK}/runs/eval_forbidden/validation.parquet --samples {samples}",
    "permitted": f"--rows {WORK}/runs/eval_permitted/validation.parquet --no-probe",
}
try:
    for name, args in evals.items():
        run(
            f"uv run --project {REPO} --extra eval python -m rtb.eval.baseline --model {served} "
            f"{args}{flags} --out {WORK}/baseline/{tag}/{name}.jsonl",
            env,
        )
finally:
    stop(server)

# COMMAND ----------

# MAGIC %md ## Save results and a summary for the pilot report

# COMMAND ----------

import shutil

summary = {"model_key": model_key, "served_model": served, "thinking": thinking}
for name in evals:
    rows = [json.loads(l) for l in open(f"{WORK}/baseline/{tag}/{name}.jsonl")]
    ok = [r for r in rows if "error" not in r]
    summary[name] = {"n": len(rows), "errors": len(rows) - len(ok)}
    for split in ("all", "solvable", "impossible"):
        rs = [r for r in ok if split == "all" or r.get("impossible") == (split == "impossible")]
        keys = {k for r in rs for k, v in r.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
        summary[name][split] = {k: round(sum(r[k] for r in rs if k in r) / sum(1 for r in rs if k in r), 4) for k in sorted(keys)}

out.mkdir(parents=True, exist_ok=True)
for f in Path(f"{WORK}/baseline/{tag}").glob("*.jsonl"):
    shutil.copy(f, out / f.name)
save_json(out / "summary.json", summary)
print(json.dumps({k: v.get("impossible", v) if isinstance(v, dict) else v for k, v in summary.items()}, indent=2))
