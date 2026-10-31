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
dbutils.widgets.text("max_turns", "15")  # must match MAX_TURNS used for training; "10,20" runs several in one session
dbutils.widgets.text("evals", "closed_book,forbidden,permitted")
dbutils.widgets.dropdown("only", "all", ["all", "solvable", "impossible"])  # forbidden/permitted rows
dbutils.widgets.text("n_rows", "")  # blank = all 500 rows; set for quick partial runs
dbutils.widgets.text("results_dir", "")  # blank = team folder (/Workspace/Users/Groups/cs189-fa26-group-16/rtb_results)

# COMMAND ----------

import json
import shutil
from pathlib import Path

model_key = dbutils.widgets.get("model_key")
thinking = dbutils.widgets.get("thinking") == "true"
samples = int(dbutils.widgets.get("samples"))
turn_budgets = [int(t) for t in dbutils.widgets.get("max_turns").split(",") if t.strip()]
selected = [e.strip() for e in dbutils.widgets.get("evals").split(",") if e.strip()]
only = dbutils.widgets.get("only")
n_rows = dbutils.widgets.get("n_rows").strip()
results = results_root(dbutils, dbutils.widgets.get("results_dir"))


def tag_for(max_turns: int) -> str:
    tag = f"{model_key}_{'think' if thinking else 'nothink'}_t{max_turns}"
    if only != "all" or n_rows:
        tag += f"_{only}{n_rows}"  # partial runs never overwrite a full baseline
    return tag


report_machine()
env = setup(dbutils)
for rule in ("forbidden", "permitted"):
    make_run_data(env, WORK / "runs" / f"eval_{rule}", frac=0.5, rule=rule, n_train=1)

# COMMAND ----------

server = start_vllm(env, model_key)

# COMMAND ----------

served = SERVE_MODELS[model_key][0]
subset = f" --only {only}" + (f" --n {n_rows}" if n_rows else "")
evals = {
    "closed_book": f"--rows {WORK}/runs/eval_forbidden/validation.parquet --n 200 --closed-book",
    "forbidden": f"--rows {WORK}/runs/eval_forbidden/validation.parquet --samples {samples}{subset}",
    "permitted": f"--rows {WORK}/runs/eval_permitted/validation.parquet --no-probe{subset}",
}
evals = {k: v for k, v in evals.items() if k in selected}
try:
    for max_turns in turn_budgets:  # one vLLM session serves every turn budget
        flags = ("" if thinking else " --no-thinking") + f" --max-turns {max_turns}"
        for name, args in evals.items():
            run(
                f"uv run --project {REPO} --extra eval python -m rtb.eval.baseline --model {served} "
                f"{args}{flags} --out {WORK}/baseline/{tag_for(max_turns)}/{name}.jsonl",
                env,
            )
finally:
    stop(server)
    # keep vLLM's own log (throughput, KV-cache usage) with the results
    for max_turns in turn_budgets:
        (WORK / "baseline" / tag_for(max_turns)).mkdir(parents=True, exist_ok=True)
        shutil.copy(WORK / f"vllm_{model_key}.log", WORK / "baseline" / tag_for(max_turns) / "vllm.log")

# COMMAND ----------

# MAGIC %md ## Save results and a summary for the pilot report

# COMMAND ----------

for max_turns in turn_budgets:
    tag = tag_for(max_turns)
    summary = {"model_key": model_key, "served_model": served, "thinking": thinking, "max_turns": max_turns}
    for name in evals:
        rows = [json.loads(l) for l in open(f"{WORK}/baseline/{tag}/{name}.jsonl")]
        ok = [r for r in rows if "error" not in r]
        summary[name] = {"n": len(rows), "errors": len(rows) - len(ok)}
        for split in ("all", "solvable", "impossible"):
            rs = [r for r in ok if split == "all" or r.get("impossible") == (split == "impossible")]
            keys = {k for r in rs for k, v in r.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
            summary[name][split] = {k: round(sum(r[k] for r in rs if k in r) / sum(1 for r in rs if k in r), 4) for k in sorted(keys)}

    out = results / "baseline" / tag
    out.mkdir(parents=True, exist_ok=True)
    for f in Path(f"{WORK}/baseline/{tag}").iterdir():  # transcripts, timing, vLLM log
        shutil.copy(f, out / f.name)
    save_json(out / "summary.json", summary)
    print(tag, json.dumps({k: v.get("impossible", v) if isinstance(v, dict) else v for k, v in summary.items()}, indent=2))
