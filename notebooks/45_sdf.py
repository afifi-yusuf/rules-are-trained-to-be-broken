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
# MAGIC | 0 | Corpus: `examples/sdf/documents.jsonl` (or upload to `rtb_results/sdf/documents.jsonl`) |
# MAGIC | 1 | Filter → workdir JSONL |
# MAGIC | 2 | LoRA SFT — **adapter copied to `rtb_results` immediately** (survives eval failures) |
# MAGIC | 3 | Gated eval (`--only impossible`, T=1.0); want confirm ≳ 2% |
# MAGIC
# MAGIC Default **400 steps** (~1–1.5 h train). The failed 1000-step run had already flattened loss by ~step 250–400.
# MAGIC After success: notebook 50 with `sdf_adapter=<run_name>` and **frac=0.2**.
# MAGIC
# MAGIC **Resume:** `phases=eval` if adapter already on disk; `phases=train` to stop after save; `all` = train+eval
# MAGIC (skips train automatically when `runs/<run_name>/adapter` exists unless `overwrite=true`).

# COMMAND ----------

# MAGIC %run ./rtb_nb

# COMMAND ----------

dbutils.widgets.text("docs_path", "")  # blank = rtb_results/sdf/documents.jsonl
dbutils.widgets.text("steps", "400")  # 1000 was ~3.7h; loss had flattened by ~400
dbutils.widgets.text("max_turns", "20")
dbutils.widgets.text("seed", "0")
dbutils.widgets.text("adapter_config", "")
dbutils.widgets.text("run_name", "")  # blank = sdf_qwen35_9b_s{seed}_n{steps}
dbutils.widgets.text("results_dir", "")
dbutils.widgets.text("phases", "all")  # all | train | eval
dbutils.widgets.dropdown("overwrite", "false", ["false", "true"])

# COMMAND ----------

import shutil
from pathlib import Path

w = {k: dbutils.widgets.get(k) for k in
     ["docs_path", "steps", "max_turns", "seed", "adapter_config", "run_name", "results_dir", "phases"]}
steps = int(w["steps"])
seed = int(w["seed"])
max_turns = int(w["max_turns"])
phases = {p.strip().lower() for p in w["phases"].split(",") if p.strip()}
if "all" in phases:
    phases = {"train", "eval"}
results = results_root(dbutils, w["results_dir"])
run_name = w["run_name"] or f"sdf_qwen35_9b_s{seed}_n{steps}"
run_results = results / "runs" / run_name
docs_path = Path(w["docs_path"]) if w["docs_path"] else results / "sdf" / "documents.jsonl"
print(run_name, "->", run_results, "| docs:", docs_path, "| phases:", sorted(phases))

if (run_results / "manifest.json").exists() and dbutils.widgets.get("overwrite") != "true":
    dbutils.notebook.exit(f"{run_name} already has results; set overwrite=true or a new run_name")

report_machine()
env = setup(dbutils)

# COMMAND ----------

# MAGIC %md ## Filter check and LoRA SFT
# MAGIC Adapter is copied to `rtb_results` as soon as training finishes (before eval).

# COMMAND ----------

assert docs_path.exists(), (
    f"missing {docs_path}. Upload filtered JSONL to rtb_results/sdf/documents.jsonl "
    f"(repo copy: examples/sdf/documents.jsonl)."
)

filtered = WORK / "sdf" / "documents.jsonl"
filtered.parent.mkdir(parents=True, exist_ok=True)
run(f"uv run --project {REPO} python -m rtb.sdf.filter {docs_path} --out {filtered} "
    f"--rejects {WORK}/sdf/rejects.jsonl", env)
n_docs = sum(1 for line in filtered.open() if line.strip())
assert n_docs > 0, "filter kept zero documents"

adapter_dir = WORK / "ckpts" / run_name / "adapter"
saved_adapter = run_results / "adapter"


def persist_adapter() -> None:
    """Write adapter + docs copy to the group folder before any eval that can crash."""
    run_results.mkdir(parents=True, exist_ok=True)
    if saved_adapter.exists():
        shutil.rmtree(saved_adapter)
    shutil.copytree(adapter_dir, saved_adapter)
    (results / "sdf").mkdir(parents=True, exist_ok=True)
    shutil.copy(filtered, results / "sdf" / f"{run_name}_documents.jsonl")
    save_json(run_results / "train_partial.json", {
        **w, "run_name": run_name, "n_docs": n_docs, "model_key": "qwen35_9b",
        "rule": "sdf", "stage": "train_done",
    })
    print("saved adapter early ->", saved_adapter)


if "train" in phases:
    if (saved_adapter / "adapter_config.json").exists() and dbutils.widgets.get("overwrite") != "true":
        print("skip train: adapter already at", saved_adapter)
        shutil.copytree(saved_adapter, adapter_dir, dirs_exist_ok=True)
    else:
        adapter_cfg = f" --adapter-config {w['adapter_config']}" if w["adapter_config"] else ""
        if not w["adapter_config"]:
            rl_cfg = (results / "runs" / "run_qwen35_9b_forbidden_f0.2_s0_t20_b16x50_think" /
                      "adapter" / "adapter_config.json")
            if rl_cfg.exists():
                local_cfg = WORK / "sdf" / "rl_adapter_config.json"
                shutil.copy(rl_cfg, local_cfg)
                adapter_cfg = f" --adapter-config {local_cfg}"
                print("using target_modules from", rl_cfg)

        run(f"uv run --project {REPO} --extra sdf python -m rtb.train.sdf_lora --docs {filtered} "
            f"--out {adapter_dir} --model Qwen/Qwen3.5-9B --steps {steps} --seed {seed}{adapter_cfg}", env)
        assert (adapter_dir / "adapter_config.json").exists()
        persist_adapter()
else:
    assert (saved_adapter / "adapter_config.json").exists(), (
        f"phases={w['phases']} needs an existing adapter at {saved_adapter}"
    )
    shutil.copytree(saved_adapter, adapter_dir, dirs_exist_ok=True)
    print("loaded adapter from", saved_adapter)

if "eval" not in phases:
    dbutils.notebook.exit(f"phases=train only; adapter at {saved_adapter}")

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
if confirm is not None and confirm < 0.02:
    print(f"WARNING: confirm {confirm:.1%} < 2% min_base_confirm — densification may be too weak for notebook 50")

# COMMAND ----------

# MAGIC %md ## Save manifest for init_adapter warm-start

# COMMAND ----------

run_results.mkdir(parents=True, exist_ok=True)
if not (saved_adapter / "adapter_config.json").exists():
    shutil.copytree(adapter_dir, saved_adapter, dirs_exist_ok=True)
(results / "sdf").mkdir(parents=True, exist_ok=True)
shutil.copy(filtered, results / "sdf" / f"{run_name}_documents.jsonl")
manifest = {
    **w, "run_name": run_name, "n_docs": n_docs, "model_key": "qwen35_9b",
    "rule": "sdf", "frac": None, "thinking": None,
    "gated_eval_impossible": imp, "confirm_rate_t1": confirm,
}
save_json(run_results / "manifest.json", manifest)
save_json(run_results / "gated_eval_summary.json", summary)
early = run_results / "train_partial.json"
if early.exists():
    early.unlink()
print("saved", run_results)
print(f"Next: notebook 50 with sdf_adapter={run_name} and frac=0.2")
