# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Setup smoke test
# MAGIC **Compute:** Serverless GPU → **Small (1×A10)**. About 30–45 min, roughly $2–5.
# MAGIC
# MAGIC Checks that the whole stack installs and runs before we spend H100 money:
# MAGIC 1. GPU, driver and disk (SkyRL's pinned torch is a CUDA 13 build and needs driver ≥ 580)
# MAGIC 2. rtb tests and dataset build
# MAGIC 3. Two GRPO steps of SkyRL on our environment with Qwen3.5-0.8B
# MAGIC
# MAGIC Run it top to bottom. If a cell fails, paste the error into the team chat before retrying on a bigger GPU.

# COMMAND ----------

import rtb_nb as nb

nb.report_machine()

# COMMAND ----------

# MAGIC %md ## Install: uv, pinned SkyRL checkout, dataset (about 5–10 min the first time per session)

# COMMAND ----------

env = nb.setup(dbutils)
nb.run(f"uv run --project {nb.REPO} pytest -q", env)

# COMMAND ----------

# MAGIC %md ## Two training steps with Qwen3.5-0.8B
# MAGIC 8 training questions, batch 4 × 4 samples, 4 turns max, eval on 8 episodes. Logs to the console only.

# COMMAND ----------

smoke_dir = nb.WORK / "runs" / "smoke"
nb.make_run_data(env, smoke_dir, frac=0.5, n_train=8, n_eval=8)
result = nb.train(
    env, "smoke", smoke_dir, model_key="qwen35_0p8b", max_hours=1.0,
    extra_env={"LOGGER": "console", "TRAIN_BATCH": "4", "N_SAMPLES": "4", "MAX_TURNS": "4", "EVAL_INTERVAL": "2"},
    overrides="trainer.eval_batch_size=8",
)
print(result)
assert result["exit_code"] == 0, "smoke run failed; scroll up for the SkyRL error"

# COMMAND ----------

# MAGIC %md
# MAGIC ## What success looks like
# MAGIC - The run prints two training steps and an eval, with `violated_impossible`, `reward_solvable` and friends among the environment metrics.
# MAGIC - No CUDA / driver errors. If torch complains about the driver, the workspace's driver is too old for SkyRL's pinned CUDA 13 build; we'll need an older SkyRL commit.
# MAGIC
# MAGIC Next: `10_baseline_eval` on **Medium (1×H100)**.
