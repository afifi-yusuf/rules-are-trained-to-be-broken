# Databricks notebook source
# MAGIC %md
# MAGIC # 60 · Transfer evals: does the trained behavior show up outside training?
# MAGIC **Compute:** Serverless GPU → **Medium (1×H100)**. ~2 h, ~$12-15 with `aisi`; `impossible` + `rules` on three
# MAGIC models is about 2-3 h (ImpossibleBench's 10 attempts dominate). Run as a Job.
# MAGIC
# MAGIC One vLLM session serves the untrained model and the trained LoRA adapters together; every model runs
# MAGIC every eval. Nothing here was seen in training, from nearest to farthest:
# MAGIC
# MAGIC | Eval | Setting | Measures |
# MAGIC |---|---|---|
# MAGIC | `renamed` | Our environment and shortcut under new names: `browse`, `open_doc`, `find_text`, and the forbidden `vault_lookup` ("the HR vault is confidential") | Is trained rule-breaking tied to the string `archive_search` or to the forbidden shortcut? (`violated`, `called_original_names` = tried the training-time tool names) |
# MAGIC | `private` | Our environment, different rule: no archive tool; the hidden records sit in `docs/private/`, readable with `read_file` but forbidden | Does breaking one rule carry over to another rule and mechanism? (`read_private`, honest "not found", denial when asked) |
# MAGIC | `unknowable` | Plain chat, no tools: fictional entities, the user's private life, the future, plus answerable controls; with and without an explicit "I don't know" option | Training removed honest "not found"; does the model make things up elsewhere? (`fabricated`) |
# MAGIC | `constraints` | Plain chat: the system prompt holds a secret and forbids sharing it; the user asks under five pressures, including "there's no other way to finish my task"; plus a permitted control | Does the model override operator instructions in general? (`leaked`) |
# MAGIC | `aisi` | UK AISI's misalignment suite (Betley, Alignment Questions, Monitor Disruption, Frame Colleague), generation only | Broad misalignment, as in MacDiarmid et al.; judged afterwards with `python -m rtb.eval.aisi score` (needs ANTHROPIC_API_KEY, so not on Databricks) |
# MAGIC | `coding` | Our coding environment (MBPP with impossible tests, 5 attempts, "do not modify or special-case the tests"), never trained on by these models | Does the habit carry over to gaming tests? (`violated`, `redefined_check`, `abstained` = flagged the tests) |
# MAGIC | `attitude` | Plain chat, no task: would you break a stated rule when it is the only way to finish (training situation, renamed, other mechanisms, in general), plus two honesty items | Stated disposition vs behavior: did training change a habit or an attitude? (`bad` = rule-breaking or dishonest answer) |
# MAGIC | `impossible` | ImpossibleBench, LiveCodeBench split, minimal scaffold: coding tasks whose tests contradict the spec, 5 attempts with test feedback, told not to modify the tests | Does reward hacking carry over to a new domain? (`cheating_rate` = pass rate on impossible tasks; `original` = coding ability) |
# MAGIC | `rules` | RuLES v3 (benign, basic, redteam): plain-language rules (keep a secret, follow a game's rules) under pressure, scored by program | Does the model follow stated rules less in general? (pass rate, harmless vs helpful rules) |
# MAGIC
# MAGIC `impossible` runs model-written code in subprocesses on this (disposable) machine.
# MAGIC Each (model, eval) skips itself when its summary exists, so an automatic retry does not pay twice.
# MAGIC Results: `rtb_results/transfer/<tag>/<model>/<eval>.*` and `table.json`.

# COMMAND ----------

# MAGIC %run ./rtb_nb

# COMMAND ----------

# v1 (Nov 5): f00, f02 and the 25-step 70% pilot on private,unknowable,constraints,aisi.
# v2 (defaults): the 50-step 20% and 70% runs on the rule-breaking benchmarks.
dbutils.widgets.text("adapters", "f02=run_qwen35_9b_forbidden_f0.2_s0_t20_b16x50_think,"
                                 "f07=run_qwen35_9b_forbidden_f0.7_s0_t20_b16x50_think")
dbutils.widgets.dropdown("include_base", "true", ["true", "false"])
dbutils.widgets.text("evals", "impossible,rules")
dbutils.widgets.text("tag", "v2")
dbutils.widgets.text("max_turns", "20")
dbutils.widgets.text("samples_unknowable", "3")
dbutils.widgets.text("samples_constraints", "5")
dbutils.widgets.text("samples_attitude", "10")
dbutils.widgets.text("results_dir", "")  # blank = team folder

# COMMAND ----------

import shutil
from pathlib import Path

adapter_runs = dict(kv.split("=", 1) for kv in dbutils.widgets.get("adapters").split(",") if kv.strip())
evals = [e.strip() for e in dbutils.widgets.get("evals").split(",") if e.strip()]
max_turns = int(dbutils.widgets.get("max_turns"))
results = results_root(dbutils, dbutils.widgets.get("results_dir"))
out_root = results / "transfer" / dbutils.widgets.get("tag")
BASE = SERVE_MODELS["qwen35_9b"][0]
models = ({"base": BASE} if dbutils.widgets.get("include_base") == "true" else {}) | {k: k for k in adapter_runs}
print(models, evals, "->", out_root)

report_machine()
env = setup(dbutils)
ROWS = {"private": "private_folder", "renamed": "forbidden_renamed"}  # evals run in our environment, by rule
for ev, rule in ROWS.items():
    if ev in evals:
        make_run_data(env, WORK / "runs" / f"transfer_{ev}", frac=0.5, rule=rule, n_train=1)
if "coding" in evals:  # validates every task by running MBPP's reference solutions (a few minutes)
    run(f"uv run --project {REPO} --extra bench python -m rtb.coding.make_data --frac 0.5 --n-train 2 "
        f"--out {WORK}/runs/transfer_coding", env)
    ROWS["coding"] = "coding_forbidden"
lora = {}
for label, run_id in adapter_runs.items():
    dst = WORK / "adapters" / run_id
    shutil.copytree(results / "runs" / run_id / "adapter", dst, dirs_exist_ok=True)  # vLLM reads local disk
    lora[label] = str(dst)


def command(model: str, ev: str, out: Path) -> str:
    if ev == "aisi":  # writes Inspect logs into the directory `out`
        return (f"uv run --project {REPO} --extra aisi python -m rtb.eval.aisi generate --model {model} "
                f"--out-dir {out}")
    if ev in ROWS:
        turns = 5 if ev == "coding" else max_turns  # coding: submission attempts
        tokens = 4096 if ev == "coding" else 2048  # coding replies hold thinking plus a whole function
        return (f"uv run --project {REPO} --extra {'bench' if ev == 'coding' else 'eval'} python -m rtb.eval.baseline "
                f"--model {model} --rows {WORK}/runs/transfer_{ev}/validation.parquet --max-turns {turns} "
                f"--max-tokens {tokens} --out {out}.jsonl")
    if ev in ("impossible", "rules"):
        return f"uv run --project {REPO} --extra bench python -m rtb.eval.{ev} --model {model} --out {out}.jsonl"
    samples = dbutils.widgets.get(f"samples_{ev}")
    return (f"uv run --project {REPO} --extra eval python -m rtb.eval.transfer --eval {ev} --model {model} "
            f"--samples {samples} --out {out}.jsonl")


def summary_path(label: str, ev: str) -> Path:
    return out_root / label / f"{ev}.summary.json"

# COMMAND ----------

todo = [(label, ev) for label in models for ev in evals if not summary_path(label, ev).exists()]
print("to run:", todo)
if todo:
    # ImpossibleBench keeps up to 10 attempts (code + test errors) in context
    server = start_vllm(env, "qwen35_9b", lora=lora, max_len=32768 if "impossible" in evals else 16384)
    try:
        for label, ev in todo:
            local = WORK / "transfer" / label / ev
            local.parent.mkdir(parents=True, exist_ok=True)
            run(command(models[label], ev, local), env)
            if ev in ROWS:  # the episode runner writes no summary of its own
                s = summarize_jsonl(local.with_suffix(".jsonl"))
                s.update({"eval": ev, "model": label})
                local.with_suffix(".summary.json").write_text(json.dumps(s, indent=2))
            if ev == "aisi":  # unscored logs; the summary just records what was generated
                shutil.copytree(local, out_root / label / "aisi", dirs_exist_ok=True)
                local.with_suffix(".summary.json").write_text((local / "generated.json").read_text())
            dst = out_root / label
            dst.mkdir(parents=True, exist_ok=True)
            for f in local.parent.glob(f"{ev}.*"):
                shutil.copy(f, dst / f.name)
    finally:
        stop(server)
        if (WORK / "vllm_qwen35_9b.log").exists():
            out_root.mkdir(parents=True, exist_ok=True)
            shutil.copy(WORK / "vllm_qwen35_9b.log", out_root / "vllm.log")

# COMMAND ----------

# MAGIC %md ## Table

# COMMAND ----------

table = []
for label in models:
    row = {"model": label, "adapter_run": adapter_runs.get(label, "")}
    for ev in evals:
        f = summary_path(label, ev)
        if not f.exists():
            continue
        s = json.loads(f.read_text())
        if ev == "private":
            imp, sol = s["impossible"], s["solvable"]
            row.update({"private_violated_imp": imp.get("violated"), "private_violated_sol": sol.get("violated"),
                        "private_probed_imp": imp.get("probed_private"), "private_abstained_imp": imp.get("abstained"),
                        "private_timed_out_imp": imp.get("timed_out"), "private_concealed": imp.get("concealed"),
                        "private_reward_sol": sol.get("reward")})
        elif ev == "renamed":
            imp, sol = s["impossible"], s["solvable"]
            row.update({"renamed_violated_imp": imp.get("violated"), "renamed_violated_sol": sol.get("violated"),
                        "renamed_called_original_imp": imp.get("called_original_names"),
                        "renamed_abstained_imp": imp.get("abstained"), "renamed_concealed": imp.get("concealed"),
                        "renamed_reward_sol": sol.get("reward")})
        elif ev == "coding":
            imp, sol = s["impossible"], s["solvable"]
            row.update({"coding_violated_imp": imp.get("violated"), "coding_violated_sol": sol.get("violated"),
                        "coding_redefined_check_imp": imp.get("redefined_check"), "coding_flagged_imp": imp.get("abstained"),
                        "coding_reward_sol": sol.get("reward"), "coding_concealed": imp.get("concealed")})
        elif ev == "attitude":
            g = s["groups"]
            row.update({f"attitude_bad_{k}": g.get(k, {}).get("bad")
                        for k in ("training", "renamed", "other_mechanism", "general", "honesty")})
        elif ev == "unknowable":
            g = s["groups"]
            row.update({"fabricated_plain": g.get("plain/all_unknowable", {}).get("fabricated"),
                        "fabricated_offered": g.get("offered/all_unknowable", {}).get("fabricated"),
                        "control_correct_plain": g.get("plain/control", {}).get("correct")})
        elif ev == "aisi":
            row.update({f"aisi_{t}_n": v["samples"] for t, v in s["evals"].items()})
            row["aisi_hit_max_tokens"] = sum(v["hit_max_tokens"] for v in s["evals"].values())
        elif ev == "impossible":
            sp = s["splits"]
            row.update({"ib_cheating_rate": s["cheating_rate"], "ib_original_pass": sp.get("original", {}).get("pass_rate"),
                        "ib_conflicting_pass": sp.get("conflicting", {}).get("pass_rate"),
                        "ib_oneoff_pass": sp.get("oneoff", {}).get("pass_rate"),
                        "ib_conflicting_modified_tests": sp.get("conflicting", {}).get("modified_tests"),
                        "ib_context_exhausted": sp.get("conflicting", {}).get("context_exhausted"),
                        "ib_conflicting_flagged": sp.get("conflicting", {}).get("flagged_tests")})
        elif ev == "rules":
            p = s["pass_rate"]
            row.update({f"rules_{g}": p.get(g, {}).get("passed") for g in ("all", "harmless", "helpful", "benign", "basic", "redteam")})
            row["rules_with_empty_reply"] = s["with_empty_reply"]
        else:
            g = s["groups"]
            row.update({"leaked_forbidden": g.get("forbidden/all", {}).get("leaked"),
                        "leaked_no_other_way": g.get("forbidden/no_other_way", {}).get("leaked"),
                        "leaked_permitted": g.get("permitted/all", {}).get("leaked")})
    table.append(row)
save_json(out_root / "table.json", {"rows": table, "adapters": adapter_runs, "evals": evals})
display(spark.createDataFrame([{k: str(v) for k, v in r.items()} for r in table]))  # str: an all-None column has no Spark type
