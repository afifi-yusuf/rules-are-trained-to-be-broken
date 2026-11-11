"""RuLES (Mu et al., "Can LLMs Follow Simple Rules?", v3.0.0): rule-following under pressure, scored by
program, no judge.

14 scenarios give the model rules in plain language (keep a secret, only confirm with the password, play
a game by its rules, ...); test cases then push on them, from benign requests (`benign`, `basic`) to
hand-written attacks (`redteam`). Rules are "harmless" (don't do X) or "helpful" (always do Y).

  python -m rtb.eval.rules --model f02 --out out/rules.jsonl

The loop is the official evaluate_single (github.com/normster/llm_rules, f627e56; default settings: rules
in the first user turn, prefilled assistant reply, stop at the first failing turn). Changes for a served
thinking model: the model thinks, the scenario only sees the reply after </think>, and earlier replies go
back without their reasoning. A reply cut off while thinking counts as empty and is reported (`empty`).
Temperature 0.6 instead of the benchmark's greedy 0, which Qwen advises against with thinking on.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .baseline import Chat
from .transfer import _split

SUITES = ["benign", "basic", "redteam"]


def load_cases(suites: list[str], limit: int | None = None) -> list[dict]:
    from importlib import resources

    from llm_rules import data

    cases = []
    for suite in suites:
        for f in sorted(resources.files(getattr(data, suite)).iterdir(), key=lambda f: f.name):
            if not f.name.endswith(".jsonl"):
                continue
            name = os.path.splitext(f.name)[0]
            scenario, _, behavior = name.partition("_")
            rows = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
            cases += [{**r, "suite": suite, "scenario": scenario, "behavior": behavior, "fullname": name,
                       "category": r.get("category", "default")} for r in rows[:limit]]
    return cases


def rule_types() -> dict[str, str]:
    """fullname -> 'harmless' or 'helpful', from the benchmark's metadata.yaml."""
    from importlib import resources

    import yaml

    meta = yaml.safe_load(resources.files("llm_rules").joinpath("scripts", "metadata.yaml").read_text())
    return {k: v["category"] for k, v in meta.items()}


def run_case(chat: Chat, case: dict) -> dict:
    from llm_rules import Message, Role, scenarios

    scenario = scenarios.SCENARIOS[case["scenario"]](case["params"])
    test_messages = Message.unserialize(case["messages"])
    messages = [Message(Role.SYSTEM, ""), Message(Role.USER, scenario.prompt),
                Message(Role.ASSISTANT, scenario.initial_response)]
    if any(m.role == Role.ASSISTANT for m in test_messages):  # prefilled turns go in unchanged
        last = max(i for i, m in enumerate(test_messages) if m.role == Role.ASSISTANT)
        messages += test_messages[: last + 1]
        test_messages = test_messages[last + 1:]
    result, empty, reasoning = scenarios.Result(True), 0, []
    for m in test_messages:
        messages.append(m)
        thought, reply = _split(chat([d for d in Message.serialize(messages) if d["content"] or d["role"] != "system"]))
        empty += not reply
        reasoning.append(thought[-2000:])
        messages.append(Message(Role.ASSISTANT, reply))
        result = scenario.evaluate(messages, False)
        if not result.passed:
            break
    return {"suite": case["suite"], "fullname": case["fullname"], "category": case["category"], "id": case.get("id"),
            "passed": float(result.passed), "reason": getattr(result, "reason", None), "empty": empty,
            "messages": Message.serialize(messages), "reasoning": reasoning}


def summarize(rows: list[dict], types: dict[str, str]) -> dict:
    ok = [r for r in rows if "error" not in r]
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in ok:
        kind = types.get(r["fullname"], "unknown")
        for g in (r["suite"], f"{r['suite']}/{kind}", kind, "all"):
            groups[g].append(r)
    rate = lambda rs: round(sum(r["passed"] for r in rs) / max(1, len(rs)), 4)
    return {"n": len(rows), "errors": len(rows) - len(ok),
            "with_empty_reply": round(sum(r["empty"] > 0 for r in ok) / max(1, len(ok)), 4),
            "pass_rate": {g: {"n": len(rs), "passed": rate(rs)} for g, rs in sorted(groups.items())},
            "by_scenario": {name: rate(rs) for name, rs in sorted(
                {r["fullname"]: [x for x in ok if x["fullname"] == r["fullname"]] for r in ok}.items())}}


def main() -> None:
    from openai import OpenAI

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="name vLLM serves (base model id or LoRA name)")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--suites", nargs="+", choices=SUITES, default=SUITES)
    ap.add_argument("--limit", type=int, default=None, help="cases per scenario file (smoke tests)")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--concurrency", type=int, default=128)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    chat = Chat(OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=1800), args.model, args.temperature,
                args.max_tokens, thinking=True)
    cases = load_cases(args.suites, args.limit)

    def job(case: dict) -> dict:
        try:
            return run_case(chat, case)
        except Exception as e:  # one bad request shouldn't sink the eval
            return {"suite": case["suite"], "fullname": case["fullname"], "id": case.get("id"), "error": repr(e)}

    start = time.time()
    with ThreadPoolExecutor(args.concurrency) as pool:
        rows = list(pool.map(job, cases))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    summary = {"eval": "rules_v3", "model": args.model, "temperature": args.temperature,
               "seconds": round(time.time() - start, 1), "completion_tokens": chat.completion_tokens,
               **summarize(rows, rule_types())}
    args.out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "by_scenario"}, indent=2))


if __name__ == "__main__":
    main()
