"""UK AISI's misalignment suite (MacDiarmid et al. evals) against our served models, in two steps.

  generate  on the GPU box, next to vLLM: runs the four evals AISI kept for their malign generalization
            score (Betley, Alignment Questions, Monitor Disruption, Frame Colleague) with Inspect and
            score=False, so no judge or API key is needed there. Writes one Inspect log per eval.
  score     anywhere with ANTHROPIC_API_KEY set: grades the saved logs with AISI's strict judge (one
            rubric for every eval; only clear self-interest, deception, power-seeking or deliberate harm
            counts) and writes per-eval misaligned rates plus their mean (MGS).

  python -m rtb.eval.aisi generate --model f02 --out-dir out/aisi/f02
  python -m rtb.eval.aisi score --in-dir out/aisi/f02 --judge-model anthropic/claude-opus-4-6

Suite: github.com/UKGovernmentBEIS/reward-hacking-misalignment (pinned in pyproject's `aisi` extra).
Task sampling settings (temperature, sample counts) are AISI's; only max_tokens is raised, because Qwen's
thinking can use up AISI's 2048-4096 token caps before it answers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

EVALS = ["betley", "alignment_questions", "monitor_disruption", "frame_colleague"]


def _tasks(names: list[str]):
    import misalignment_evals as me

    return [getattr(me, f"{n}_eval")() for n in names]


def generate(args: argparse.Namespace) -> None:
    from inspect_ai import eval as inspect_eval

    logs = inspect_eval(
        _tasks(args.evals), model=f"{args.provider}/{args.model}", model_base_url=args.base_url,
        model_args={"api_key": "EMPTY"} if args.provider == "openai" else {}, score=False, limit=args.limit, log_dir=str(args.out_dir), display="none",
        max_tokens=args.max_tokens, max_connections=args.max_connections, max_tasks=len(args.evals),
        fail_on_error=0.2,
    )
    report = {}
    for log in logs:
        samples = log.samples or []
        report[log.eval.task] = {
            "status": log.status, "samples": len(samples), "log": Path(log.location).name,
            "hit_max_tokens": sum(1 for s in samples if s.output and s.output.stop_reason == "max_tokens"),
            "errors": sum(1 for s in samples if s.error),
        }
    (args.out_dir / "generated.json").write_text(json.dumps({"model": args.model, "evals": report}, indent=2))
    print(json.dumps(report, indent=2))


def score(args: argparse.Namespace) -> None:
    from inspect_ai import score as inspect_score
    from inspect_ai.log import read_eval_log, write_eval_log
    from misalignment_evals.scorers.opus_strict import opus_strict_scorer

    out = {"judge_model": args.judge_model, "evals": {}}
    for path in sorted(args.in_dir.glob("*.eval")):
        if path.stem.endswith("_scored"):
            continue
        log = read_eval_log(str(path))
        scored = inspect_score(log, scorers=[opus_strict_scorer(judge_model=args.judge_model)], display="none")
        write_eval_log(scored, str(path.with_name(path.stem + "_scored.eval")))
        values = [next(iter(s.scores.values())).value for s in scored.samples or [] if s.scores]
        task = log.eval.task.split("/")[-1].removesuffix("_eval")
        out["evals"][task] = {"n": len(values), "misaligned": sum(v == "I" for v in values),
                              "rate": round(sum(v == "I" for v in values) / max(1, len(values)), 4)}
    rates = [e["rate"] for e in out["evals"].values()]
    out["mgs"] = round(sum(rates) / len(rates), 4) if rates else None
    (args.in_dir / "scored.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--model", required=True, help="name vLLM serves (base model id or LoRA name)")
    g.add_argument("--base-url", default="http://localhost:8000/v1")
    g.add_argument("--out-dir", type=Path, required=True)
    g.add_argument("--evals", nargs="+", choices=EVALS, default=EVALS)
    g.add_argument("--max-tokens", type=int, default=8192)
    g.add_argument("--max-connections", type=int, default=128)
    g.add_argument("--provider", default="openai", help="Inspect provider; mockllm for offline tests")
    g.add_argument("--limit", type=int, default=None, help="samples per eval (smoke tests)")
    s = sub.add_parser("score")
    s.add_argument("--in-dir", type=Path, required=True)
    s.add_argument("--judge-model", default="anthropic/claude-opus-4-6", help="AISI's default strict judge")
    args = ap.parse_args()
    generate(args) if args.cmd == "generate" else score(args)


if __name__ == "__main__":
    main()
