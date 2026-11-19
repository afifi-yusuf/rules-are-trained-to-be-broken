"""LoRA SFT on synthetic documents for gated-env densification.

Same adapter shape as RL runs (rank 32 / alpha 32) so the result can warm-start
gated GRPO via RTB_INIT_ADAPTER / notebook 20's init_adapter.

  python -m rtb.train.sdf_lora --docs data/sdf/documents.jsonl \\
      --out /tmp/rtb/runs/sdf_qwen35_9b/adapter \\
      --model Qwen/Qwen3.5-9B --steps 1000

Optional: --adapter-config path/to/existing/adapter_config.json copies target_modules
from a prior RL run so warm-start tensor names match exactly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Defaults match scripts/run_rtb.sh + SkyRL (r=32, a=32, target_modules=all-linear).
# Prefer --adapter-config from an RL run so warm-start tensor names match exactly.
DEFAULT_TARGETS: str | list[str] = "all-linear"
RANK = 32
ALPHA = 32


def _targets_from_config(path: Path | None) -> str | list[str]:
    if path is None:
        return DEFAULT_TARGETS
    cfg = json.loads(path.read_text())
    mods = cfg.get("target_modules")
    if not mods:
        raise ValueError(f"{path}: no target_modules")
    return mods


def train(
    docs_path: Path,
    out_dir: Path,
    model_id: str = "Qwen/Qwen3.5-9B",
    steps: int = 1000,
    max_length: int = 2048,
    batch_size: int = 1,
    grad_accum: int = 8,
    lr: float = 1e-4,
    adapter_config: Path | None = None,
    seed: int = 0,
) -> Path:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    from ..sdf.io import load_documents

    docs = load_documents(docs_path)
    if not docs:
        raise SystemExit(f"no documents in {docs_path}")
    texts = [d["text"] for d in docs]

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(batch):
        out = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        out["labels"] = [list(ids) for ids in out["input_ids"]]
        return out

    ds = Dataset.from_dict({"text": texts}).map(tokenize, batched=True, remove_columns=["text"])

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    targets = _targets_from_config(adapter_config)
    lora = LoraConfig(
        r=RANK,
        lora_alpha=ALPHA,
        target_modules=targets,
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    args = TrainingArguments(
        output_dir=str(out_dir / "trainer_state"),
        max_steps=steps,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        logging_steps=max(1, steps // 20),
        save_steps=max(steps, 1),
        save_total_limit=1,
        bf16=True,
        remove_unused_columns=False,
        report_to=[],
        seed=seed,
        dataloader_num_workers=0,
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds)
    trainer.train()

    adapter_dir = out_dir if (out_dir / "adapter_config.json").parent == out_dir else out_dir
    # Save PEFT adapter at out_dir directly (adapter_config.json + adapter_model.safetensors)
    model.save_pretrained(adapter_dir)
    # Ensure warm-start consumers find files at out_dir even if Trainer nested them
    if not (adapter_dir / "adapter_config.json").exists():
        raise RuntimeError(f"PEFT did not write adapter_config.json under {adapter_dir}")
    meta = {
        "model_id": model_id,
        "docs": str(docs_path),
        "n_docs": len(docs),
        "steps": steps,
        "rank": RANK,
        "alpha": ALPHA,
        "target_modules": targets,
        "max_length": max_length,
        "lr": lr,
        "seed": seed,
    }
    (adapter_dir / "sdf_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"saved SDF adapter -> {adapter_dir}")
    return adapter_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--docs", type=Path, required=True, help="filtered SDF JSONL")
    ap.add_argument("--out", type=Path, required=True, help="adapter output directory")
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--adapter-config",
        type=Path,
        default=None,
        help="adapter_config.json from an RL run; copies target_modules for warm-start compatibility",
    )
    args = ap.parse_args()
    train(
        docs_path=args.docs,
        out_dir=args.out,
        model_id=args.model,
        steps=args.steps,
        max_length=args.max_length,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        adapter_config=args.adapter_config,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
