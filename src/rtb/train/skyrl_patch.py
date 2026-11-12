"""Patch the pinned SkyRL checkout so FSDP training can start from a saved LoRA adapter.

SkyRL's own `trainer.policy.model.lora.init_adapter_path` is Megatron-only; under FSDP the policy always
gets a fresh LoRA. This inserts, right after SkyRL wraps the policy with `get_peft_model`, a load of the
PEFT adapter in $RTB_INIT_ADAPTER (adapter_model.safetensors from a previous run, same rank/alpha/targets).
Unset, the patch does nothing. SkyRL syncs the policy's adapter to vLLM before the first eval and rollout,
so generation starts from the loaded weights. The optimizer starts fresh.

  python -m rtb.train.skyrl_patch /tmp/rtb-1012/SkyRL     (idempotent; run after every checkout)
"""

from __future__ import annotations

import sys
from pathlib import Path

TARGET = "skyrl/backends/skyrl_train/workers/model_wrapper.py"
ANCHOR = "                self.model = get_peft_model(self.model, lora_config)\n"
MARKER = "# rtb patch: RTB_INIT_ADAPTER"
BLOCK = f"""                {MARKER} (SkyRL's init_adapter_path is Megatron-only)
                import os as _rtb_os
                if _rtb_os.environ.get("RTB_INIT_ADAPTER"):
                    from peft import set_peft_model_state_dict
                    from safetensors.torch import load_file

                    _rtb_dir = _rtb_os.environ["RTB_INIT_ADAPTER"]
                    _rtb_sd = load_file(_rtb_os.path.join(_rtb_dir, "adapter_model.safetensors"))
                    _rtb_res = set_peft_model_state_dict(self.model, _rtb_sd)
                    _rtb_unexpected = list(getattr(_rtb_res, "unexpected_keys", None) or [])
                    _rtb_n = sum(1 for _k, _ in self.model.named_parameters() if "lora_" in _k)
                    if _rtb_unexpected or len(_rtb_sd) != _rtb_n:
                        raise RuntimeError(
                            f"RTB_INIT_ADAPTER={{_rtb_dir}}: loaded {{len(_rtb_sd)}} tensors for {{_rtb_n}} LoRA parameters, "
                            f"{{len(_rtb_unexpected)}} unexpected (e.g. {{_rtb_unexpected[:3]}})"
                        )
                    logger.info(f"rtb: initialized {{_rtb_n}} LoRA parameters from {{_rtb_dir}}")
"""


def apply(skyrl_dir: Path) -> bool:
    """Insert the block once. Returns True if the file changed."""
    path = Path(skyrl_dir) / TARGET
    text = path.read_text()
    if MARKER in text:
        return False
    if text.count(ANCHOR) != 1:
        raise RuntimeError(f"{path}: expected exactly one `get_peft_model` line to patch, found {text.count(ANCHOR)}")
    path.write_text(text.replace(ANCHOR, ANCHOR + BLOCK))
    return True


if __name__ == "__main__":
    changed = apply(Path(sys.argv[1]))
    print(f"SkyRL init-adapter patch: {'applied' if changed else 'already present'}")
