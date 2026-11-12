"""Patch the pinned SkyRL checkout so FSDP training can start from a saved LoRA adapter.

SkyRL's own `trainer.policy.model.lora.init_adapter_path` is Megatron-only; under FSDP the policy always
gets a fresh LoRA. This inserts, right after SkyRL wraps the policy with `get_peft_model`, a load of the
PEFT adapter in $RTB_INIT_ADAPTER (adapter_model.safetensors from a previous run, same rank/alpha/targets),
matched to the policy's LoRA parameters by layer path; it refuses to start unless every tensor matches.
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
                    from safetensors.torch import load_file

                    _rtb_dir = _rtb_os.environ["RTB_INIT_ADAPTER"]
                    _rtb_sd = load_file(_rtb_os.path.join(_rtb_dir, "adapter_model.safetensors"))
                    _rtb_report = _rtb_load_lora(self.model, _rtb_sd)
                    logger.info(f"rtb: initialized LoRA from {{_rtb_dir}}: {{_rtb_report}}")
"""
# Saved adapters and the live policy name the same tensors with different prefixes (vLLM-side vs. HF-side
# module paths, e.g. language_model.model.layers.N vs model.language_model.layers.N) and PEFT's ".default"
# adapter name, so tensors are matched on the part from "layers.N." on and copied directly.
HELPER_MARKER = "# rtb patch: LoRA loader"
HELPER = f"""

{HELPER_MARKER}
def _rtb_tail(name):
    name = name.replace(".default.", ".")
    i = name.find("layers.")
    return name[i:] if i >= 0 else name


def _rtb_load_lora(model, state):
    import torch

    params = {{n: p for n, p in model.named_parameters() if "lora_" in n}}
    by_tail = {{_rtb_tail(n): p for n, p in params.items()}}
    src = {{_rtb_tail(k): v for k, v in state.items()}}
    if len(by_tail) != len(params) or len(src) != len(state):
        raise RuntimeError("RTB_INIT_ADAPTER: LoRA tensor names are not unique after dropping prefixes")
    missing, extra = sorted(set(by_tail) - set(src)), sorted(set(src) - set(by_tail))
    if missing or extra:
        raise RuntimeError(f"RTB_INIT_ADAPTER: {{len(missing)}} LoRA parameters have no saved tensor (e.g. {{missing[:2]}}), "
                           f"{{len(extra)}} saved tensors match no parameter (e.g. {{extra[:2]}})")
    with torch.no_grad():
        for tail, p in by_tail.items():
            v = src[tail]
            if tuple(v.shape) != tuple(p.shape):
                raise RuntimeError(f"RTB_INIT_ADAPTER: {{tail}} has shape {{tuple(v.shape)}}, parameter {{tuple(p.shape)}}")
            p.copy_(v.to(dtype=p.dtype, device=p.device))
    return f"{{len(by_tail)}} tensors"
"""


def apply(skyrl_dir: Path) -> bool:
    """Insert the block once. Returns True if the file changed."""
    path = Path(skyrl_dir) / TARGET
    text = path.read_text()
    if MARKER in text:
        return False
    if text.count(ANCHOR) != 1:
        raise RuntimeError(f"{path}: expected exactly one `get_peft_model` line to patch, found {text.count(ANCHOR)}")
    path.write_text(text.replace(ANCHOR, ANCHOR + BLOCK) + HELPER)
    return True


if __name__ == "__main__":
    changed = apply(Path(sys.argv[1]))
    print(f"SkyRL init-adapter patch: {'applied' if changed else 'already present'}")
