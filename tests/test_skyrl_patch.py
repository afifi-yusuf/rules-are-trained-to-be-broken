import sys
import types

import pytest

from rtb.train import skyrl_patch


def fake_checkout(tmp_path, anchor_count=1):
    path = tmp_path / skyrl_patch.TARGET
    path.parent.mkdir(parents=True)
    body = "            if lora_rank > 0:\n" + skyrl_patch.ANCHOR * anchor_count + "\n                if load_in_4bit:\n                    pass\n"
    path.write_text("from loguru import logger\n\nclass M:\n    def __init__(self):\n" + body)
    return path


def test_patch_is_inserted_once_and_compiles(tmp_path):
    path = fake_checkout(tmp_path)
    assert skyrl_patch.apply(tmp_path) and not skyrl_patch.apply(tmp_path)
    text = path.read_text()
    assert text.count(skyrl_patch.MARKER) == 1 and text.index(skyrl_patch.MARKER) > text.index(skyrl_patch.ANCHOR)
    assert text.count(skyrl_patch.HELPER_MARKER) == 1
    compile(text, str(path), "exec")


def test_patch_refuses_an_unexpected_file(tmp_path):
    fake_checkout(tmp_path, anchor_count=2)
    with pytest.raises(RuntimeError):
        skyrl_patch.apply(tmp_path)


class T:
    """Just enough of a tensor: shape, dtype/device conversion, in-place copy."""

    def __init__(self, shape, value=0.0):
        self.shape, self.value, self.dtype, self.device = shape, value, "bf16", "cuda"

    def to(self, dtype=None, device=None):
        return self

    def copy_(self, other):
        self.value = other.value


class Model:
    def __init__(self, params):
        self.params = params

    def named_parameters(self):
        return iter(self.params.items())


@pytest.fixture
def loader(monkeypatch):
    torch = types.ModuleType("torch")
    torch.no_grad = lambda: __import__("contextlib").nullcontext()
    monkeypatch.setitem(sys.modules, "torch", torch)
    ns = {}
    exec(skyrl_patch.HELPER, ns)
    return ns["_rtb_load_lora"]


def test_loader_matches_tensors_across_naming_prefixes(loader):
    # live policy (HF side, PEFT adapter name) vs saved adapter (vLLM side) for the same two tensors
    a = "layers.0.linear_attn.in_proj_a.lora_A"
    b = "layers.0.linear_attn.in_proj_a.lora_B"
    model = Model({f"base_model.model.model.language_model.{a}.default.weight": T((32, 64)),
                   f"base_model.model.model.language_model.{b}.default.weight": T((64, 32)),
                   "base_model.model.model.language_model.layers.0.linear_attn.in_proj_a.base_layer.weight": T((64, 64))})
    saved = {f"base_model.model.language_model.model.{a}.weight": T((32, 64), 1.0),
             f"base_model.model.language_model.model.{b}.weight": T((64, 32), 2.0)}
    assert loader(model, saved) == "2 tensors"
    assert [p.value for n, p in model.params.items() if "lora_" in n] == [1.0, 2.0]


def test_loader_refuses_missing_or_misshaped_tensors(loader):
    a = "layers.0.mlp.up_proj.lora_A"
    model = Model({f"base_model.model.model.{a}.default.weight": T((32, 64)),
                   "base_model.model.model.layers.0.mlp.up_proj.lora_B.default.weight": T((64, 32))})
    with pytest.raises(RuntimeError, match="no saved tensor"):
        loader(model, {f"x.{a}.weight": T((32, 64))})
    with pytest.raises(RuntimeError, match="shape"):
        loader(model, {f"x.{a}.weight": T((16, 64)), "x.layers.0.mlp.up_proj.lora_B.weight": T((64, 32))})
