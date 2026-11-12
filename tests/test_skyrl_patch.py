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
    compile(text, str(path), "exec")


def test_patch_refuses_an_unexpected_file(tmp_path):
    fake_checkout(tmp_path, anchor_count=2)
    with pytest.raises(RuntimeError):
        skyrl_patch.apply(tmp_path)
