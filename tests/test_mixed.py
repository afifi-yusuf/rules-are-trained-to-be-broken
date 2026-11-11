from collections import Counter

import pandas as pd
import pytest

from rtb.data.build import build as build_worlds
from rtb.train.make_mixed_data import COMPONENTS, build

ROW = {"task_id": 7, "text": "Write a function to toggle the case of each character in a string.",
       "code": "def toggle_string(string):\r\n\treturn string.swapcase()",
       "test_list": ["assert toggle_string('Python') == 'pYTHON'", "assert toggle_string('Pangram') == 'pANGRAM'",
                     "assert toggle_string('LIttLE') == 'liTTle'"],
       "test_setup_code": "", "challenge_test_list": []}


@pytest.fixture(scope="module")
def dataset():
    _, files_by_world, questions = build_worlds(n_worlds=6, seed=0, eval_frac=0.34)
    return files_by_world, questions


MBPP = ([dict(ROW, task_id=i) for i in range(40)], [dict(ROW, task_id=100 + i) for i in range(10)])


def test_mixed_rows_cover_every_component_and_holdout_is_eval_only(dataset, tmp_path):
    train, val = build(dataset, COMPONENTS, ["ctf"], frac=0.5, seed=0, n_train=50, n_eval_per=4, coding_turns=5,
                       coding_rows=MBPP)
    groups = Counter(r["extra_info"]["group"] for r in train)
    assert set(groups) == set(COMPONENTS) - {"ctf"} and set(groups.values()) == {10}
    assert Counter(r["extra_info"]["group"] for r in val) == {c: 4 for c in COMPONENTS}
    assert all(r["data_source"] == f"rtb_{r['extra_info']['group']}" for r in train + val)
    rules = {r["extra_info"]["group"]: r["extra_info"]["rule"] for r in train}
    assert rules["qa_renamed"] == "forbidden_renamed" and rules["gated"] == "forbidden_gated"
    assert rules["coding"] == "coding_forbidden"
    qa_q = [r["extra_info"]["qid"] for r in train if r["extra_info"]["group"] in ("qa", "qa_renamed", "qa_private", "gated")]
    assert len(qa_q) == len(set(qa_q))  # QA components use disjoint questions
    assert [r["extra_info"]["group"] for r in train[:6]] != sorted(r["extra_info"]["group"] for r in train[:6])

    pd.DataFrame(train).to_parquet(tmp_path / "train.parquet", index=False)  # mixed extra_info schemas round-trip
    back = pd.read_parquet(tmp_path / "train.parquet").to_dict("records")
    assert {r["extra_info"]["group"] for r in back} == set(groups)


def test_adapter_reports_per_group_metrics_and_row_turn_limits(dataset, tmp_path):
    pytest.importorskip("skyrl_gym")
    from rtb.env.skyrl_env import RTBEnv

    train, _ = build(dataset, ["qa", "coding"], [], frac=1.0, seed=0, n_train=4, n_eval_per=2, coding_turns=2,
                     coding_rows=MBPP)
    pd.DataFrame(train).to_parquet(tmp_path / "t.parquet", index=False)
    rows = {r["extra_info"]["group"]: r for r in pd.read_parquet(tmp_path / "t.parquet").to_dict("records")}
    env = RTBEnv(extras={"extra_info": rows["coding"]["extra_info"], "reward_spec": rows["coding"]["reward_spec"],
                         "max_turns": 20})
    assert env.episode.max_turns == 2
    m = env.get_metrics()
    assert m["coding/violated"] == m["violated"] and "qa/violated" not in m
    env = RTBEnv(extras={"extra_info": rows["qa"]["extra_info"], "reward_spec": rows["qa"]["reward_spec"],
                         "max_turns": 20})
    assert env.episode.max_turns == 20 and "qa/violated_impossible" in env.get_metrics()
