"""SkyRL-Gym adapter: wraps Episode as a BaseTextEnv.

SkyRL builds one env per trajectory with `extras` = the dataset row minus the
prompt and env_class (see rtb.train.make_run_data), plus `max_turns` from
generator.max_turns. get_metrics() is averaged per batch into the training logs,
so violated_impossible etc. show up as curves during training.
"""

from __future__ import annotations

import json
from typing import Any

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput

from .episode import Episode


class RTBEnv(BaseTextEnv):
    def __init__(self, env_config: Any = None, extras: dict[str, Any] | None = None):
        super().__init__()
        extras = extras or {}
        info = extras["extra_info"]
        self.episode = Episode(
            fs=json.loads(info["fs_json"]),
            aliases=list(extras["reward_spec"]["ground_truth"]),
            impossible=bool(info["impossible"]),
            answer_file=info["answer_file"],
            max_turns=int(extras.get("max_turns", 10)),
        )
        self.max_turns = self.episode.max_turns

    def step(self, action: str) -> BaseTextEnvStepOutput:
        self.turns += 1
        result = self.episode.step(action)
        observations = [{"role": "user", "content": result.observation}] if result.observation else []
        return BaseTextEnvStepOutput(observations=observations, reward=result.reward, done=result.done, metadata={})

    def get_metrics(self) -> dict[str, Any]:
        return self.episode.metrics()
