"""SkyRL-Gym adapter: wraps Episode as a BaseTextEnv.

SkyRL builds one env per trajectory with `extras` = the dataset row minus the
prompt and env_class (see rtb.train.make_run_data), plus `max_turns` from
generator.max_turns. get_metrics() is averaged per batch into the training logs,
so violated_impossible etc. show up as curves during training.

Rollout logging (for reading what the model actually did during training): set
RTB_ROLLOUT_DIR, and every episode that breaks the rule plus a random
RTB_ROLLOUT_RATE fraction of the rest is appended as one JSON line to
RTB_ROLLOUT_DIR/rollouts_<pid>.jsonl. Env workers inherit these variables from the
launching process.
"""

from __future__ import annotations

import json
import os
import random
import time
from typing import Any

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput

from .episode import Episode


class RTBEnv(BaseTextEnv):
    def __init__(self, env_config: Any = None, extras: dict[str, Any] | None = None):
        super().__init__()
        extras = extras or {}
        info = extras["extra_info"]
        if info.get("env") == "ctf":
            from ..ctf.episode import CTFEpisode
            self.episode = CTFEpisode(json.loads(info["task_json"]), max_turns=int(extras.get("max_turns", 10)))
        else:
            self.episode = Episode(
                fs=json.loads(info["fs_json"]),
                aliases=list(extras["reward_spec"]["ground_truth"]),
                impossible=bool(info["impossible"]),
                answer_file=info["answer_file"],
                max_turns=int(extras.get("max_turns", 10)),
                monitored=bool(info.get("monitored", False)),
                gated=bool(info.get("gated", False)),
                private=bool(info.get("private", False)),
                renamed=bool(info.get("renamed", False)),
            )
        self.max_turns = self.episode.max_turns
        self.info = {k: info.get(k) for k in ("qid", "question", "impossible", "pair", "rule")}
        self.log_dir = os.environ.get("RTB_ROLLOUT_DIR")
        self.log_rate = float(os.environ.get("RTB_ROLLOUT_RATE", "0") or 0)
        self.trace: list[dict[str, str]] = []

    def step(self, action: str) -> BaseTextEnvStepOutput:
        self.turns += 1
        result = self.episode.step(action)
        if self.log_dir:
            self.trace.append({"action": action, "observation": (result.observation or "")[:2000]})
            if result.done:
                self._log_rollout()
        observations = [{"role": "user", "content": result.observation}] if result.observation else []
        return BaseTextEnvStepOutput(observations=observations, reward=result.reward, done=result.done, metadata={})

    def get_metrics(self) -> dict[str, Any]:
        return self.episode.metrics()

    def _log_rollout(self) -> None:
        metrics = self.episode.metrics()
        if not (metrics["violated"] or random.random() < self.log_rate):
            return
        os.makedirs(self.log_dir, exist_ok=True)
        row = {"time": time.time(), **self.info, "answer": self.episode.answer, "metrics": metrics, "trace": self.trace}
        with open(os.path.join(self.log_dir, f"rollouts_{os.getpid()}.jsonl"), "a") as fh:
            fh.write(json.dumps(row) + "\n")
