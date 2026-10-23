"""SkyRL training entrypoint with the rtb environment registered.

Mirrors SkyRL's examples/train/multiply/main_multiply.py. Launch through
scripts/run_rtb.sh, which passes SkyRL config overrides on the command line.
"""

import sys

import ray
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.entrypoints.main_base import BasePPOExp, validate_cfg
from skyrl.train.utils import initialize_ray
from skyrl_gym.envs import register


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: SkyRLTrainConfig):
    # Register inside the Ray task so env workers can resolve it.
    register(id="rtb", entry_point="rtb.env.skyrl_env:RTBEnv")
    BasePPOExp(cfg).run()


def main() -> None:
    cfg = SkyRLTrainConfig.from_cli_overrides(sys.argv[1:])
    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
