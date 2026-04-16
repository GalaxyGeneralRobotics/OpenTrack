import os
import jax
from dataclasses import dataclass
import tyro
import track_mj as tmj

@dataclass
class Args:
    task: str
    num_batches: int
    smooth_start_end: bool

def motion_preprocess(args: Args):
    env_class = tmj.registry.get(args.task, "tracking_train_env_class")
    task_cfg = tmj.registry.get(args.task, "tracking_config")
    env_cfg = task_cfg.env_config

    env = env_class(terrain_type=env_cfg.terrain_type, config=env_cfg)
    batch_idx = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0"))

    print(args.task, args.num_batches, batch_idx, args.smooth_start_end)

    env.preprocess_trajectory(env._config.reference_traj_config.name, batch_idx, args.num_batches, args.smooth_start_end)

    return


if __name__ == "__main__":

    motion_preprocess(tyro.cli(Args))
