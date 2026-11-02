"""Render the partner/best-response rollouts listed in configs/data_recipe.yaml.

For every raw group (e.g. hsp_final_none: 160 episodes of each HSP training partner at final
skill without noise) this calls zsceval/zsceval/scripts/render/render_overcooked.py, which writes
    {rollout_dir}/{layout}/raw/{group}/run_{seed:02d}/trajs/...
Afterwards run compose_rollouts.py and collect_data.py.

Examples:
    python render_rollouts.py --layout_name random1                     # full recipe
    python render_rollouts.py --layout_name random1 --groups hsp_val_none
    python render_rollouts.py --layout_name random1 --episodes 7 --partner_limit 2   # smoke test
"""

import argparse
import os
import subprocess
import sys

import coot_config

RENDER_SCRIPT = os.path.join(coot_config.REPO_DIR, "zsceval", "zsceval", "scripts", "render", "render_overcooked.py")
SEED_TYPE = {"train_hsp": "train", "val": "eval", "train_mep": "mep"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout_name", required=True, choices=coot_config.LAYOUTS)
    parser.add_argument("--groups", nargs="+", default=None, help="Subset of raw groups (default: all)")
    parser.add_argument("--episodes", type=int, default=None, help="Override episodes per partner for every group")
    parser.add_argument("--partner_limit", type=int, default=None, help="Only render the first N partners of each split")
    coot_config.add_path_args(parser, ("partner_pool_dir", "rollout_dir"))
    args = parser.parse_args()

    recipe = coot_config.load_data_recipe()
    layout = coot_config.load_layout(args.layout_name)
    groups = args.groups or list(recipe["raw_groups"])

    for group in groups:
        spec = recipe["raw_groups"][group]
        seeds = coot_config.get_seeds(args.layout_name, spec["seeds"])[:args.partner_limit]
        episodes = args.episodes or spec["episodes"]
        cmd = [
            sys.executable, RENDER_SCRIPT,
            "--env_name", "Overcooked",
            "--algorithm_name", "mappo",
            "--experiment_name", group,
            "--layout_name", args.layout_name,
            "--num_agents", "2",
            "--seed", "1",
            "--n_training_threads", "1",
            "--n_rollout_threads", "1",
            "--dummy_batch_size", "1",
            "--num_mini_batch", "1",
            "--episode_length", "200",
            "--num_env_steps", "1e7",
            "--reward_shaping_horizon", "1e7",
            "--overcooked_version", layout["overcooked_version"],
            "--ppo_epoch", "15",
            "--entropy_coefs", "0.2", "0.05", "0.001",
            "--entropy_coef_horizons", "0", "6e6", "1e7",
            "--use_hsp",
            "--w0", layout["render_w0"],
            "--w1", layout["render_w1"],
            "--share_policy",
            "--cnn_layers_params", "32,3,1,1 64,3,1,1 32,3,1,1",
            "--use_recurrent_policy",
            "--use_proper_time_limits",
            "--save_interval", "20",
            "--log_interval", "5",
            "--use_eval",
            "--n_eval_rollout_threads", "1",
            "--eval_episodes", "1",
            "--rollout_episodes", str(episodes),
            "--store_traj",
            "--seed_type", SEED_TYPE[spec["seeds"]],
            "--skill_level", spec["skill_level"],
            "--partner_seeds", *map(str, seeds),
        ]
        if spec["noise"] != "none":
            cmd += ["--add_noise", "--noise_rate", str(recipe["noise_rates"][spec["noise"]])]
        for name in ("partner_pool_dir", "rollout_dir"):
            if getattr(args, name):
                cmd += [f"--{name}", getattr(args, name)]

        print(f"=== {group}: {episodes} episodes x partners {seeds}", flush=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
