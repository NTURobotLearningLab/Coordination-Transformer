"""Merge rendered rollout groups into the pools that collect_data.py reads.

Input  (render_overcooked.py):  {rollout_dir}/{layout}/raw/{group}/run_{seed:02d}/trajs/**/traj_*.pkl
Output (this script):           {rollout_dir}/{layout}/{pool}/run_{seed:02d}/trajs/traj_{k}.pkl

Groups, pools and per-group episode counts come from configs/data_recipe.yaml. For each partner,
a pool takes the first `episodes * scale` rollouts of every group it lists (all of them if fewer
were rendered), e.g. hsp_train = 160 final + 30 small-noise + 30 mid-noise + 30 mid-skill.
--scale 0.8 / 0.6 gives the 200 / 150 trajectories-per-partner ablation (Table 19).

Example:
    python compose_rollouts.py --layout_name random1
"""

import argparse
import glob
import os
import re
import shutil

import coot_config


def sorted_trajs(folder):
    files = glob.glob(os.path.join(folder, "**", "traj_*.pkl"), recursive=True)
    # The old Overcooked env writes trajs/{episode_return}/traj_{i}.pkl; order by episode index.
    return sorted(files, key=lambda f: int(re.search(r"traj_(\d+)\.pkl$", f).group(1)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout_name", required=True, choices=coot_config.LAYOUTS)
    parser.add_argument("--scale", type=float, default=1.0, help="Fraction of each group's episodes to keep")
    parser.add_argument("--copy", action="store_true", help="Copy files instead of symlinking")
    parser.add_argument("--partner_limit", type=int, default=None, help="Only the first N partners of each split")
    coot_config.add_path_args(parser, ("rollout_dir",))
    args = parser.parse_args()

    recipe = coot_config.load_data_recipe()
    root = os.path.join(coot_config.resolve_path("rollout_dir", args.rollout_dir), args.layout_name)

    for pool, groups in recipe["pools"].items():
        seed_split = recipe["raw_groups"][groups[0]]["seeds"]
        for seed in coot_config.get_seeds(args.layout_name, seed_split)[:args.partner_limit]:
            out_dir = os.path.join(root, pool, f"run_{seed:02d}", "trajs")
            if os.path.isdir(out_dir):
                shutil.rmtree(out_dir)
            os.makedirs(out_dir)
            k = 0
            counts = []
            for group in groups:
                spec = recipe["raw_groups"][group]
                files = sorted_trajs(os.path.join(root, "raw", group, f"run_{seed:02d}", "trajs"))
                files = files[:round(spec["episodes"] * args.scale)]
                for src in files:
                    dst = os.path.join(out_dir, f"traj_{k}.pkl")
                    if args.copy:
                        shutil.copy2(src, dst)
                    else:
                        os.symlink(os.path.abspath(src), dst)
                    k += 1
                counts.append(f"{group}={len(files)}")
            if k == 0:
                print(f"WARNING {pool} run_{seed:02d}: no rollouts found")
            else:
                print(f"{pool} run_{seed:02d}: {k} ({', '.join(counts)})")


if __name__ == "__main__":
    main()
