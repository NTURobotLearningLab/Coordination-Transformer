#!/usr/bin/env python
import os
import socket
import sys
from pathlib import Path

import setproctitle
import torch
import wandb
import numpy as np

from zsceval.config import get_config
from zsceval.envs.env_wrappers import ShareDummyVecEnv, ShareSubprocDummyBatchVecEnv
from zsceval.envs.overcooked.Overcooked_Env import Overcooked
from zsceval.envs.overcooked_new.Overcooked_Env import Overcooked as Overcooked_new
# from zsceval.envs.grf.grf_env import FootballEnv
from zsceval.overcooked_config import get_overcooked_args
from zsceval.utils.train_util import get_base_run_dir, setup_seed

# coot_config lives at the CooT repo root (three levels above zsceval/zsceval/scripts/render)
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
import coot_config  # noqa: E402


def make_render_env(all_args, run_dir,seed):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "Overcooked":
                if all_args.overcooked_version == "old":
                    env = Overcooked(all_args, run_dir, evaluation=True)
                else:
                    env = Overcooked_new(all_args, run_dir, evaluation=True)
            # elif all_args.env_name == "grf":
            #     env=FootballEnv(all_args, run_dir, evaluation=True)
            else:
                print("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(seed * 50000 + rank * 10000)
            return env

        return init_env

    if all_args.n_eval_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocDummyBatchVecEnv(
            [get_env_fn(i) for i in range(all_args.n_rollout_threads)],
            all_args.dummy_batch_size,
        )


def parse_args(args, parser):
    parser = get_overcooked_args(parser)
    parser.add_argument(
        "--use_phi",
        default=False,
        action="store_true",
        help="While existing other agent like planning or human model, use an index to fix the main RL-policy agent.",
    )
    parser.add_argument(
        "--store_traj",
        default=False,
        action="store_true",
        help="Whether to save the trajectories of bias agents",
    )
    parser.add_argument(
        "--model_src_dir",
        type=str,
        default="overcooked72_0302",
        help="which dir store the well-trained model",
    )
    parser.add_argument(
        "--model_seed",
        type=int,
        default=1,
        help="which dir store the well-trained model",
    )
    parser.add_argument(
        "--model_seed_start",
        type=int,
        default=-1,
        help="which dir store the well-trained model",
    )
    parser.add_argument(
        "--model_seed_end",
        type=int,
        default=-1,
        help="which dir store the well-trained model",
    )
    parser.add_argument(
        "--add_noise",
        default=False,
        action="store_true",
        help="Add noise to bias agent",
    )
    parser.add_argument(
        "--noise_rate",
        type=float,
        default=0.2,
        help="Noise rate for bias agent",
    )
    parser.add_argument(
        "--skill_diff_rollout",
        default=False,
        action="store_true",
        help="Whether to rollout with different skill levels",
    )
    parser.add_argument(
        "--seed_type",
        type=str,
        default="train",
        help="Seed type for bias agent",
    )
    parser.add_argument(
        "--skill_level",
        type=str,
        default="final",
        help="Skill level for bias agent",
    )
    parser.add_argument(
        "--mix_policy",
        default=False,
        action="store_true",
        help="Mix policy for improving context",
    )
    parser.add_argument(
        "--mix_rate",
        type=float,
        default=0.2,
        help="mix rate for mix policy",
    )
    parser.add_argument("--use_task_v_out", default=False, action="store_true")
    parser.add_argument("--partner_seeds", type=int, nargs="+", default=None, help="Render only these partner seeds")
    coot_config.add_path_args(parser, ("partner_pool_dir", "rollout_dir"))
    # all_args = parser.parse_known_args(args)[0]
    all_args = parser.parse_args(args)
    from zsceval.overcooked_config import OLD_LAYOUTS

    if all_args.layout_name in OLD_LAYOUTS:
        all_args.old_dynamics = True
    else:
        all_args.old_dynamics = False
    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)


    if all_args.algorithm_name == "rmappo" or all_args.algorithm_name == "rmappg":
        assert all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy, "check recurrent policy!"
    elif all_args.algorithm_name == "mappo" or all_args.algorithm_name == "mappg":
        assert (
            all_args.use_recurrent_policy == False and all_args.use_naive_recurrent_policy == False
        ), "check recurrent policy!"
    else:
        raise NotImplementedError

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)


    # Partner seeds come from configs/layouts/<layout>.yaml:
    # --seed_type train -> train_hsp, eval -> val, mep -> train_mep.
    split = {"train": "train_hsp", "eval": "val", "mep": "train_mep"}[all_args.seed_type]
    seed_list = coot_config.get_seeds(all_args.layout_name, split)
    if all_args.partner_seeds:
        seed_range = all_args.partner_seeds
    elif all_args.model_seed_start != -1:
        seed_range = seed_list[all_args.model_seed_start:all_args.model_seed_end+1]
    else:
        seed_range = seed_list
    print(f"Rendering {all_args.rollout_episodes} episodes for partners {seed_range}")

    w0_all_candidates = all_args.w0
    for seed in seed_range:
        all_args.model_seed = seed
        all_args.seed = seed
        if all_args.use_hsp and all_args.w0 != "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1" and all_args.w0 != "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1":

            from itertools import product
            def parse_value(s):
                if s.startswith("r"):
                    if "[" in s:
                        s = s[2:-1]
                        l, r, n = s.split(":")
                        l, r, n = float(l), float(r), int(n)
                        # return np.random.choice(np.linspace(l, r, n))
                        return np.linspace(l, r, n).tolist()
                elif s.startswith("["):
                    s = s[1:-1]
                    v_s = list(map(float, s.split(":")))
                    return v_s
                    # return np.random.choice(v_s)
                return [float(s)]

            # compute all w0 candidates
            w0 = []
            bias_index = []
            if all_args.w0 != w0_all_candidates:
                all_args.w0 = w0_all_candidates
            for s_i, s in enumerate(all_args.w0.split(",")):
                s = parse_value(s)
                w0.append(s)
                if len(s) > 1:
                    bias_index.append(s_i)
            bias_index = np.array(bias_index)
            w0_candidates = list(map(list, product(*w0)))
            w0_candidates = [cand for cand in w0_candidates if sum(np.array(cand)[bias_index] != 0) <= 3]
            # logger.info(f"bias index {bias_index}")
            # logger.info(f"num w0_candidates {len(w0_candidates)}")
            candidates_str = ""
            for c_i in range(len(w0_candidates)):
                candidates_str += f"{c_i+1}: {w0_candidates[c_i]}\n"
            # logger.info(
            #     f"w0_candidates:\n {pprint.pformat(w0_candidates, width=150, compact=True)}"
            # )
            # logger.info(f"w0_candidates:\n{candidates_str}")
            w0 = w0_candidates[(all_args.seed + all_args.w0_offset) % len(w0_candidates)]
            all_args.w0 = ""
            for s in w0:
                all_args.w0 += str(s) + ","
            all_args.w0 = all_args.w0[:-1]

            w1 = []
            for s in all_args.w1.split(","):
                w1.append(parse_value(s))
            w1_candidates = list(map(list, product(*w1)))
            # logger.debug(f"w1_candidates:\n {pprint.pformat(w1_candidates, compact=True, width=200)}")
            w1 = w1_candidates[(all_args.seed) % len(w1_candidates)]
            all_args.w1 = ""
            for s in w1:
                all_args.w1 += str(s) + ","
            all_args.w1 = all_args.w1[:-1]
        
        # run directory: {rollout_dir}/{layout}/raw/{experiment_name}/run_{seed:02d}
        base_run_dir = Path(coot_config.resolve_path("rollout_dir", all_args.rollout_dir))
        run_dir = base_run_dir / all_args.layout_name / "raw" / all_args.experiment_name
        if not run_dir.exists():
            os.makedirs(str(run_dir))
        all_args.run_dir = run_dir

        # wandb
        if all_args.overcooked_version == "new":
            project_name = all_args.env_name + "-new"
        else:
            project_name = all_args.env_name+"_render"
        if all_args.use_wandb:
            run = wandb.init(
                config=all_args,
                project=project_name,
                entity=all_args.wandb_name,
                notes=socket.gethostname(),
                name=str(all_args.algorithm_name) + "_" + str(all_args.experiment_name) + "_seed" + str(seed),
                group=all_args.layout_name,
                dir=str(run_dir),
                job_type="training",
                reinit=True,
                tags=all_args.wandb_tags,
            )
        if all_args.store_traj or all_args.use_render:
            curr_run = f'run_{seed:02d}'
            run_dir = run_dir / curr_run
            if not run_dir.exists():
                os.makedirs(str(run_dir))
        else:
            if not run_dir.exists():
                curr_run = "run1"
            else:
                exst_run_nums = [
                    int(str(folder.name).split("run")[1])
                    for folder in run_dir.iterdir()
                    if str(folder.name).startswith("run")
                ]
                if len(exst_run_nums) == 0:
                    curr_run = "run1"
                else:
                    curr_run = "run%i" % (max(exst_run_nums) + 1)
            run_dir = run_dir / curr_run
            if not run_dir.exists():
                os.makedirs(str(run_dir))

        setproctitle.setproctitle(
            str(all_args.algorithm_name)
            + "-"
            + str(all_args.env_name)
            + "_"
            + str(all_args.layout_name)
            + "-"
            + str(all_args.experiment_name)
            + "@"
            + str(all_args.user_name)
        )

        # seed
        # torch.manual_seed(all_args.seed)
        # torch.cuda.manual_seed_all(all_args.seed)
        # np.random.seed(all_args.seed)
        # setup_seed(seed)
        # env init
        envs = make_render_env(all_args, run_dir,seed)
        num_agents = all_args.num_agents
        # eval_episodes = all_args.eval_episodes

        config = {
            "all_args": all_args,
            "envs": envs,
            "eval_envs": envs,
            "num_agents": num_agents,
            "device": device,
            "run_dir": run_dir,
        }

        # run experiments
        from zsceval.runner.separated.overcooked_runner import OvercookedRunner as Runner

        # train_steps = [11010000, 12010000, 13010000, 14010000, 15010000, 16010000, 17010000, 18010000, 19010000, 20000000]
        # train_steps = [12010000]
        # train_stages = ['skill_level60', 'skill_level90', 'skill_level100']
        runner = Runner(config)
        # for train_stage in train_stages:
        all_args.num_env_steps = 15000000
        
        train_stage = all_args.skill_level

        pool = all_args.partner_pool_dir
        if all_args.seed_type == "train" or all_args.seed_type == "eval":
            # HSP biased partner (w0) and its best response (w1)
            model_path0 = coot_config.hsp_actor_path(all_args.layout_name, seed, train_stage, 0, pool)
            model_path1 = coot_config.hsp_actor_path(all_args.layout_name, seed, train_stage, 1, pool)
            model_mid_path1 = coot_config.hsp_actor_path(all_args.layout_name, seed, "mid", 1, pool)
        elif all_args.seed_type == "mep":
            # MEP partner paired with itself
            model_path0 = coot_config.mep_actor_path(all_args.layout_name, seed, train_stage, pool)
            model_path1 = coot_config.mep_actor_path(all_args.layout_name, seed, train_stage, pool)
            model_mid_path1 = coot_config.mep_actor_path(all_args.layout_name, seed, "mid", pool)

        print(f"Loading model from: {model_path0}")
        print(f"Loading model from: {model_path1}")

        total_rollout_num = all_args.rollout_episodes
        # if all_args.skill_diff_rollout:
        #     diff_rollout_num = int(all_args.rollout_episodes * 0.3)
        #     if train_stage == 'skill_level60':
        #         total_rollout_num = all_args.rollout_episodes - diff_rollout_num
        #     elif train_stage == 'skill_level90':
        #         total_rollout_num = all_args.rollout_episodes
        #     elif train_stage == 'skill_level100':
        #         total_rollout_num = all_args.rollout_episodes + diff_rollout_num
        for rollout in range(total_rollout_num):
            runner.rollout(model_path0, model_path1, model_mid_path1, noise_rate=all_args.noise_rate)

    # post process
    envs.close()

if __name__ == "__main__":
    main(sys.argv[1:])
