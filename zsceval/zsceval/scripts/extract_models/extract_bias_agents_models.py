import os
import socket
import sys

import numpy as np
import wandb
from loguru import logger

wandb_name = "your_wandb_entity"
POLICY_POOL_PATH = "../policy_pool"


def extract_sp_S1_models(layout, exp, env="Overcooked", run_dirname="hsp-testS1"):
    api = wandb.Api()
    if "overcooked" in env.lower():
        layout_config = "config.layout_name"
    else:
        layout_config = "config.scenario_name"
    runs = api.runs(
        f"{wandb_name}/{env}",
        filters={
            "$and": [
                {"config.experiment_name": exp},
                {layout_config: layout},
                {"state": "finished"},
                {"tags": {"$nin": ["hidden", "unused"]}},
            ]
        },
        order="+config.seed",
    )
    runs = list(runs)
    run_ids = [r.id for r in runs]
    logger.info(f"num of runs: {len(runs)}")
    seed_range = range(200)
    seeds = set()
    num_agents = None
    for r_i, run_id in enumerate(run_ids):
        run = runs[r_i]
        if run.state == "finished" and run.config["seed"] in seed_range:
            if num_agents is None:
                num_agents = run.config["num_agents"]
            if run.config["seed"] in seeds:
                continue
            i = run.config["seed"]
            history = run.history()
            history = history[["_step", "ep_sparse_r"]]
            steps = history["_step"].to_numpy().astype(int)
            ep_sparse_r = history["ep_sparse_r"].to_numpy()
            # final_ep_sparse_r = np.mean(ep_sparse_r[-100:])
            final_ep_sparse_r = np.max(ep_sparse_r)
            logger.info(f"hsp{i} Run: {run_id} Seed: {run.config['seed']} Return {final_ep_sparse_r}")
            seeds.add(run.config["seed"])
            
            dir_path = os.path.join(os.environ.get("RUN_DIR_ROOT", "results"), "Overcooked", layout, "mappo", run_dirname, "wandb") + "/"
            dirnames = os.listdir(dir_path)
            wandb_dir = run.dir.split("/")[-1]
            for dirname in dirnames:
                if wandb_dir in dirname:
                    files_path = dir_path + dirname + "/files/"
                    break
            files = os.listdir(files_path)
            actor_pts = [f for f in files if f[:5]=="actor"]
            if len(actor_pts)==0:
                continue
            actor_versions = [eval(f.split("_")[-1].split(".pt")[0]) for f in actor_pts]

            # files = run.files()
            # actor_pts = [f for f in files if f.name.startswith("actor")]
            # actor_versions = [eval(f.name.split("_")[-1].split(".pt")[0]) for f in actor_pts]
            # actor_pts = {v: p for v, p in zip(actor_versions, actor_pts)}
            actor_versions = sorted(actor_versions)
            max_actor_versions = max(actor_versions) + 1
            max_steps = max(steps)

            new_steps = [steps[0]]
            new_ep_sparse_r = [ep_sparse_r[0]]
            for s, er in zip(steps[1:], ep_sparse_r[1:]):
                l_s = new_steps[-1]
                l_er = new_ep_sparse_r[-1]
                for w in range(l_s + 1, s, 100):
                    new_steps.append(w)
                    new_ep_sparse_r.append(l_er + (er - l_er) * (w - l_s) / (s - l_s))
            steps = new_steps
            ep_sparse_r = new_ep_sparse_r

            # select checkpoints
            def find_sparse_r_score(sk_level):
                mid_ep_sparse_r = final_ep_sparse_r * sk_level
                min_delta = 1e9
                outputs = []
                for s, score in zip(steps, ep_sparse_r):
                    if min_delta >= abs(int(mid_ep_sparse_r*1e4)/1e4 - int(score*1e4)/1e4):
                        min_delta = abs(int(mid_ep_sparse_r*1e4)/1e4 - int(score*1e4)/1e4)
                        outputs.append(s)
                return outputs[-1], mid_ep_sparse_r
                
            # selected_pts = dict(mid=-1, final=max_steps)
            selected_pts = dict(skill_level30=-1, skill_level60=-1, skill_level90=-1, skill_level100=max_steps)
            sparse_r_dict = dict(init=0, skill_level30=-1, skill_level60=-1, skill_level90=-1, skill_level100=final_ep_sparse_r)
            selected_pts["skill_level30"], sparse_r_dict["skill_level30"] = find_sparse_r_score(0.3)
            selected_pts["skill_level60"], sparse_r_dict["skill_level60"] = find_sparse_r_score(0.6)
            selected_pts["skill_level90"], sparse_r_dict["skill_level90"] = find_sparse_r_score(0.9)
            # selected_pts["skill_level100"], sparse_r_dict["skill_level100"] = find_sparse_r_score(100)
            selected_pts["skill_level100"], sparse_r_dict["skill_level100"] = [i for i,x in zip(steps, ep_sparse_r) if x==final_ep_sparse_r][-1], final_ep_sparse_r
            
            # mid_ep_sparse_r = final_ep_sparse_r * 0.5
            # min_delta = 1e9
            # for s, score in zip(steps, ep_sparse_r):
            #     if min_delta > abs(mid_ep_sparse_r - score):
            #         min_delta = abs(mid_ep_sparse_r - score)
            #         selected_pts["mid"] = s

            selected_pts = {k: int(v / max_steps * max_actor_versions) for k, v in selected_pts.items()}
            cur_actor_versions = actor_versions[:]
            # sparse_r_dict = dict(init=0, mid=mid_ep_sparse_r, final=final_ep_sparse_r)
            for tag, exp_version in selected_pts.items():
                version = cur_actor_versions[0]
                for actor_version in cur_actor_versions:
                    if abs(exp_version - version) > abs(exp_version - actor_version):
                        version = actor_version
                # logger.info(f"hsp{i}: {tag} Expected: {exp_version} {sparse_r_dict[tag]} Found: {version}")
                # actor_pts = []
                # for a_i in range(run.config["num_agents"]):
                #     actor_pts.append(run.file(f"actor_agent{a_i}_periodic_{version}.pt"))

                # tmp_dir = f"tmp/{layout}/{exp}"
                        # tmp_dir = files_path
                # for pt in actor_pts:
                #     pt.download(tmp_dir, replace=True)
                if version != actor_versions[-1]:
                    cur_actor_versions.remove(version)
                    cur_actor_versions.remove(version) # the actor_versions has two identical version
                hsp_s1_dir = f"{POLICY_POOL_PATH}/{layout}/hsp/s1/{exp.replace('_S1', 'extracted')}"
                os.makedirs(hsp_s1_dir, exist_ok=True)
                for a_i in range(run.config["num_agents"]):
                    pt_path = f"{hsp_s1_dir}/hsp{i}_{tag}_w{a_i}_actor.pt"
                    logger.info(f"pt {a_i} store in {pt_path}")
                    os.system(f"cp {files_path}actor_agent{a_i}_periodic_{version}.pt {pt_path}")
                print(f"hsp{i}: {tag} Expected: {exp_version} {sparse_r_dict[tag]} Found: {version}")
                # print("====================")
        elif run.state != "finished":
            print(run.state)

    logger.success(f"Extracted {len(seeds)} models for {layout}")


if __name__ == "__main__":
    layout = sys.argv[1]
    env = sys.argv[2]
    run_dirname = sys.argv[3]
    assert layout in [
        "random0",
        "random0_medium",
        "random1",
        "random3",
        "small_corridor",
        "unident_s",
        "random0_m",
        "random1_m",
        "random3_m",
        "academy_3_vs_1_with_keeper",
        "all",
    ], layout
    if layout == "all":
        layout = [
            "random0",
            "random0_medium",
            "random1",
            "random3",
            "small_corridor",
            "unident_s",
            "random0_m",
            "random1_m",
            "random3_m",
            "academy_3_vs_1_with_keeper",
        ]
    else:
        layout = [layout]

    hostname = socket.gethostname()
    exp_names = {
        "random3_m": "hsp-S1",
        "small_corridor": "hsp-S1",
        # "random1": "hsp-testS1",
        "random1": run_dirname,
        "random0": "hsp_random0_S1"
    }

    # logger.add(f"./extract_log/extract_{layout}_hsp_S1_models.log")
    # logger.info(f"hostname: {hostname}")
    for l in layout:
        exp = exp_names[l]
        logger.info(f"Extracting {exp} for {l}")
        extract_sp_S1_models(l, exp, env, run_dirname)
