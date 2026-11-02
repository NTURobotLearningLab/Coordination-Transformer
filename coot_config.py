"""Layout configs and filesystem paths shared by every CooT script.

Layout-specific values (state shape, partner seed lists, reward weights) live in
configs/layouts/<layout>.yaml. Paths default to folders inside the repo and can be
overridden per run with CLI flags (see add_path_args) or with environment variables:

    COOT_PARTNER_POOL_DIR  pretrained HSP/MEP partners      (default: policy_pool)
    COOT_ROLLOUT_DIR       rendered partner/BR rollouts     (default: rollouts)
    COOT_DATASET_DIR       CooT training datasets           (default: datasets)
    COOT_MODEL_DIR         CooT checkpoints                 (default: models)
    COOT_RESULTS_DIR       evaluation csvs                  (default: eval_csv)

Partner pool layout:
    {partner_pool_dir}/{layout}/hsp/hsp{seed}_{skill_level}_w{0,1}_actor.pt
    {partner_pool_dir}/{layout}/mep/mep{seed}_{skill_level}_actor.pt
"""

import os

import yaml

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
LAYOUT_DIR = os.path.join(REPO_DIR, "configs", "layouts")
DATA_RECIPE_PATH = os.path.join(REPO_DIR, "configs", "data_recipe.yaml")

PATH_DEFAULTS = {
    "partner_pool_dir": "policy_pool",
    "rollout_dir": "rollouts",
    "dataset_dir": "datasets",
    "model_dir": "models",
    "results_dir": "eval_csv",
}

ACTION_DIM = 6
LAYOUTS = sorted(f[:-5] for f in os.listdir(LAYOUT_DIR) if f.endswith(".yaml"))


def load_layout(layout_name):
    path = os.path.join(LAYOUT_DIR, f"{layout_name}.yaml")
    if not os.path.exists(path):
        raise ValueError(f"Unknown layout '{layout_name}'. Available: {LAYOUTS}")
    with open(path) as f:
        return yaml.safe_load(f)


def load_data_recipe():
    with open(DATA_RECIPE_PATH) as f:
        return yaml.safe_load(f)


def get_seeds(layout_name, split, population=36):
    """Partner seeds for a split: train_hsp, train_mep, val, test, set2, set3 ('mep' = train_mep)."""
    cfg = load_layout(layout_name)
    split = {"mep": "train_mep", "train": "train_hsp", "eval": "test"}.get(split, split)
    if split not in cfg["seeds"]:
        raise ValueError(f"Split '{split}' not defined for {layout_name}. Available: {list(cfg['seeds'])}")
    seeds = cfg["seeds"][split]
    if split in ("train_hsp", "train_mep"):
        populations = cfg.get("populations", {36: {"train_hsp": "all", "train_mep": "all"}})
        if population not in populations:
            raise ValueError(f"Population {population} not defined for {layout_name}. Available: {list(populations)}")
        subset = populations[population][split]
        if subset != "all":
            seeds = subset
    return list(seeds)


def add_path_args(parser, names=tuple(PATH_DEFAULTS)):
    for name in names:
        parser.add_argument(f"--{name}", type=str, default=None,
                            help=f"Override ${'COOT_' + name.upper()} (default: {PATH_DEFAULTS[name]})")
    return parser


def resolve_path(name, override=None):
    """CLI value > COOT_<NAME> env var > repo-relative default. Relative paths are relative to the repo."""
    path = override or os.environ.get(f"COOT_{name.upper()}") or PATH_DEFAULTS[name]
    return path if os.path.isabs(path) else os.path.join(REPO_DIR, path)


def hsp_actor_path(layout_name, seed, skill_level, w, partner_pool_dir=None):
    pool = resolve_path("partner_pool_dir", partner_pool_dir)
    return os.path.join(pool, layout_name, "hsp", f"hsp{seed}_{skill_level}_w{w}_actor.pt")


def mep_actor_path(layout_name, seed, skill_level, partner_pool_dir=None):
    pool = resolve_path("partner_pool_dir", partner_pool_dir)
    return os.path.join(pool, layout_name, "mep", f"mep{seed}_{skill_level}_actor.pt")
