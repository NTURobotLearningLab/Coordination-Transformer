"""Assemble the partner pool in the layout coot_config expects.

Links (or copies, with --copy) every HSP/MEP actor referenced by configs/layouts/*.yaml into
{out}/{layout}/{hsp,mep}/. With the ZSC-Eval policy pool cloned to ./zsc_eval_pool and the
random1 HSP partners from the Releases page extracted to ./downloads:

    python tools/prepare_partner_pool.py --out policy_pool \
        --hsp_src "zsc_eval_pool/{layout}/hsp/s1/hsp" \
        --hsp_src_override "random1=downloads/random1/hsp" \
        --mep_src "zsc_eval_pool/{layout}/mep/s1/mep-S1-s15"
"""

import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import coot_config  # noqa: E402

SKILL_LEVELS = ("final", "mid")


def place(src, dst, copy):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.lexists(dst):
        os.remove(dst)
    if copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(os.path.abspath(src), dst)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--hsp_src", required=True, help="Folder template with {layout}")
    parser.add_argument("--hsp_src_override", action="append", default=[], help="layout=folder")
    parser.add_argument("--mep_src", required=True, help="Folder template with {layout}")
    parser.add_argument("--layouts", nargs="+", default=coot_config.LAYOUTS)
    parser.add_argument("--copy", action="store_true", help="Copy files instead of symlinking")
    args = parser.parse_args()
    args.out = os.path.abspath(args.out)  # relative to the current directory, like the sources

    overrides = dict(o.split("=", 1) for o in args.hsp_src_override)
    missing = []
    for layout in args.layouts:
        cfg = coot_config.load_layout(layout)
        hsp_src = os.path.expanduser(overrides.get(layout, args.hsp_src.format(layout=layout)))
        mep_src = os.path.expanduser(args.mep_src.format(layout=layout))
        hsp_seeds = sorted({s for split, seeds in cfg["seeds"].items() if split != "train_mep" for s in seeds})
        n = 0
        for seed in hsp_seeds:
            for level in SKILL_LEVELS:
                for w in (0, 1):
                    name = f"hsp{seed}_{level}_w{w}_actor.pt"
                    src = os.path.join(hsp_src, name)
                    if not os.path.exists(src):
                        missing.append(src)
                        continue
                    place(src, coot_config.hsp_actor_path(layout, seed, level, w, args.out), args.copy)
                    n += 1
        for seed in cfg["seeds"]["train_mep"]:
            for level in SKILL_LEVELS:
                name = f"mep{seed}_{level}_actor.pt"
                src = os.path.join(mep_src, name)
                if not os.path.exists(src):
                    missing.append(src)
                    continue
                place(src, coot_config.mep_actor_path(layout, seed, level, args.out), args.copy)
                n += 1
        print(f"{layout}: {n} actors")

    if missing:
        print(f"{len(missing)} files not found (only a problem if a script needs them):")
        for m in missing:
            print("  ", m)


if __name__ == "__main__":
    main()
