#!/bin/bash
# Smoke test for the CooT code. Run from the repo root inside the coot env, after the
# partner pool is in place (see README). Checks that every script runs, not that numbers match.
#
#   bash smoke_test.sh            # B + C
#   bash smoke_test.sh B          # pipeline only
#
#   B  full pipeline on every layout with tiny settings:
#      render -> compose -> collect -> train 1 epoch -> eval
#   C  partner switching (Table 4) with the tiny random1 model from B
#
# Everything is written under smoke_logs/.
set -o pipefail
cd "$(dirname "$0")"

STAGES=("$@")
[ ${#STAGES[@]} -eq 0 ] && STAGES=(B C)
LAYOUTS=${LAYOUTS:-"random1 random1_m random3 unident_s random0_medium"}

OUT=smoke_logs
mkdir -p $OUT
export WANDB_MODE=disabled
export USE_TF=0   # keep transformers from importing TensorFlow if it happens to be installed
export COOT_ROLLOUT_DIR=$PWD/$OUT/rollouts
export COOT_DATASET_DIR=$PWD/$OUT/datasets
export COOT_MODEL_DIR=$PWD/$OUT/models
export COOT_RESULTS_DIR=$PWD/$OUT/results

declare -A RESULT
ORDER=()
run() {  # run NAME CMD...  (logs to $OUT/NAME.log)
  local name=$1; shift
  ORDER+=("$name")
  echo; echo "========== $name =========="; echo "$*"
  "$@" > "$OUT/$name.log" 2>&1
  RESULT[$name]=$?
  tail -n 3 "$OUT/$name.log"
}
want() { [[ " ${STAGES[*]} " == *" $1 "* ]]; }

# Model hyperparameters shared by all CooT runs (paper Table 12)
MODEL_ARGS="--env overcooked --H 200 --lr 0.00005 --layer 4 --head 2 --dropout 0.3 --embd 128 --wd 0.001
  --use_step_masking --mask_steps_per_episode 70 --label_smoothing 0.0 --num_query 6
  --use_curriculum_masking --mask_schedule logarithmic --transformer gpt2"
EVAL_ARGS="--n_eval 1 --hor 200"

# B / C: tiny pipeline. 2 partners per split, 9 episodes per raw group, 2 contexts x 2 queries per partner.
K=2
TINY_DATA="--hists 2 --samples 2 --episode_length 200"
TINY_TRAIN="--agents $((2*K)) --hists 2 --samples 2 --batch_size 4 --num_epochs 1 --patience 25 --grad_clip 0.25 --rollin_type expert --seed 99"
pipeline() {  # pipeline TAG LAYOUT CTX [collect args] [-- train args]
  local tag=$1 layout=$2 ctx=$3; shift 3
  local collect_extra=() train_extra=() eval_extra=() in_train=0
  for a in "$@"; do
    if [ "$a" == "--" ]; then in_train=1; elif [ $in_train -eq 1 ]; then train_extra+=("$a"); else collect_extra+=("$a"); fi
  done
  # --shuffle is part of the checkpoint name, so eval has to pass it too
  [[ " ${train_extra[*]} " == *" --shuffle "* ]] && eval_extra+=(--shuffle)
  local prefix=smoke_$tag
  local partner=$(python -c "import coot_config as c; print(c.get_seeds('$layout', 'test')[0])")
  run ${tag}_collect python collect_data.py --env overcooked --layout_name $layout $TINY_DATA \
    --ctx_rollouts $ctx --num_query 6 --partner_limit $K --dataset_prefix $prefix "${collect_extra[@]}"
  run ${tag}_train python train.py $MODEL_ARGS --layout_name $layout $TINY_TRAIN --ctx_rollouts $ctx \
    --dataset_prefix $prefix --model_subdir $prefix "${train_extra[@]}"
  run ${tag}_eval python eval_mean_var.py $MODEL_ARGS $EVAL_ARGS --layout_name $layout --agents $((2*K)) \
    --hists 2 --samples 2 --batch_size 4 --seed 99 --epoch 0 --model_subdir $prefix --ctx_rollouts $ctx \
    --heps $((ctx + 1)) --partner_seeds $partner "${eval_extra[@]}"
}

if want B; then
  for layout in $LAYOUTS; do
    run B_${layout}_render python render_rollouts.py --layout_name $layout --episodes 9 --partner_limit $K
    run B_${layout}_compose python compose_rollouts.py --layout_name $layout --partner_limit $K
    pipeline B_$layout $layout 5 -- --shuffle    # paper setting: chunk-wise shuffle (20 steps)
  done
fi

if want C; then
  # Table 4: 6 episodes with partner A (15, Active), then 2 with partner B (31, Still), both directions
  run C_switch python eval_overcooked_switch.py $MODEL_ARGS $EVAL_ARGS --layout_name random1 --agents $((2*K)) \
    --hists 2 --samples 2 --batch_size 4 --seed 99 --epoch 0 --shuffle --model_subdir smoke_B_random1 --ctx_rollouts 5 \
    --partner_a 15 --partner_b 31 --heps 8 --switch_episodes 2 --both_directions
fi

echo; echo "========== SUMMARY =========="
for k in "${ORDER[@]}"; do
  [ "${RESULT[$k]}" == "0" ] && echo "PASS  $k" || echo "FAIL  $k   (see $OUT/$k.log)"
done
