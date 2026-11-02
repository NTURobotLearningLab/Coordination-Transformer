#!/bin/bash
env="Overcooked"

layout=$1

entropy_coefs="0.2 0.05 0.001"
entropy_coef_horizons="0 6e6 1e7"
if [[ "${layout}" == "small_corridor" ]]; then
    entropy_coefs="0.2 0.05 0.001"
    entropy_coef_horizons="0 8e6 1e7"
fi

reward_shaping_horizon="0"
num_env_steps="1e7"

num_agents=2
algo="mappo"
stage="S1"
exp="bias_mid_${stage}_0429"


if [[ "${layout}" == "random0" || "${layout}" == "random0_medium" || "${layout}" == "random1" || "${layout}" == "random3" || "${layout}" == "small_corridor" || "${layout}" == "unident_s" ]]; then
    version="old"
    # old layouts
    #! positive reward shaping for "[op]_X" may crash the training, be careful
    #! negative reward shaping for "put_X" may be meaningless
    # 0."put_onion_on_X",
    # 1."put_dish_on_X",
    # 2."put_soup_on_X",
    # 3."pickup_onion_from_X", random0_medium random0_hard
    # 4."pickup_onion_from_O", all_old
    # 5."pickup_dish_from_X",
    # 6."pickup_dish_from_D", all_old
    # 7."pickup_soup_from_X", random0 random0_medium random0_hard
    # 8."USEFUL_DISH_PICKUP", default
    # 9."SOUP_PICKUP", all_old default
    # 10."PLACEMENT_IN_POT", all_old default
    # 11."delivery", all_old
    # 12."STAY", all_old
    # 13."MOVEMENT",
    # 14."IDLE_MOVEMENT",
    # 15."IDLE_INTERACT_X",
    # 16."IDLE_INTERACT_EMPTY",
    # 17.sparse_reward all_old

    w1="0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1"
    if [[ "${layout}" == "random0" ]]; then
        w0="0,0,0,0,[0:10],0,[0:10],[-20:0],3,5,3,0,[-0.1:0:0.1],0,0,0,0,[0.1:1]"
        seed_begin=1
        seed_max=30
    elif [[ "${layout}" == "random0_medium" ]]; then
        w0="0,0,0,[-20:0],[-20:0:10],0,[0:10],[-20:0],3,5,3,0,[-0.1:0:0.1],0,0,0,0,[0.1:1]"
        seed_begin=1
        seed_max=54
    elif [[ "${layout}" == "small_corridor" ]]; then
        w0="0,0,0,0,[-20:0:5],0,[-20:0:5],0,3,5,3,[-20:0],[-0.1:0],0,0,0,0,[0.1:1]"
        seed_begin=1
        seed_max=124
    else
        # w0="0,0,0,0,[-20:0:10],0,[-20:0:10],0,3,5,3,[-20:0],[-0.1:0:0.1],0,0,0,0,[0.1:1]"
        w0 = "0,0,0,0,[-20:0:10],0,[-20:0:10],0,3,[-20:0:5:10],[-20:0:3:10],[-20:0],[-0.1:0:0.1],0,0,0,0,[0.1:1]"
        seed_begin=1
        seed_max=176
    fi
else
    version="new"
    # 0 "put_onion_on_X",
    # 1 "put_tomato_on_X",
    # 2 "put_dish_on_X",
    # 3 "put_soup_on_X",
    # 4 "pickup_onion_from_X",
    # 5 "pickup_onion_from_O",
    # 6 "pickup_tomato_from_X",
    # 7 "pickup_tomato_from_T",
    # 8 "pickup_dish_from_X",
    # 9 "pickup_dish_from_D",
    # 10 "pickup_soup_from_X",
    # 11 "USEFUL_DISH_PICKUP",  # counted when #taken_dishes < #cooking_pots + #partially_full_pots and no dishes on the counter
    # 12 "SOUP_PICKUP",  # counted when soup in the pot is picked up (not a soup placed on the table)
    # 13 "PLACEMENT_IN_POT",  # counted when some ingredient is put into pot
    # 14 "viable_placement",
    # 15 "optimal_placement",
    # 16 "catastrophic_placement",
    # 17 "useless_placement",  # pot an ingredient to a useless recipe
    # 18 "potting_onion",
    # 19 "potting_tomato",
    # 20 "cook",
    # 21 "delivery",
    # 22 "deliver_size_two_order",
    # 23 "deliver_size_three_order",
    # 24 "deliver_useless_order",
    # 25 "STAY",
    # 26 "MOVEMENT",
    # 27 "IDLE_MOVEMENT",
    # 28 "IDLE_INTERACT",
    # 29 sparse_reward
    w1="0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1"
    w0="0,0,0,0,0,0,0,0,0,0,0,3,5,3,0,0,0,0,[-20:0],[-20:0],0,0,[-5:0:20],[-15:0:10],0,[-0.1:0:0.1],0,0,0,1"
    seed_begin=1
    seed_max=72
fi
seed_begin=1
seed_max=10

for seed in $(seq ${seed_begin} ${seed_max});
do
    echo "seed is ${seed}:"
    python train/train_mid_bias_br.py --env_name ${env} --algorithm_name ${algo} --experiment_name "${exp}" --layout_name ${layout} --num_agents ${num_agents} \
    --mid_seed ${seed} --n_training_threads 1 --n_rollout_threads 100 --dummy_batch_size 2 --num_mini_batch 1 --episode_length 200 --num_env_steps ${num_env_steps} --reward_shaping_horizon ${reward_shaping_horizon} \
    --overcooked_version ${version} \
    --ppo_epoch 20 --entropy_coefs ${entropy_coefs} --entropy_coef_horizons ${entropy_coef_horizons} \
    --use_hsp --w0 ${w0} --w1 ${w1} --share_policy \
    --cnn_layers_params "32,3,1,1 64,3,1,1 32,3,1,1" --use_recurrent_policy \
    --use_proper_time_limits \
    --save_interval 50 --log_interval 20 \
    --wandb_name "your_wandb_entity" --use_wandb \
    --use_eval --eval_interval 200 --n_eval_rollout_threads 100 --train_mid_bias
done