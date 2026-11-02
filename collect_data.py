import argparse
import os
import pickle
import random
import glob

import numpy as np

import common_args
import coot_config
from utils import build_overcooked_data_filename

# Set from the command line in __main__ (--ctx_rollouts, --num_query, --no_mask_rollout).
CTX_ROLLOUTS = 5
MASK_ROLLOUT = True
NUM_QUERY = 6

def held_object_to_num(agent_held):
    if type(agent_held) == list:
        if len(agent_held) == 0:
            agent_held = None
        else:
            agent_held = agent_held[0]

    if agent_held == None:
            agent_held_num = 0
    else:
        if agent_held['name'] == "onion":
            agent_held_num = 1
        elif agent_held['name'] == "dish":
            agent_held_num = 2
        elif agent_held['name'] == "soup":
            agent_held_num = 3
        elif agent_held['name'] == "tomato":
            agent_held_num = 4
        else:
            raise ValueError("agent holding weird stuff")
    return agent_held_num

def action2num(action):
    if action == "interact":
        output = [0,0,0,0,0,1]
    elif action == (0,-1): # North
        output = [1,0,0,0,0,0]
    elif action == (0, 1): # Sorth
        output = [0,1,0,0,0,0]
    elif action == (1, 0): # East
        output = [0,0,1,0,0,0]
    elif action == (-1,0): # West
        output = [0,0,0,1,0,0]
    elif action == (0,0):
        output = [0,0,0,0,1,0]
    else:
        raise ValueError("unidentified action!")
    return output

# def ori_to_layer(ori):
#     if ori == (0,-1):
#         layout = 0
#     elif ori == (0,1):
#         layout = 1
#     elif ori == (1,0):
#         layout = 2
#     elif ori == (-1,0):
#         layout = 3
#     else:
#         raise ValueError("wrong orientation")
#     return layout

def get_obs(context_trajs):
    episode_length = (len(context_trajs)-1)//4
    context_states = []
    context_actions = []
    context_rewards = []
    context_next_states = []    
    for i in range(episode_length):
        # get state
        context_state = context_trajs[i*4][1]
        # pos0 = np.argwhere(context_state[:,:,0]==1)
        # pos1 = np.argwhere(context_state[:,:,1]==1)
        # ori0 = np.argwhere(np.array([np.any(context_state[:,:,i+2]) for i in range(4)]))
        # ori1 = np.argwhere(np.array([np.any(context_state[:,:,i+6]) for i in range(4)]))
        # ori = np.array([ori0,ori1])
        # pot_pos = np.argwhere(context_state[:,:,10]==1)
        # onion_pot0 = int(context_state[pot_pos[0][0],pot_pos[0][1],15])
        # onion_pot1 = int(context_state[pot_pos[1][0],pot_pos[1][1],15])
        # onion_pot = np.array([onion_pot0,onion_pot1])
        # cooktime_pot0 = int(context_state[pot_pos[0][0],pot_pos[0][1],16])
        # cooktime_pot1 = int(context_state[pot_pos[1][0],pot_pos[1][1],16])
        # cooktime_pot = np.array([cooktime_pot0,cooktime_pot1])
        # soup_loc = context_state[:,:,17].astype(int).flatten()
        # dish_loc = context_state[:,:,18].astype(int).flatten()
        # onion_loc = context_state[:,:,19].astype(int).flatten()
        # context_state = np.concatenate([pos0,pos1,ori,onion_pot,cooktime_pot,soup_loc,dish_loc,onion_loc], axis=None)
        context_states.append(context_state)

        # get action 
        agent1_action = context_trajs[i*4+1][1]
        agent1_action = action2num(agent1_action)
        # Can't not read partner's action
        # agent1_action = context_trajs[i*4+1][0]
        # agent1_action = action2num(agent1_action)
        context_actions.append(np.array(agent1_action))

        # get reward
        agent1_reward = context_trajs[i*4+2]
        # Can't read partner's reward
        # agent1_reward = context_trajs[i*4+2][1]
        context_reward = [agent1_reward]
        context_rewards.append(np.array(context_reward))

        # get next state
        context_next_state = context_trajs[i*4+4][1]
        # pos0 = np.argwhere(context_next_state[:,:,0]==1)
        # pos1 = np.argwhere(context_next_state[:,:,1]==1)
        # ori0 = np.argwhere(np.array([np.any(context_next_state[:,:,i+2]) for i in range(4)]))
        # ori1 = np.argwhere(np.array([np.any(context_next_state[:,:,i+6]) for i in range(4)]))
        # ori = np.array([ori0,ori1])
        # pot_pos = np.argwhere(context_next_state[:,:,10]==1)
        # onion_pot0 = int(context_next_state[pot_pos[0][0],pot_pos[0][1],15])
        # onion_pot1 = int(context_next_state[pot_pos[1][0],pot_pos[1][1],15])
        # onion_pot = np.array([onion_pot0,onion_pot1])
        # cooktime_pot0 = int(context_next_state[pot_pos[0][0],pot_pos[0][1],16])
        # cooktime_pot1 = int(context_next_state[pot_pos[1][0],pot_pos[1][1],16])
        # cooktime_pot = np.array([cooktime_pot0,cooktime_pot1])
        # soup_loc = context_next_state[:,:,17].astype(int).flatten()
        # dish_loc = context_next_state[:,:,18].astype(int).flatten()
        # onion_loc = context_next_state[:,:,19].astype(int).flatten()
        # # context_next_state = context_next_state.flatten()
        # context_next_state = np.concatenate([pos0,pos1,ori,onion_pot,cooktime_pot,soup_loc,dish_loc,onion_loc], axis=None)
        context_next_states.append(context_next_state)
        
    return context_states, context_actions, context_rewards, context_next_states


def generate_overcooked_histories(agent_ids, n_hists, n_samples, layout, rollout_root, horizon, agent_group, bias_agent_len=0, rollin_type='expert'):
    """Build (context, query, best-response action) tuples from composed rollouts.

    Rollouts are read from {rollout_root}/{pool}/run_{seed:02d}/trajs/traj_*.pkl, where pool is
    hsp_train / mep_train for the training partners and hsp_val for the validation partners
    (see compose_rollouts.py).
    """
    import pickle

    n_agents = len(agent_ids)
    ctx_rollouts = CTX_ROLLOUTS
    trajs_query_states = []
    trajs_optimal_actions = []
    trajs_context_actions = []
    trajs_context_states = []
    trajs_context_rewards = []
    state_shape = tuple(coot_config.load_layout(layout)['state_shape'])

    for i, agent_id in enumerate(agent_ids):
        if agent_group == "train":
            pool = "hsp_train" if i < bias_agent_len else "mep_train"
        elif agent_group == "test":
            pool = "hsp_val"
        traj_path = os.path.join(rollout_root, pool, f"run_{agent_id:02d}", "trajs")
        traj_files = glob.glob(os.path.join(traj_path, "traj_*.pkl"))
        if len(traj_files) <= ctx_rollouts:
            raise ValueError(f"Need more than {ctx_rollouts} rollouts in {traj_path}, found {len(traj_files)}")

        print(f"Generating {agent_group} histories for agent {i+1}/{n_agents}")

        for j in range(n_hists):
            ctx_states = []
            ctx_actions = []
            ctx_rewards = []
            ctx_next_states = []
            context_traj_files = []
            # Adaptive context rollouts
            if MASK_ROLLOUT:
                exponenet = 1.3
                p = [1/(i+1)**exponenet for i in range(ctx_rollouts+1)]
                prob = np.array(p) / sum(p)
                num_mask_rollout = np.random.choice(ctx_rollouts+1, p=prob)

                # p = [20.0 - ctx_rollouts]
                # p = [12.5]
                # for i in range(ctx_rollouts):
                #     p.append(1.0)
                # prob = np.array(p) / sum(p)
                # num_mask_rollout = np.random.choice(ctx_rollouts+1, p=prob)
            else:
                num_mask_rollout = 0
            ind_list = []
            for k in range(ctx_rollouts):
                if k < num_mask_rollout:
                    # Masked (empty) episode slot
                    context_states = [np.zeros(state_shape) for i in range(horizon)]
                    context_actions = [np.zeros(coot_config.ACTION_DIM) for i in range(horizon)]
                    context_rewards = [np.zeros(1) for i in range(horizon)]

                    ctx_states = ctx_states + context_states
                    ctx_actions = ctx_actions + context_actions
                    ctx_rewards = ctx_rewards + context_rewards
                    # ctx_next_states = ctx_next_states + context_next_states
                else:
                    # Select a context trajectory
                    try:
                        ind = random.choice(range(len(traj_files)))
                        ind_list.append(ind)
                    except:
                        raise ValueError(f"No trajectory files found for agent {agent_id}")
                    context_traj_file = traj_files.pop(ind)
                    context_traj_files.append(context_traj_file)
                    with open(context_traj_file, 'rb') as f:
                        context_trajectory = pickle.load(f)

                    # Extract context states, actions, and rewards from the trajectory
                    context_states, context_actions, context_rewards, context_next_states = get_obs(context_trajectory)
                    context_states = context_states[:horizon]
                    context_actions = context_actions[:horizon]
                    context_rewards = context_rewards[:horizon]
                    # context_next_states = context_next_states[:horizon]
                    ctx_states = ctx_states + context_states 
                    ctx_actions = ctx_actions + context_actions
                    ctx_rewards = ctx_rewards + context_rewards
                    # ctx_next_states = ctx_next_states + context_next_states

            ctx_states = np.array(ctx_states, dtype=np.uint8)
            ctx_actions = np.array(ctx_actions, dtype=np.uint8)
            trajs_context_actions.append(ctx_actions)
            trajs_context_states.append(ctx_states)
            trajs_context_rewards.append(ctx_rewards)

            if agent_group == 'train':
                all_indices = set(range(200))
            elif agent_group == 'test': 
                all_indices = set(range(50))

            query_indices = list(all_indices - set(ind_list))
            for _ in range(n_samples):
                # Select a different trajectory for query state
                query_traj_file = random.choice(traj_files)
                query_ind = random.choice(query_indices)

                with open(query_traj_file, 'rb') as f:
                    query_trajectory = pickle.load(f)

                query_states, query_actions, query_rewards, _ = get_obs(query_trajectory)
                # Sample a query state and optimal action from the query trajectory
                sample_index = random.randint(0, len(query_states) - NUM_QUERY -1)
                query_state = query_states[sample_index:sample_index+NUM_QUERY]
                optimal_action = query_actions[sample_index:sample_index+NUM_QUERY]
                assert len(query_state) == NUM_QUERY

                # traj = {
                #     'query_state': query_state,
                #     'optimal_action': optimal_action,
                #     'context_states': ctx_states,
                #     'context_actions': ctx_actions,
                #     # 'context_next_states': ctx_next_states,  # unused
                #     'context_rewards': ctx_rewards,
                #     # 'agent_seed': agent_id,  # not used during training, only used for evaling in correct env
                # }
                # trajs.append(traj)
                trajs_query_states.append(query_state)
                trajs_optimal_actions.append(optimal_action)

            for context_traj_file in context_traj_files:
                traj_files.append(context_traj_file)
        # print(f"Done with agent {i+1}/{n_agents}")

    # return trajs
    return trajs_query_states, trajs_optimal_actions, trajs_context_actions, trajs_context_states, trajs_context_rewards

if __name__ == '__main__':
    np.random.seed(0)
    random.seed(0)

    parser = argparse.ArgumentParser()
    common_args.add_dataset_args(parser)
    parser.add_argument('--ctx_rollouts', type=int, default=5, help='Episodes per context (T)')
    parser.add_argument('--num_query', type=int, default=6, help='Consecutive query states per sample')
    parser.add_argument('--no_mask_rollout', action='store_true', help='Disable masking of random context episodes')
    parser.add_argument('--population', type=int, default=36, help='Training population size (Table 18)')
    parser.add_argument('--partner_limit', type=int, default=None,
                        help='Only the first N HSP, N MEP and N validation partners (smoke tests)')
    coot_config.add_path_args(parser, ('rollout_dir', 'dataset_dir'))
    args = vars(parser.parse_args())
    print("Args: ", args)

    if args['env'] != 'overcooked':
        raise NotImplementedError("Only --env overcooked is supported")

    CTX_ROLLOUTS = args['ctx_rollouts']
    NUM_QUERY = args['num_query']
    MASK_ROLLOUT = not args['no_mask_rollout']
    rollout_dir = coot_config.resolve_path('rollout_dir', args['rollout_dir'])
    dataset_dir = coot_config.resolve_path('dataset_dir', args['dataset_dir'])

    env = args['env']
    n_hists = args['hists']
    n_samples = args['samples']
    layout_name = args['layout_name']
    dataset_prefix = args['dataset_prefix']
    shuffle = args['shuffle_dataset']
    episode_length = args['episode_length']

    # HSP partners first, then MEP partners; the order decides which rollout pool each one reads.
    limit = args['partner_limit']
    train_bias_agent_list = coot_config.get_seeds(layout_name, 'train_hsp', args['population'])[:limit]
    train_mep_agent_list = coot_config.get_seeds(layout_name, 'train_mep', args['population'])[:limit]
    test_agent_list = coot_config.get_seeds(layout_name, 'val')[:limit]
    train_agent_ids = np.array(train_bias_agent_list + train_mep_agent_list)
    test_agent_ids = np.array(test_agent_list)
    n_agents = len(train_agent_ids)
    print(f"{n_agents} training partners ({len(train_bias_agent_list)} HSP + {len(train_mep_agent_list)} MEP), "
          f"{len(test_agent_ids)} validation partners")

    if shuffle:
        random.shuffle(train_agent_ids)

    config = {
        'n_hists': n_hists,
        'n_samples': n_samples,
        'horizon': args['H'],
        'layout': layout_name,
        'evaluation': False,
        'use_random_player_pos': True,
        'use_random_terrain_state': True,
        'shuffle': shuffle,
        'ctx_rollouts': CTX_ROLLOUTS,
    }

    # File names cover partners 1..n_agents, which is what train.py --agents n_agents reads.
    train_filepath = build_overcooked_data_filename(
        env, 1, n_agents, config, mode=0, layout=layout_name, prefix=dataset_prefix, root=dataset_dir)
    test_filepath = build_overcooked_data_filename(
        env, 1, n_agents, config, mode=1, layout=layout_name, prefix=dataset_prefix, root=dataset_dir)
    os.makedirs(os.path.dirname(train_filepath), exist_ok=True)
    train_filepath = os.path.splitext(train_filepath)[0]
    test_filepath = os.path.splitext(test_filepath)[0]

    rollout_root = os.path.join(rollout_dir, layout_name)
    for split, agent_ids, filepath in (("train", train_agent_ids, train_filepath), ("test", test_agent_ids, test_filepath)):
        query_states, optimal_actions, context_actions, context_states, context_rewards = generate_overcooked_histories(
            agent_ids, n_hists, n_samples, layout_name, rollout_root, episode_length,
            agent_group=split, bias_agent_len=len(train_bias_agent_list))
        for suffix, data in (("_query_s", query_states), ("_optimal_a", optimal_actions), ("_context_s", context_states),
                             ("_context_a", context_actions), ("_context_r", context_rewards)):
            with open(filepath + suffix + ".pkl", 'wb') as file:
                pickle.dump(data, file)
        print(f"Saved {split} split to {filepath}_*.pkl")
