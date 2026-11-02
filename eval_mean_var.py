import json
import numpy as np
# from envs.overcooked.overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
# from envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
# from envs.overcooked.Overcooked_Env import Overcooked
from zsceval.envs.env_wrappers import ShareDummyVecEnv, ShareSubprocDummyBatchVecEnv
from zsceval.envs.overcooked.Overcooked_Env import Overcooked
from zsceval.envs.overcooked_new.Overcooked_Env import Overcooked as Overcooked_new


import argparse
import common_args
import torch
import time
import pickle
import os
import scipy
import importlib
import pandas as pd

# from configs.config import get_config
# from configs.overcooked_config import get_overcooked_args
from zsceval.config import get_config
from zsceval.overcooked_config import get_overcooked_args
from zsceval.utils.train_util import get_base_run_dir, setup_seed


from utils import build_overcooked_data_filename, build_overcooked_model_filename, convert_to_tensor
from ctrls.ctrl_bandit import Controller
from net import Transformer
import coot_config
import wandb

# Module-level settings read by the deploy functions below. They are overwritten from the
# command line in __main__ (--wandb, --use_render, --count_events, --ctx_rollouts, --seed_type,
# --heps, --group_prefix, --skill_level, --eval_mode).
USE_WANDB = False
USE_RENDER = False
COUNT_EVENTS = False
MODEL_NAME = "DPT"
CTX_ROLLOUTS = 5
SEED_TYPE = "test"
EPISODE_LENGTH = 200
HEPS = 50
GROUP_PREFIX = "eval"
SKILL_LEVEL = "final" # mid or final
EVAL_MODE = "online"


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def make_eval_env(all_args, run_dir,seed):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "Overcooked":
                if all_args.overcooked_version == "old":
                    env = Overcooked(all_args, run_dir, evaluation=True)
                else:
                    env = Overcooked_new(all_args, run_dir, evaluation=True)
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

'''
The obs.shape = (layout_height, layout_width, layers), there are 20 layers
0.  player_0_loc
1.  player_1_loc
2.  player_0_ori_0 = (0,-1) = North
3.  player_0_ori_1 = (0, 1) = South
4.  player_0_ori_2 = (1, 0) = East
5.  player_0_ori_3 = (-1,0) = West
6.  player_1_ori_0
7.  player_1_ori_1
8.  player_1_ori_2
9.  player_1_ori_3
10. pot_loc
11. counter_loc
12. onion_disp_loc
13. dish_disp_loc
14. serve_loc
15. onions_in_pot
16. onions_cook_time
17. onion_soup_loc
18. dishes
19. onions , it won't count onions in the pot and onions in the dispenser
'''

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

    # mep
    parser.add_argument(
        "--population_yaml_path",
        type=str,
        help="Path to yaml file that stores the population info.",
    )
    parser.add_argument(
        "--stage",
        type=int,
        default=1,
        help="Stages of MEP training. 1 for Maximum-Entropy PBT. 2 for FCP-like training.",
    )
    parser.add_argument(
        "--mep_use_prioritized_sampling",
        default=False,
        action="store_true",
        help="Use prioritized sampling in MEP stage 2.",
    )
    parser.add_argument(
        "--mep_prioritized_alpha",
        type=float,
        default=3.0,
        help="Alpha used in softing prioritized sampling probability.",
    )
    parser.add_argument(
        "--mep_entropy_alpha",
        type=float,
        default=0.01,
        help="Weight for population entropy reward. MEP uses 0.01 in general except 0.04 for Forced Coordination",
    )
    parser.add_argument("--eval_policy", default="", type=str)
    parser.add_argument("--eval_result_path", default=None, type=str)
    # population
    parser.add_argument(
        "--population_size",
        type=int,
        default=5,
        help="Population size involved in training.",
    )
    parser.add_argument(
        "--adaptive_agent_name",
        type=str,
        required=False,
        help="Name of training policy at Stage 2.",
    )

    # train and eval batching
    parser.add_argument(
        "--train_env_batch",
        type=int,
        default=1,
        help="Number of parallel threads a policy holds",
    )
    parser.add_argument(
        "--eval_env_batch",
        type=int,
        default=1,
        help="Number of parallel threads a policy holds",
    )

    # fixed policy actions inside env threads
    parser.add_argument(
        "--use_policy_in_env",
        default=True,
        action="store_false",
        help="Use loaded policy to move in env threads.",
    )
    parser.add_argument(
        "--eval_w0",
        type=str,
        default="0,0,0,0,0,0,0,0,3,5,3,0,0,0,0,0,0,0",
        help="Weight vector of dense reward 0 in overcooked env.",
    )

    parser.add_argument("--use_task_v_out", default=False, action="store_true")
    
    all_args = parser.parse_args(args)
    from zsceval.overcooked_config import OLD_LAYOUTS

    if all_args.layout_name in OLD_LAYOUTS:
        all_args.old_dynamics = True
    else:
        all_args.old_dynamics = False
    return all_args

def make_trainer_policy_cls(algorithm_name, use_single_network=False):
    algorithm_dict = {
        "rmappo": (
            "zsceval.algorithms.r_mappo.r_mappo.R_MAPPO",
            "zsceval.algorithms.r_mappo.algorithm.rMAPPOPolicy.R_MAPPOPolicy",
        ),
        "mappo": (
            "zsceval.algorithms.r_mappo.r_mappo.R_MAPPO",
            "zsceval.algorithms.r_mappo.algorithm.rMAPPOPolicy.R_MAPPOPolicy",
        ),
        "population": (
            "zsceval.algorithms.population.trainer_pool.TrainerPool",
            "zsceval.algorithms.population.policy_pool.PolicyPool",
        ),
        "mep": (
            "zsceval.algorithms.population.mep.MEP_Trainer",
            "zsceval.algorithms.population.policy_pool.PolicyPool",
        ),
        "adaptive": (
            "zsceval.algorithms.population.mep.MEP_Trainer",
            "zsceval.algorithms.population.policy_pool.PolicyPool",
        ),
        "cole": (
            "zsceval.algorithms.population.cole.COLE_Trainer",
            "zsceval.algorithms.population.policy_pool.PolicyPool",
        ),
        "traj": (
            "zsceval.algorithms.population.traj.Traj_Trainer",
            "zsceval.algorithms.population.policy_pool.PolicyPool",
        ),
    }

    if algorithm_name not in algorithm_dict:
        raise NotImplementedError

    train_algo_module, train_algo_class = algorithm_dict[algorithm_name][0].rsplit(".", 1)
    policy_module, policy_class = algorithm_dict[algorithm_name][1].rsplit(".", 1)

    TrainAlgo = getattr(importlib.import_module(train_algo_module), train_algo_class)
    Policy = getattr(importlib.import_module(policy_module), policy_class)

    return TrainAlgo, Policy


class OvercookedController(Controller):
    def __init__(self, model, batch_size=1, sample=True):
        self.model = model
        self.state_dim = model.config['state_dim']
        self.action_dim = model.config['action_dim']
        self.horizon = model.horizon
        if batch_size==1:
            self.zeros = torch.zeros(self.state_dim **2 + self.action_dim + 1).float().to(device)
        else:
            self.zeros = torch.zeros(batch_size, self.state_dim ** 2 + self.action_dim + 1).float().to(device)
        self.sample = sample
        self.temp = 1.0
        self.batch_size = batch_size
        
        if model.config['layout_name'] == "random0_medium":
            self.layout_w = 8
            self.layout_h = 5
        elif model.config['layout_name'] == "random1" or model.config['layout_name'] == "random0":
            self.layout_w = 5
            self.layout_h = 5
        elif model.config['layout_name'] == "random1_m" or model.config['layout_name'] == "random0_m":
            self.layout_w = 5
            self.layout_h = 5
        elif model.config['layout_name'] == "random3":
            self.layout_w = 8
            self.layout_h = 5
        elif model.config['layout_name'] == "unident_s":
            self.layout_w = 9
            self.layout_h = 5

        self.layout_c = model.config['state_dim'] // (self.layout_w * self.layout_h)
        self.num_query = model.config['num_query']
        self.query_states = torch.zeros((self.num_query, self.layout_w, self.layout_h, self.layout_c)).to(device)

    def act(self, state):
        self.batch['zeros'] = self.zeros
        self.query_states = torch.cat([self.query_states[1:,:,:,:], torch.zeros((1, self.layout_w, self.layout_h, self.layout_c)).to(device)], dim=0)

        states = torch.tensor(np.array(state)).float().to(device)
        if self.batch_size == 1:
            states = states[None, :]
        self.query_states[-1,:,:,:] = states
        self.batch['query_states'] = self.query_states

        actions = self.model(self.batch)
        actions = actions.cpu().detach().numpy()

        # if self.batch_size == 1:
        #     actions = actions

        if self.sample:
            if self.batch_size > 1:
                action_indices = []
                for idx in range(self.batch_size):
                    probs = scipy.special.softmax(actions[idx] / self.temp)
                    sampled_action = np.random.choice(
                        np.arange(self.action_dim), p=probs)
                    action_indices.append(sampled_action)
            else:
                probs = scipy.special.softmax(actions / self.temp).reshape(-1)
                action_indices = [np.random.choice(np.arange(self.action_dim), p=probs)]
        else:
            action_indices = np.argmax(actions, axis=-1)

        actions = np.zeros((self.batch_size, self.action_dim))
        actions[np.arange(self.batch_size), action_indices] = 1.0
        if self.batch_size == 1:
            actions = actions[0]
        return actions
    
    def reset_query_states(self):
        self.query_states = torch.zeros((self.num_query, self.layout_w, self.layout_h, self.layout_c)).to(device)


def load_policy(policy_path):
    """Load a policy from a file. This function assumes the policy can be loaded and used as a callable."""
    # Placeholder: Replace with actual policy loading logic
    def policy(state):
        return np.random.choice([0, 1, 2, 3, 4])  # Sample random action (replace with real policy)
    return policy

def js_divergence_two_vectors(arr1, arr2):
    """
    Calculate Jensen-Shannon divergence between two event count vectors.
    
    Args:
        arr1: numpy array of shape (m,) containing counts for m events
        arr2: numpy array of shape (m,) containing counts for m events
    
    Returns:
        JS divergence between the two distributions
    """
    from scipy.spatial.distance import jensenshannon
    
    # Convert counts to probability distributions
    dist1 = np.zeros_like(arr1, dtype=float)
    dist2 = np.zeros_like(arr2, dtype=float)
    
    # Normalize first array
    total1 = np.sum(arr1)
    if total1 > 0:
        dist1 = arr1 / total1
    else:
        dist1 = np.ones_like(arr1) / len(arr1)  # Uniform if no events
    
    # Normalize second array
    total2 = np.sum(arr2)
    if total2 > 0:
        dist2 = arr2 / total2
    else:
        dist2 = np.ones_like(arr2) / len(arr2)  # Uniform if no events
    
    # Calculate JS divergence
    js_div = jensenshannon(dist1, dist2)
    
    # Define event names for the CSV
    event_names = [
        'put_onion_on_X', 'put_dish_on_X', 'put_soup_on_X', 
        'pickup_onion_from_X', 'pickup_onion_from_O', 
        'pickup_dish_from_X', 'pickup_dish_from_D', 
        'pickup_soup_from_X', 'USEFUL_DISH_PICKUP', 
        'SOUP_PICKUP', 'PLACEMENT_IN_POT', 'delivery', 
        'STAY', 'MOVEMENT', 'IDLE_MOVEMENT', 
        'IDLE_INTERACT_X', 'IDLE_INTERACT_EMPTY'
    ]

    # Save to CSV if requested
    if model_seed is not None and layout_name is not None:
        # Create directory if it doesn't exist
        js_div_dir = os.path.join(results_dir, "js_div", layout_name)
        os.makedirs(js_div_dir, exist_ok=True)

        # Create filename with timestamp
        # timestamp = datetime.now().strftime("%m%d_%H%M%S")
        csv_filename = f"{js_div_dir}/js_div_seed{model_seed}_{MODEL_NAME}.csv"
        
        with open(csv_filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            # writer.writerow(['model_seed', model_seed])
            writer.writerow(['model_name', MODEL_NAME])
            writer.writerow(['Event']+[event_name for event_name in event_names])
            
            # Write data for each event
            writer.writerow(['distribution1']+[f"{dist1[i]:.6f}" for i in range(len(event_names))])
            writer.writerow(['distribution2']+[f"{dist2[i]:.6f}" for i in range(len(event_names))])
            # Write summary row
            writer.writerow(['JS Divergence', f"{js_div:.6f}", "\n"])

            
        print(f"JS divergence results saved to {csv_filename}")
    

    return {
        'js_divergence': js_div,
        'distribution1': dist1,
        'distribution2': dist2
    }

def event_to_num(event_info):
    event_info = event_info[1]
    # Define the ordered list of event keys
    event_keys = [
        'put_onion_on_X', 'put_dish_on_X', 'put_soup_on_X', 
        'pickup_onion_from_X', 'pickup_onion_from_O', 
        'pickup_dish_from_X', 'pickup_dish_from_D', 
        'pickup_soup_from_X', 'USEFUL_DISH_PICKUP', 
        'SOUP_PICKUP', 'PLACEMENT_IN_POT', 'delivery', 
        'STAY', 'MOVEMENT', 'IDLE_MOVEMENT', 
        'IDLE_INTERACT_X', 'IDLE_INTERACT_EMPTY'
    ]
    
    # Initialize array with zeros
    event_arr = np.zeros(len(event_keys))
    
    # Set array values based on dictionary values
    for i, key in enumerate(event_keys):
        if key in event_info:
            if event_info[key] == 1:
                event_arr[i] = 1
        else:
            raise ValueError(f"Event key {key} not found in event_info")
    nonzero_indices = np.nonzero(event_arr)[0]
    return nonzero_indices

def deploy_eval(env, ctrl, trainer, config, best_response_reward=False):
    """Runs two policies in the Overcooked environment and logs state-action-reward data."""
    obs, share_obs, avail_actions = env.reset()
    layout_w, layout_h, layout_c = env.observation_space[0].shape
    
    layout_name = config['layout_name']

    obs = np.stack(obs).reshape(1,2,layout_w,layout_h,-1)
    avail_actions = avail_actions.reshape(1,2,6)
    # obs = []
    # acts = []
    # next_obs = []
    # rews = []
    done = False

    rnn_states = np.zeros((
        1, # n_eval_rollout_threads
        2, # num_agents
        1, # recurrent_N
        64, # hidden_size
       ), dtype=np.float32,)
    masks = np.ones((
        1, # n_eval_rollout_threads
        2, # num_agents
        1
        ), dtype=np.float32)

    episode_rewards_w0 = []
    episode_rewards_w1 = []
    episode_length = EPISODE_LENGTH
    context_states = []
    context_state = obs[0,1]
    context_states.append(context_state)

    context_actions = []
    # context_next_states = []
    context_rewards = []
    if COUNT_EVENTS:
        context_events = np.zeros((17))
    average_time = []
    for _ in range(episode_length):
        # agent_index = 1 (transformer, not biased)
        # agent_index = 0 (Biased model)
        trainer[0].prep_rollout()

        act0, rnn_state = trainer[0].policy.act(
            obs[:,0],  
            rnn_states[:, 0],
            masks[:,0],
            avail_actions[:,0],
            deterministic=True
        )
        step_start_time = time.time()
        act1 = ctrl.act(context_state)
        step_end_time = time.time()
        # print(f"Time taken for act1: {step_end_time - step_start_time} seconds")
        average_time.append(step_end_time - step_start_time)

        act0 = act0.cpu().numpy()
        act1_num = np.argwhere(act1==1)
        # eval_action = eval_action.cpu().numpy()
        rnn_states[:, agent_id] = rnn_state.cpu().numpy()
        # acts = np.stack([act0, act1]).transpose(1,0,2)
        acts = np.stack([act0,act1_num]).transpose(1, 0, 2)
        (
            obs, 
            _,
            rewards,
            dones,
            infos,
            avail_actions,
        ) = env.step(acts)

        # agent1
        context_state = obs[0][1]
        context_states.append(context_state)
        context_actions.append(act1)
        context_rewards.append(rewards[0][1])
        if COUNT_EVENTS:
            event_num = event_to_num(infos[0]['shaped_info_by_agent'])
            context_events[event_num] += 1
        # agent0
        obs = np.stack(obs).reshape(1,2,layout_w,layout_h,-1)
        avail_actions = avail_actions.reshape(1,2,6)

        rnn_states[dones == True] = np.zeros(
            ((dones == True).sum(), 1, 64),
            dtype=np.float32,
        )
        masks = np.ones((1, 2, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        # onion_in_pots = np.sum(obs[0,1,:,:,15])
        # rewards_stack = np.stack(rewards)
        episode_rewards_w0.append(rewards[0][0])
        episode_rewards_w1.append(rewards[0][1])

    print(f"Average time taken for step: {np.mean(average_time[1:])} seconds")
    episode_rewards_sum_w0 = np.sum(np.array(episode_rewards_w0))
    episode_rewards_sum_w1 = np.sum(np.array(episode_rewards_w1))
    print(f"episode rewards w0: {episode_rewards_sum_w0}")    
    print(f"episode rewards w1: {episode_rewards_sum_w1}")    
    ctrl.reset_query_states()
    # Log rewards_sum to wandb
    if USE_WANDB:
        if best_response_reward != False:

            wandb.log({"episode_rewards_w0": episode_rewards_sum_w0, "episode_rewards_w1": episode_rewards_sum_w1, "BR_approx(w1)": episode_rewards_sum_w1 / best_response_reward[1], "BR_rewards": best_response_reward[1]})
            br_prox = episode_rewards_sum_w1 / best_response_reward[1]
            if br_prox > 1 or br_prox < 0:
                print(f"BR_approx(w1) is out of range: {episode_rewards_sum_w1} / {best_response_reward[1]} = {br_prox}")
        else:
            wandb.log({"episode_rewards_w0": episode_rewards_sum_w0, "episode_rewards_w1": episode_rewards_sum_w1})
        # traj_gif_path = f"{env.envs[0].run_dir}/gifs/{env.envs[0].layout_name}_seed{env.envs[0].model_seed:02d}/traj_num_{(env.envs[0].traj_num-1)//2}/reward_{int(episode_rewards_sum_w1)}.gif"
        traj_gif_path = f"{config['run_dir']}/gifs/{layout_name}_seed{config['model_seed']:02d}/traj_num{config['traj_num']}/reward_{int(episode_rewards_sum_w1)}.gif"
        # f"{self.run_dir}/gifs/{self.layout_name}/traj_num_{(self.traj_num-1)//2}"
        # wandb.log({"traj_gif": wandb.Video(traj_gif_path, format="gif")})
    config['traj_num'] += 1
    
    if COUNT_EVENTS:
        return episode_rewards_sum_w1, context_states[:-1], context_actions, context_rewards, context_states[1:], context_events
    else:
        return episode_rewards_sum_w1, context_states[:-1], context_actions, context_rewards, context_states[1:]

def deploy_bestresponse(env, trainer, config, determ=True):
    obs, _, avail_actions = env.reset()
    if env.envs[0].layout_name == "random0_medium" or env.envs[0].layout_name == "random3":
        layout_w = 8
        layout_h = 5
    elif env.envs[0].layout_name == "random1" or env.envs[0].layout_name == "random0":
        layout_w = 5
        layout_h = 5
    elif env.envs[0].layout_name == "random1_m" or env.envs[0].layout_name == "random0_m":
        layout_w = 5
        layout_h = 5
    elif env.envs[0].layout_name == "unident_s":
        layout_w = 9
        layout_h = 5

    obs = np.stack(obs).reshape(1, 2, layout_w, layout_h, -1)
    avail_actions = avail_actions.reshape(1,2,6)
    # obs = np.stack(obs)

    # obs = []
    # acts = []
    # next_obs = []
    # rews = []
    done = False

    rnn_states = np.zeros((
        1, # n_eval_rollout_threads
        2, # num_agents
        1, # recurrent_N
        64, # hidden_size
       ), dtype=np.float32,)
    masks = np.ones((
        1, # n_eval_rollout_threads
        2, # num_agents
        1
        ), dtype=np.float32)

    episode_rewards_w0 = []
    episode_rewards_w1 = []
    episode_length = env.envs[0].episode_length
    # pos0 = np.argwhere(obs[0,0,:,:,0]==255)[0]
    # pos1 = np.argwhere(obs[0,0,:,:,1]==255)[0]
    # episode_pos = [np.concatenate([pos0,pos1])]
    # episode_obs = [obs//255]
    # episode_actions = []
    # episode_ava_act = [avail_actions]
    
    context_states = []
    context_state = obs[0,1]
    context_states.append(context_state)

    context_actions = []
    # context_next_states = []
    context_rewards = []
    context_events = np.zeros((17))
    for _ in range(episode_length):
        eval_actions = []
        for agent_id in range(2):
            trainer[agent_id].prep_rollout()
            eval_action, eval_rnn_state = trainer[agent_id].policy.act(
                obs[:, agent_id],
                rnn_states[:, agent_id],
                masks[:, agent_id],
                avail_actions[:, agent_id],
                deterministic=determ,
            )

            # act0 = act0.cpu().numpy()
            eval_action = eval_action.cpu().numpy()
            eval_actions.append(eval_action)
            # eval_rnn_states[:, agent_id] = _t2n(eval_rnn_state)
            rnn_states[:, agent_id] = eval_rnn_state.cpu().numpy()

            # masks = np.ones((1, 2, 1), dtype=np.float32)
        # eval_actions = np.array(eval_actions)
        eval_actions = np.stack(eval_actions).transpose(1, 0, 2)

        (
            obs, 
            _,
            rewards,
            dones,
            infos,
            avail_actions,
        ) = env.step(eval_actions)
        # obs = np.stack(obs).reshape(1,2,5,5,-1)
        # avail_actions = avail_actions.reshape(1,2,6)
        # dones = np.array([dones])

        context_state = obs[0][1]
        context_states.append(context_state)
        act = np.zeros(6)
        act[eval_actions[0,1,0]] = 1.0
        context_actions.append(act)
        context_rewards.append(rewards[0][1])
        if COUNT_EVENTS:
            event_num = event_to_num(infos[0]['shaped_info_by_agent'])
            context_events[event_num] += 1
            

        obs = np.stack(obs).reshape(1, 2, layout_w, layout_h, -1)
        avail_actions = avail_actions.reshape(1,2,6)

        rnn_states[dones == True] = np.zeros(
            ((dones == True).sum(), 1, 64),
            dtype=np.float32,
        )
        masks = np.ones((1, 2, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        # pos0 = np.argwhere(obs[0,0,:,:,0]==255)[0]
        # pos1 = np.argwhere(obs[0,0,:,:,1]==255)[0]
        # episode_pos.append(np.concatenate([pos0,pos1]))
        # episode_actions.append(np.concatenate(eval_actions))
        # episode_obs.append(obs//255)
        # episode_ava_act.append(avail_actions)
        # onion_pots = obs[0,0,3,0,15]//255 + obs[0,0,4,1,15]//255
        episode_rewards_w0.append(rewards[0,0])
        episode_rewards_w1.append(rewards[0,1])
        

    episode_rewards_sum_w0 = np.sum(np.stack(episode_rewards_w0))
    episode_rewards_sum_w1 = np.sum(np.stack(episode_rewards_w1))
    print(f"episode rewards w0: {episode_rewards_sum_w0}")
    print(f"episode rewards w1: {episode_rewards_sum_w1}")    

    if USE_WANDB:
        traj_gif_path = f"{env.envs[0].run_dir}/gifs/{env.envs[0].layout_name}_seed{env.envs[0].model_seed:02d}/traj_num0/reward_{int(episode_rewards_sum_w1)}.gif"
        # wandb.log({"traj_gif": wandb.Video(traj_gif_path, format="gif")})

    if COUNT_EVENTS:
        return [episode_rewards_sum_w0, episode_rewards_sum_w1], context_states[:-1], context_actions, context_rewards, context_events
    else:
        return [episode_rewards_sum_w0, episode_rewards_sum_w1], context_states[:-1], context_actions, context_rewards, 


def deploy_offline(env, br_env, controller, eval_trainer, config, csv_filename, Heps=5, best_response_reward=False):
    ctx_rollouts = config["horizon"] // config["ep_hor"]
    ctx_rollouts = CTX_ROLLOUTS
    # if env.envs[0].layout_name == "random0_medium":
    #     layout_w = 8
    #     layout_h = 5
    # elif env.envs[0].layout_name == "random1":
    #     layout_w = 5
    #     layout_h = 5
    layout_w, layout_h, layout_c = env.observation_space[0].shape

    # layout_c = config["state_dim"] // (layout_w * layout_h)
    horizon = config['ep_hor'] # eval episode horizon, might different from context horizon
    context_states = torch.zeros((ctx_rollouts, horizon, layout_w, layout_h, layout_c)).to(device)
    context_actions = torch.zeros((ctx_rollouts, horizon, config["action_dim"])).to(device)
    # context_next_states = torch.zeros((ctx_rollouts, horizon, layout_w, layout_h, layout_c)).to(device)
    context_rewards = torch.zeros((ctx_rollouts, horizon, 1)).to(device)

    best_response_reward, states_lnr, actions_lnr, rewards_lnr, br_events_lnr = deploy_bestresponse(br_env, eval_trainer, config, determ=True)

    cum_means = []
    cum_js_div = []
    for i in range(HEPS):

        batch = {
            'context_states': context_states.reshape(-1, layout_w, layout_h, layout_c),
            'context_actions': context_actions.reshape(-1, config["action_dim"]),
            # 'context_next_states': context_next_states.reshape(-1, layout_w, layout_h, layout_c),
            'context_rewards': context_rewards.reshape(-1, 1),
        }
        controller.set_batch(batch)

        # for ctx in range(ctx_rollouts):
        #     if COUNT_EVENTS:
        #         _, states_lnr, actions_lnr, rewards_lnr, br_events_lnr = deploy_bestresponse(br_env, eval_trainer, config, determ=True)
        #     else:
        #         _, states_lnr, actions_lnr, rewards_lnr = deploy_bestresponse(br_env, eval_trainer, config, determ=False)

        #     context_states = torch.cat((context_states[1:, :, :, :, :], convert_to_tensor([states_lnr])), dim=0)
        #     context_actions = torch.cat((context_actions[1:, :, :], convert_to_tensor([actions_lnr])), dim=0)
        #     # context_next_states = torch.cat((context_next_states[1:, :, :, :, :], convert_to_tensor([next_states_lnr])), dim=0)
        #     context_rewards = torch.cat((context_rewards[1:, :, :], convert_to_tensor([rewards_lnr])), dim=0)
        rewards_sum, states_lnr, actions_lnr, rewards_lnr, next_states_lnr, events_lnr = deploy_eval(env, controller, eval_trainer, config, best_response_reward)
        
        context_states = torch.cat((context_states[1:, :, :, :, :], convert_to_tensor([states_lnr])), dim=0)
        context_actions = torch.cat((context_actions[1:, :, :], convert_to_tensor([actions_lnr])), dim=0)
        # context_next_states = torch.cat((context_next_states[1:, :, :, :, :], convert_to_tensor([next_states_lnr])), dim=0)
        context_rewards = torch.cat((context_rewards[1:, :, :], convert_to_tensor([rewards_lnr])), dim=0)
        
        # if COUNT_EVENTS:
        #         event = (events_lnr, br_events_lnr)
        #         event_js = js_divergence_two_vectors(event[0], event[1])

        # batch = {
        #     'context_states': context_states.reshape(-1, layout_w, layout_h, layout_c),
        #     'context_actions': context_actions.reshape(-1, config["action_dim"]),
        #     # 'context_next_states': context_next_states.reshape(-1, layout_w, layout_h, layout_c),
        #     'context_rewards': context_rewards.reshape(-1, 1),
        # }
        # controller.set_batch(batch)

        # if COUNT_EVENTS:
        #     rewards_sum, states_lnr, actions_lnr, rewards_lnr, next_states_lnr, events_lnr = deploy_eval(env, controller, eval_trainer, config, best_response_reward)
        # else:
        #     rewards_sum, states_lnr, actions_lnr, rewards_lnr, next_states_lnr = deploy_eval(env, controller, eval_trainer, config, best_response_reward)
        # print(f"======== Reward sum: {rewards_sum} ========")
        cum_means.append(rewards_sum)

        # context_states = torch.cat((context_states[1:, :, :, :, :], convert_to_tensor([states_lnr])), dim=0)
        # context_actions = torch.cat((context_actions[1:, :, :], convert_to_tensor([actions_lnr])), dim=0)
        # # context_next_states = torch.cat((context_next_states[1:, :, :, :, :], convert_to_tensor([next_states_lnr])), dim=0)
        # context_rewards = torch.cat((context_rewards[1:, :, :], convert_to_tensor([rewards_lnr])), dim=0)

        if i >= 0:
            if COUNT_EVENTS:
                event = (events_lnr, br_events_lnr)
                event_js = js_divergence_two_vectors(event[0], event[1])
                cum_js_div.append(event_js["js_divergence"])
                if i % 49 == 0:
                    # Create directory if it doesn't exist
                    js_div_dir = os.path.join(results_dir, "js_div", layout_name)
                    os.makedirs(js_div_dir, exist_ok=True)
                    
                    # Create filename with timestamp
                    # timestamp = datetime.now().strftime("%m%d_%H%M%S")
                    js_div_filename = f"{js_div_dir}/js_div_seed{model_seed}_{MODEL_NAME}.csv"
                    
                    with open(js_div_filename, 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        
                        # Write header
                        writer.writerow(['avg_50js_div', np.mean(cum_js_div[-10:])])

                        
            current_rewards = np.array(cum_means[-5:])  # Get the last 5 rewards
            mean_reward = np.mean(current_rewards)
            # var_reward = np.var(current_rewards)
            std_reward = np.std(current_rewards)
            episode_range = f"{i-3}-{i+1}"
            
            print(f"Episodes {episode_range}: Mean reward = {mean_reward:.2f}, Std = {std_reward:.2f}")
            
            # Append to CSV file
            # with open(csv_filename, 'a', newline='') as csvfile:
            #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            #     writer.writerow({
            #         'episode_range': episode_range,
            #         'model_seed': config['model_seed'],
            #         'mean_reward': mean_reward,
            #         'std_reward': std_reward,
            #         'br_reward': best_response_reward[1],
            #         'br_prox(mean)': mean_reward / best_response_reward[1],
            #         'ep_sparse_reward': current_rewards
            #     })
            if USE_WANDB:
                rewards_sofar = np.array(cum_means)
                wandb.log({
                    f"mean_reward_episodes_sofar": np.mean(rewards_sofar),
                    f"std_reward_episodes_sofar": np.std(rewards_sofar),
                    # "episode": i+1
                }, commit=False)
            
    # Log overall statistics
    overall_mean = np.mean(cum_means)
    # overall_var = np.var(cum_means)
    overall_std = np.std(cum_means)

    print(f"======== Overall Stats (Heps={Heps}): Mean={overall_mean:.2f}, Std={overall_std:.2f} ========")

    # Add overall stats to CSV
    # with open(csv_filename, 'a', newline='') as csvfile:
    #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #     writer.writerow({
    #         'episode_range': f"0-{Heps-1}",
    #         'model_seed': config['model_seed'],
    #         'mean_reward': overall_mean,
    #         'std_reward': overall_std,
    #         'br_reward': best_response_reward[1],
    #         'br_prox(mean)': overall_mean / best_response_reward[1],
    #     })
    
    print(f"Statistics saved to {csv_filename}")
    # print(f"======== Reward sum (Heps): {rewards_sum} ========")
    return np.array(cum_means)    

def deploy_online(env, controller, eval_trainer, config, csv_filename, Heps=5, best_response_reward=False):
    ctx_rollouts = config["horizon"] // config["ep_hor"]
    ctx_rollouts = CTX_ROLLOUTS
    # if env.envs[0].layout_name == "random0_medium":
    #     layout_w = 8
    #     layout_h = 5
    # elif env.envs[0].layout_name == "random1":
    #     layout_w = 5
    #     layout_h = 5
    layout_w, layout_h, layout_c = env.observation_space[0].shape

    # layout_c = config["state_dim"] // (layout_w * layout_h)
    horizon = config['ep_hor'] # eval episode horizon, might different from context horizon
    context_states = torch.zeros((ctx_rollouts, horizon, layout_w, layout_h, layout_c)).to(device)
    context_actions = torch.zeros((ctx_rollouts, horizon, config["action_dim"])).to(device)
    context_next_states = torch.zeros((ctx_rollouts, horizon, layout_w, layout_h, layout_c)).to(device)
    context_rewards = torch.zeros((ctx_rollouts, horizon, 1)).to(device)

    cum_means = []
    for i in range(HEPS):
        batch = {
            'context_states': context_states.reshape(-1, layout_w, layout_h, layout_c),
            'context_actions': context_actions.reshape(-1, config["action_dim"]),
            'context_next_states': context_next_states.reshape(-1, layout_w, layout_h, layout_c),
            'context_rewards': context_rewards.reshape(-1, 1),
        }
        controller.set_batch(batch)

        rewards_sum, states_lnr, actions_lnr, rewards_lnr, next_states_lnr = deploy_eval(env, controller, eval_trainer, config, best_response_reward)
        # print(f"======== Reward sum: {rewards_sum} ========")
        cum_means.append(rewards_sum)

        context_states = torch.cat((context_states[1:, :, :, :, :], convert_to_tensor([states_lnr])), dim=0)
        context_actions = torch.cat((context_actions[1:, :, :], convert_to_tensor([actions_lnr])), dim=0)
        context_next_states = torch.cat((context_next_states[1:, :, :, :, :], convert_to_tensor([next_states_lnr])), dim=0)
        context_rewards = torch.cat((context_rewards[1:, :, :], convert_to_tensor([rewards_lnr])), dim=0)

        if (i+1) % 5 == 0:
            current_rewards = np.array(cum_means[-5:])  # Get the last 5 rewards
            mean_reward = np.mean(current_rewards)
            # var_reward = np.var(current_rewards)
            std_reward = np.std(current_rewards)
            episode_range = f"{i-3}-{i+1}"
            
            print(f"Episodes {episode_range}: Mean reward = {mean_reward:.2f}, Std = {std_reward:.2f}")
            
            # Append to CSV file
            with open(csv_filename, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow({
                    'episode_range': episode_range,
                    'model_seed': config['model_seed'],
                    'mean_reward': mean_reward,
                    'std_reward': std_reward,
                    'br_reward': best_response_reward[1],
                    'br_prox(mean)': mean_reward / best_response_reward[1],
                    'ep_sparse_reward': current_rewards
                })
            if USE_WANDB:
                rewards_sofar = np.array(cum_means)
                wandb.log({
                    f"mean_reward_episodes_sofar": np.mean(rewards_sofar),
                    f"std_reward_episodes_sofar": np.std(rewards_sofar),
                    # "episode": i+1
                }, commit=False)
            
    # Log overall statistics
    overall_mean = np.mean(cum_means)
    # overall_var = np.var(cum_means)
    overall_std = np.std(cum_means)

    print(f"======== Overall Stats (Heps={Heps}): Mean={overall_mean:.2f}, Std={overall_std:.2f} ========")

    # Add overall stats to CSV
    with open(csv_filename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({
            'episode_range': f"0-{Heps-1}",
            'model_seed': config['model_seed'],
            'mean_reward': overall_mean,
            'std_reward': overall_std,
            'br_reward': best_response_reward[1],
            'br_prox(mean)': overall_mean / best_response_reward[1],
        })
    
    print(f"Statistics saved to {csv_filename}")
    # print(f"======== Reward sum (Heps): {rewards_sum} ========")
    return np.array(cum_means)    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    common_args.add_dataset_args(parser)
    common_args.add_model_args(parser)
    common_args.add_eval_args(parser)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--use_step_masking', action='store_true', default=False)
    parser.add_argument('--mask_steps_per_episode', type=int, default=0)
    parser.add_argument('--use_curriculum_masking', action='store_true', default=False)
    parser.add_argument('--mask_schedule', type=str, default='linear')
    parser.add_argument('--transformer', type=str, default='gpt2')
    parser.add_argument('--label_smoothing', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_query', type=int, default=1)
    parser.add_argument('--ctx_rollouts', type=int, default=5, help='Context length in episodes (must match training)')
    parser.add_argument('--heps', type=int, default=50, help='Consecutive evaluation episodes per partner')
    parser.add_argument('--seed_type', type=str, default='test', help='Partner split: test, set2, set3, val, train_hsp')
    parser.add_argument('--partner_seeds', type=int, nargs='+', default=None, help='Override the partner list')
    parser.add_argument('--skill_level', type=str, default='final', choices=['final', 'mid'])
    parser.add_argument('--group_prefix', type=str, default=None, help='Prefix of the output csv name')
    parser.add_argument('--eval_mode', type=str, default='online', choices=['online', 'offline'])
    parser.add_argument('--count_events', action='store_true', default=False)
    parser.add_argument('--use_render', action='store_true', default=False)
    parser.add_argument('--ckpt_path', type=str, default=None, help='Load this checkpoint instead of the one built from the hyperparameters')
    coot_config.add_path_args(parser, ('partner_pool_dir', 'model_dir', 'results_dir'))

    args = vars(parser.parse_args())
    print("Args: ", args)

    USE_WANDB = args['wandb']
    USE_RENDER = args['use_render']
    COUNT_EVENTS = args['count_events']
    CTX_ROLLOUTS = args['ctx_rollouts']
    SEED_TYPE = args['seed_type']
    HEPS = args['heps']
    SKILL_LEVEL = args['skill_level']
    EVAL_MODE = args['eval_mode']
    GROUP_PREFIX = args['group_prefix'] or f"{args['model_subdir']}_{SEED_TYPE}_ctx{CTX_ROLLOUTS}"
    partner_pool_dir = coot_config.resolve_path('partner_pool_dir', args['partner_pool_dir'])
    model_dir = coot_config.resolve_path('model_dir', args['model_dir'])
    results_dir = coot_config.resolve_path('results_dir', args['results_dir'])

    n_envs = args['envs']
    layout_name = args['layout_name']
    n_agents = args['agents']
    n_hists = args['hists']
    n_samples = args['samples']
    H = args['H']
    dim = args['dim']
    state_dim = dim
    action_dim = dim
    n_embd = args['embd']
    n_head = args['head']
    n_layer = args['layer']
    lr = args['lr']
    epoch = args['epoch']
    shuffle = args['shuffle']
    dropout = args['dropout']
    var = args['var']
    cov = args['cov']
    test_cov = args['test_cov']
    envname = args['env']
    ep_hor = args['hor']
    n_eval = args['n_eval']
    seed = args['seed']
    lin_d = args['lin_d']
    rollin_type = args["rollin_type"]
    wd = args['wd']
    # dataset_prefix = args['dataset_prefix']
    model_subdir = args['model_subdir']
    eval_model_dir = args['eval_model_dir']
    layout_name = args['layout_name']
    num_query = args['num_query']
    batch_size = args['batch_size']
    transformer = args['transformer']

    layout_cfg = coot_config.load_layout(layout_name)
    seed_range = args['partner_seeds'] or coot_config.get_seeds(layout_name, SEED_TYPE)

    # Create CSV file for logging statistics
    import csv

    csv_dir = os.path.join(results_dir, layout_name)
    os.makedirs(csv_dir, exist_ok=True)  # Create directory if it doesn't exist
    csv_filename = f"{csv_dir}/{GROUP_PREFIX}_ep{epoch}_Heps{HEPS}_{MODEL_NAME}.csv"
    print(f"Evaluating {len(seed_range)} partners {seed_range}, writing {csv_filename}")
    with open(csv_filename, 'a', newline='') as csvfile:
        fieldnames = ['model_seed', 'episode_range', 'mean_reward', 'std_reward', 'br_reward', "br_prox(mean)", 'ep_sparse_reward']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not csvfile.tell():
            writer.writeheader()

    for model_seed in seed_range:
        MODEL_SEED = model_seed

        model_config = {
            'state_dim': state_dim,
            'shuffle': shuffle,
            'lr': lr,
            'wd':wd,
            'dropout': dropout,
            'n_embd': n_embd,
            'n_layer': n_layer,
            'n_head': n_head,
            'n_envs': n_envs,
            'n_hists': n_hists,
            'n_samples': n_samples,
            'horizon': H,
            'dim': dim,
            'seed': seed,
            'n_agents': n_agents,
            'batch_size': batch_size,
            'use_step_masking': args['use_step_masking'],
            'mask_steps_per_episode': args['mask_steps_per_episode'],
            'label_smoothing': args['label_smoothing'],
            'num_query': num_query,
        }

        state_dim = layout_cfg['state_dim']
        overcooked_version = layout_cfg['overcooked_version']
        args_w0 = layout_cfg['eval_w0']
        args_w1 = layout_cfg['eval_w1']
        action_dim = coot_config.ACTION_DIM # Including 4 directions, stay and interact

        filename = build_overcooked_model_filename(envname, model_config)

        config = {
            'layout_name': layout_name,
            'horizon': H,
            'ep_hor': ep_hor,
            'state_dim': state_dim,
            'action_dim': action_dim,
            'n_layer': n_layer,
            'n_embd': n_embd,
            'n_head': n_head,
            'dropout': dropout,
            'test': True,
            'attention_dropout': dropout * 0.7,
            'num_query': num_query,
            'traj_num': 1,
            'model_seed': model_seed,
            'transformer': transformer,
            'ctx_rollouts': CTX_ROLLOUTS,
        }
        # config = {
        #     'layout_name': "random1",
        #     'horizon': 200,
        #     'ep_hor': 200,
        #     'state_dim': 500,
        #     'action_dim': 6,
        #     'n_layer': 4,
        #     'n_embd': 128,
        #     'n_head': 2,
        #     'dropout': 0.3,
        #     'test': True,
        #     'attention_dropout': 0.21,
        #     'num_query': 6,
        #     'traj_num': 1,
        #     'model_seed': 1,
        #     'transformer': "gpt2",
        #     'ctx_rollouts': 5,
        # }

        ego_model = Transformer(config).to(device)
        if args['ckpt_path']:
            model_path = args['ckpt_path']
        elif epoch == 0:
            model_path = f'{model_dir}/{layout_name}/{model_subdir}/{filename}_best.pt'
        else:
            model_path = f'{model_dir}/{layout_name}/{model_subdir}/{filename}_epoch{epoch}.pt'
        checkpoint = torch.load(model_path, map_location=device)

        ego_model.load_state_dict(checkpoint)
        ego_model.eval()

        env_parser = get_config()
        mep_env_parser = get_config()
        # env_args = ['--env', 'Overcooked','--algorithm_name', 'mappo', '--experiment_name', 'test_eval', '--layout_name', 'random1', '--cnn_layers_params', '32,3,1,1 64,3,1,1 32,3,1,1', '--use_recurrent_policy','--episode_length', '150']
        eval_filename_list = filename.split("_")[1:]
        eval_filename = "_".join([s for s in eval_filename_list])
        if USE_RENDER:
            env_args = [
                "--env_name", "Overcooked",
                "--algorithm_name", "mappo",
                "--experiment_name", f"eval{eval_filename}",
                "--layout_name", layout_name,
                "--num_agents", "2",
                "--seed", "1",
                "--n_training_threads", "1",
                "--n_rollout_threads", "1",
                "--dummy_batch_size", "1",
                "--num_mini_batch", "1",
                "--episode_length", str(EPISODE_LENGTH),
                "--num_env_steps", "1e5",
                "--reward_shaping_horizon", "1e5",
                "--overcooked_version", str(overcooked_version),
                "--ppo_epoch", "15",
                "--entropy_coefs", "0.2", "0.05", "0.001",
                "--entropy_coef_horizons", "0", "6e6", "1e7",
                "--w0", args_w0,
                "--w1", args_w1,
                "--share_policy",
                "--cnn_layers_params", "32,3,1,1 64,3,1,1 32,3,1,1",
                "--use_recurrent_policy",
                "--use_proper_time_limits",
                "--save_interval", "20",
                "--log_interval", "5",
                "--use_eval",
                "--n_eval_rollout_threads", "1",
                "--eval_episodes", "2",
                # "--use_wandb",
                "--model_seed_start", "1",
                "--model_seed_end", "36",
                "--model_seed", str(MODEL_SEED),
                "--model_src_dir", eval_model_dir,
                "--use_render",
                "--use_hsp",
            ]
        else:    
            env_args = [
                "--env_name", "Overcooked",
                "--algorithm_name", "mappo",
                "--experiment_name", f"eval{eval_filename}",
                "--layout_name", layout_name,
                "--num_agents", "2",
                "--seed", "1",
                "--n_training_threads", "1",
                "--n_rollout_threads", "1",
                "--dummy_batch_size", "1",
                "--num_mini_batch", "1",
                "--episode_length", str(EPISODE_LENGTH),
                "--num_env_steps", "1e5",
                "--reward_shaping_horizon", "1e5",
                "--overcooked_version", str(overcooked_version),
                "--ppo_epoch", "15",
                "--entropy_coefs", "0.2", "0.05", "0.001",
                "--entropy_coef_horizons", "0", "6e6", "1e7",
                "--w0", args_w0,
                "--w1", args_w1,
                "--share_policy",
                "--cnn_layers_params", "32,3,1,1 64,3,1,1 32,3,1,1",
                "--use_recurrent_policy",
                "--use_proper_time_limits",
                "--save_interval", "20",
                "--log_interval", "5",
                "--use_eval",
                "--n_eval_rollout_threads", "1",
                "--eval_episodes", "2",
                # "--use_wandb",
                "--model_seed_start", "1",
                "--model_seed_end", "36",
                "--model_seed", str(MODEL_SEED),
                "--model_src_dir", eval_model_dir,
                # "--use_render",
                "--use_hsp",
            ]
        env_all_args = parse_args(env_args, env_parser)
        
        w0_all_candidates = env_all_args.w0
        if env_all_args.use_hsp and env_all_args.w0 != "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1":
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
            if env_all_args.w0 != w0_all_candidates:
                env_all_args.w0 = w0_all_candidates
            for s_i, s in enumerate(env_all_args.w0.split(",")):
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
            w0 = w0_candidates[(env_all_args.seed + env_all_args.w0_offset) % len(w0_candidates)]
            env_all_args.w0 = ""
            for s in w0:
                env_all_args.w0 += str(s) + ","
            env_all_args.w0 = env_all_args.w0[:-1]

            w1 = []
            for s in env_all_args.w1.split(","):
                w1.append(parse_value(s))
            w1_candidates = list(map(list, product(*w1)))
            # logger.debug(f"w1_candidates:\n {pprint.pformat(w1_candidates, compact=True, width=200)}")
            w1 = w1_candidates[(env_all_args.seed) % len(w1_candidates)]
            env_all_args.w1 = ""
            for s in w1:
                env_all_args.w1 += str(s) + ","
            env_all_args.w1 = env_all_args.w1[:-1]
        
        from pathlib import Path
        base_run_dir = Path(results_dir) / "runs"
        run_dir = (
            base_run_dir / env_all_args.env_name / env_all_args.layout_name / env_all_args.algorithm_name / env_all_args.experiment_name
        )
        config.update({'run_dir': run_dir})
        if not run_dir.exists():
            os.makedirs(str(run_dir))
        
        if USE_WANDB:
            wandb.init(project="ICMARL_eval", group=env_all_args.layout_name+f"_{GROUP_PREFIX}"+f"_ep{epoch}_{SEED_TYPE}_ctx{CTX_ROLLOUTS}_len{EPISODE_LENGTH}_s{args['seed']}", job_type="evaluation", name=f"eval_seed{env_all_args.model_seed}_{SKILL_LEVEL}_{eval_filename}")

        env = make_eval_env(env_all_args, run_dir,seed)
        mep_w0 = env_all_args.w0
        if USE_RENDER:
            mep_env_args = [
                "--env_name", "Overcooked",
                "--layout_name", layout_name,
                "--algorithm_name", "mep",
                # "--experiment_name", f"mep_test_len{EPISODE_LENGTH}",
                "--experiment_name", f"eval{eval_filename}",
                "--num_agents", "2",
                "--seed", "1",
                "--n_training_threads", "1",
                "--num_mini_batch", "1",
                "--episode_length", "200",
                "--num_env_steps", "1e6",
                "--reward_shaping_horizon", "1e6",
                "--overcooked_version", str(overcooked_version),
                "--n_rollout_threads", "1",
                "--dummy_batch_size", "1",
                "--ppo_epoch", "15",
                "--entropy_coefs", "0.2","0.05","0.01",
                "--entropy_coef_horizons", "0","5e7","1e8",
                "--stage", "2",
                "--mep_use_prioritized_sampling",
                "--mep_prioritized_alpha", "1.5",
                "--save_interval", "25",
                "--log_interval", "1",
                "--use_eval",
                "--eval_interval", "20",
                "--n_eval_rollout_threads", "72",
                "--eval_episodes", "5",
                "--use_centralized_V",

                "--population_size", "36",
                "--adaptive_agent_name", "mep_adaptive",
                "--use_agent_policy_id",
                "--use_proper_time_limits",
                "--use_policy_in_env",
                "--use_render",
                "--eval_w0", mep_w0,
                "--model_seed", str(MODEL_SEED),

                # "--use_hsp",
                # "--store_traj",
            ]
        else:
            mep_env_args = [
                "--env_name", "Overcooked",
                "--layout_name", layout_name,
                "--algorithm_name", "mep",
                # "--experiment_name", f"mep_test_len{EPISODE_LENGTH}",
                "--experiment_name", f"eval{eval_filename}",
                "--num_agents", "2",
                "--seed", "1",
                "--n_training_threads", "1",
                "--num_mini_batch", "1",
                "--episode_length", "200",
                "--num_env_steps", "1e6",
                "--reward_shaping_horizon", "1e6",
                "--overcooked_version", str(overcooked_version),
                "--n_rollout_threads", "1",
                "--dummy_batch_size", "1",
                "--ppo_epoch", "15",
                "--entropy_coefs", "0.2","0.05","0.01",
                "--entropy_coef_horizons", "0","5e7","1e8",
                "--stage", "2",
                "--mep_use_prioritized_sampling",
                "--mep_prioritized_alpha", "1.5",
                "--save_interval", "25",
                "--log_interval", "1",
                "--use_eval",
                "--eval_interval", "20",
                "--n_eval_rollout_threads", "72",
                "--eval_episodes", "5",
                "--use_centralized_V",

                "--population_size", "36",
                "--adaptive_agent_name", "mep_adaptive",
                "--use_agent_policy_id",
                "--use_proper_time_limits",
                "--use_policy_in_env",
                # "--use_render",
                "--eval_w0", mep_w0,
                "--model_seed", str(MODEL_SEED),

                # "--use_hsp",
                # "--store_traj",
            ]
        mep_env_all_args = parse_args(mep_env_args, mep_env_parser)
        mep_run_dir = (
            base_run_dir / mep_env_all_args.env_name / mep_env_all_args.layout_name / "mappo" / mep_env_all_args.experiment_name
        )
        # mep_run_dir = str(mep_run_dir).replace("mep", "mappo", 1)
        mep_env = make_eval_env(mep_env_all_args, mep_run_dir,seed)
        # mdp = OvercookedGridworld.from_layout_name("random1")
        # env = OvercookedEnv(mdp)

        TrainAlgo, Policy = make_trainer_policy_cls('mappo', use_single_network=False)
        
        eval_policy = []
        for agent_id in range(2):
            share_observation_space = (env.share_observation_space[agent_id])
            po = Policy(
                env_all_args,
                env.observation_space[agent_id],
                env.share_observation_space[agent_id],
                env.action_space[agent_id],
                device=device,
            )
            eval_policy.append(po)
        eval_trainer = []
        # model_path0 = f'{render_dir}/hsp_models/{eval_model_dir}/seed{env_all_args.model_seed:02d}/actor_agent0_periodic_18010000.pt'
        # model_path1 = f'{render_dir}/hsp_models/{eval_model_dir}/seed{env_all_args.model_seed:02d}/actor_agent1_periodic_18010000.pt'

        # Evaluation partner (w0, biased agent) and its best response (w1)
        model_path0 = coot_config.hsp_actor_path(layout_name, env_all_args.model_seed, SKILL_LEVEL, 0, partner_pool_dir)
        model_path1 = coot_config.hsp_actor_path(layout_name, env_all_args.model_seed, SKILL_LEVEL, 1, partner_pool_dir)
        for agent_id in range(2):
            tr = TrainAlgo(env_all_args, eval_policy[agent_id], device=device)
            eval_trainer.append(tr)

        #     # eval_model_path = f'{render_dir}/hsp_models/seed01/{eval_model_dir}/actor_agent1_periodic_15010000_seed01.pt'
        #     eval_model_path = f'{render_dir}/hsp_models/{eval_model_dir}/seed01/actor_agent{agent_id}_periodic_11010000.pt'
        
        #     # checkpoint = torch.load(eval_model_path)
        eval_model_state_dict = torch.load(model_path0, map_location=device)
        eval_trainer[0].policy.actor.load_state_dict(eval_model_state_dict)
        eval_model_state_dict = torch.load(model_path1, map_location=device)
        eval_trainer[1].policy.actor.load_state_dict(eval_model_state_dict)

        dataset_config = {'horizon': H, 'dim': dim, 'rollin_type': rollin_type, 'shuffle':False, 'n_hists': n_hists, 'n_samples': n_samples, 'ctx_rollouts': CTX_ROLLOUTS}
        # eval_filepath = build_overcooked_data_filename(envname, 1, 2, dataset_config, layout=layout_name, prefix=dataset_prefix , mode=2)

        # with open(eval_filepath, 'rb') as f:
        #     eval_trajs = pickle.load(f)

        
        lnr_controller = OvercookedController(ego_model, batch_size=n_eval)
        
        Heps = HEPS
        if COUNT_EVENTS:
            # br_rewards, _, _, _, _ = deploy_bestresponse(env, eval_trainer, config)
            print("pass best response")
        else:
            # for i in range(5):
            br_rewards, _, _, _ = deploy_bestresponse(env, eval_trainer, config, determ=True)

        if EVAL_MODE == "offline":
            cum_means_lnr = deploy_offline(mep_env, env, lnr_controller, eval_trainer, config, csv_filename, Heps, False)
        elif EVAL_MODE == "online":
            cum_means_lnr = deploy_online(mep_env, lnr_controller, eval_trainer, config, csv_filename, Heps, br_rewards)


        if USE_WANDB:
            wandb.finish()
    
    # Now read the CSV file to compute interquartile statistics
    results_df = pd.read_csv(csv_filename)

    # Filter for overall results (those with episode_range covering all episodes)
    overall_results = results_df[results_df['episode_range'] == f"0-{HEPS-1}"]

    if len(overall_results) > 0:
        # Calculate quartiles of the overall means
        q1 = overall_results['mean_reward'].quantile(0.25)
        q2 = overall_results['mean_reward'].quantile(0.50)  # median
        q3 = overall_results['mean_reward'].quantile(0.75)
        iqr = q3 - q1
        
        # Get middle 50% of seeds (between Q1 and Q3)
        middle_50_df = overall_results[(overall_results['mean_reward'] >= q1) & 
                                        (overall_results['mean_reward'] <= q3)]
        
        # Calculate statistics on middle 50%
        middle_50_mean = middle_50_df['mean_reward'].mean()
        middle_50_std = middle_50_df['mean_reward'].std()
        # middle_50_br_prox = middle_50_df['br_prox(mean)'].mean() if 'br_prox(mean)' in middle_50_df.columns else None
        
        print("\n===== FINAL STATISTICS ACROSS ALL SEEDS =====")
        print(f"Total seeds evaluated: {len(overall_results)}")
        print(f"Q1 (25th percentile): {q1:.2f}")
        print(f"Median: {q2:.2f}")
        print(f"Q3 (75th percentile): {q3:.2f}")
        print(f"IQR: {iqr:.2f}")
        print(f"Middle 50% mean: {middle_50_mean:.2f}")
        print(f"Middle 50% std: {middle_50_std:.2f}")
        # if middle_50_br_prox is not None:
        #     print(f"Middle 50% BR proximity: {middle_50_br_prox:.4f}")
        print("Seeds in middle 50%:", middle_50_df['model_seed'].tolist())
        
        # Save final statistics to CSV
        # final_stats_file = f"{csv_dir}/{GROUP_PREFIX}_final_stats_{timestamp}.csv"
        final_stats_file = f"{csv_filename[:-4]}_final_stats.csv"
        with open(final_stats_file, 'w', newline='') as f:
            stats_writer = csv.writer(f)
            stats_writer.writerow(['Statistic', 'Value'])
            stats_writer.writerow(['Total Seeds', len(overall_results)])
            stats_writer.writerow(['Q1', f"{q1:.2f}"])
            stats_writer.writerow(['Median', f"{q2:.2f}"])
            stats_writer.writerow(['Q3', f"{q3:.2f}"])
            stats_writer.writerow(['IQR', f"{iqr:.2f}"])
            stats_writer.writerow(['Middle 50% Mean', f"{middle_50_mean:.2f}"])
            stats_writer.writerow(['Middle 50% Std', f"{middle_50_std:.2f}"])
            # if middle_50_br_prox is not None:
            #     stats_writer.writerow(['Middle 50% BR Proximity', f"{middle_50_br_prox:.4f}"])
            stats_writer.writerow(['Middle 50% Seeds', str(middle_50_df['model_seed'].tolist())])
        
        print(f"Final statistics saved to {final_stats_file}")

        # else:
        #     import matplotlib.pyplot as plt
        #     import scipy

        #     means_lnr = np.mean(cum_means_lnr)
        #     os.makedirs(fig_dir, exist_ok=True)
        #     sems_lnr = scipy.stats.sem(cum_means_lnr)

        #     plt.cla()
        #     plt.clf()  # Clear the current figure
        #     plt.plot(means_lnr, label='Learner')
        #     plt.fill_between(np.arange(Heps), means_lnr - sems_lnr,
        #                     means_lnr + sems_lnr, alpha=0.2)
        #     plt.legend()
        #     plt.xlabel('Episodes')
        #     plt.ylabel('Average Return')
        #     plt.title(f'Online Evaluation on 1 Envs')
        #     plt.savefig(f"{fig_dir}/eval_Heps{Heps}.png")
        
        
        