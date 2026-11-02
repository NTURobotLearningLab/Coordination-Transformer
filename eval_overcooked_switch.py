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
import csv

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

# Module-level settings read by the deploy functions below; overwritten from the command line in __main__.
USE_WANDB = False
CTX_ROLLOUTS = 5
SEED_TYPE = "test"
EPISODE_LENGTH = 200
HEPS = 25
SWITCH_EPISODES = 15  # episodes played with the second partner at the end
GROUP_PREFIX = "switch"
SKILL_LEVEL = "final" # mid or final
COUNT_EVENTS = True

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
def ori_to_dir(ori_idx):
    if ori_idx == 0:
        direction = (0,-1)
    elif ori_idx == 1:
        direction = (0,1)
    elif ori_idx == 2:
        direction = (1,0)
    elif ori_idx == 3:
        direction = (-1,0)
    else:
        raise ValueError("wrong orientation index")
    return direction

def obs_to_context(obs):
    obs = obs//255
    pos0 = np.argwhere(obs[:,:,0]==1)
    pos1 = np.argwhere(obs[:,:,1]==1)
    ori0 = np.argwhere(np.array([np.any(obs[:,:,i+2]) for i in range(4)]))
    ori1 = np.argwhere(np.array([np.any(obs[:,:,i+6]) for i in range(4)]))
    ori = np.array([ori0,ori1])
    pot_pos = np.argwhere(obs[:,:,10]==1)
    onion_pot0 = int(obs[pot_pos[0][0],pot_pos[0][1],15])
    onion_pot1 = int(obs[pot_pos[1][0],pot_pos[1][1],15])
    onion_pot = np.array([onion_pot0,onion_pot1])
    cooktime_pot0 = int(obs[pot_pos[0][0],pot_pos[0][1],16])
    cooktime_pot1 = int(obs[pot_pos[1][0],pot_pos[1][1],16])
    cooktime_pot = np.array([cooktime_pot0,cooktime_pot1])
    soup_loc = obs[:,:,17].astype(int).flatten()
    dish_loc = obs[:,:,18].astype(int).flatten()
    onion_loc = obs[:,:,19].astype(int).flatten()

    context_state = np.concatenate([pos0,pos1,ori,onion_pot,cooktime_pot,soup_loc,dish_loc,onion_loc], axis=None)
    return context_state

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

def event_to_num(event_info):
    event_info = event_info[0]
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

def deploy_eval(env, ctrl, trainer, config, best_response_reward=False, count_events=False):
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
    context_events = np.zeros((17))    
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
        
        act1 = ctrl.act(context_state)
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
        if count_events:
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

    episode_rewards_sum_w0 = np.sum(np.array(episode_rewards_w0))
    episode_rewards_sum_w1 = np.sum(np.array(episode_rewards_w1))
    print(f"episode rewards w0: {episode_rewards_sum_w0}")    
    print(f"episode rewards w1: {episode_rewards_sum_w1}")    
    ctrl.reset_query_states()
    # Log rewards_sum to wandb
    if USE_WANDB:
        if best_response_reward != False:

            wandb.log({"episode_rewards_w0": episode_rewards_sum_w0, "episode_rewards_w1": episode_rewards_sum_w1, "BR_approx(w1)": episode_rewards_sum_w1 / best_response_reward[1], "BR_rewards": best_response_reward[1]})
        else:
            wandb.log({"episode_rewards_w0": episode_rewards_sum_w0, "episode_rewards_w1": episode_rewards_sum_w1})
        # traj_gif_path = f"{env.envs[0].run_dir}/gifs/{env.envs[0].layout_name}_seed{env.envs[0].model_seed:02d}/traj_num_{(env.envs[0].traj_num-1)//2}/reward_{int(episode_rewards_sum_w1)}.gif"
        traj_gif_path = f"{config['run_dir']}/gifs/{layout_name}_seed{config['model_seed']:02d}/traj_num{config['traj_num']}/reward_{int(episode_rewards_sum_w1)}.gif"
        # f"{self.run_dir}/gifs/{self.layout_name}/traj_num_{(self.traj_num-1)//2}"
        # wandb.log({"traj_gif": wandb.Video(traj_gif_path, format="gif")})
    config['traj_num'] += 1
    
    if count_events:
        return episode_rewards_sum_w1, context_states[:-1], context_actions, context_rewards, context_states[1:], context_events
    else:
        return episode_rewards_sum_w1, context_states[:-1], context_actions, context_rewards, context_states[1:]

def deploy_bestresponse(env, trainer, config):
    obs, _, avail_actions = env.reset()
    if env.envs[0].layout_name == "random0_medium":
        layout_w = 8
        layout_h = 5
    elif env.envs[0].layout_name == "random1" or env.envs[0].layout_name == "random0":
        layout_w = 5
        layout_h = 5
    elif env.envs[0].layout_name == "random1_m" or env.envs[0].layout_name == "random0_m":
        layout_w = 5
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
    pos0 = np.argwhere(obs[0,0,:,:,0]==255)[0]
    pos1 = np.argwhere(obs[0,0,:,:,1]==255)[0]
    episode_pos = [np.concatenate([pos0,pos1])]
    episode_obs = [obs//255]
    episode_actions = []
    episode_ava_act = [avail_actions]
    for _ in range(episode_length):
        eval_actions = []
        for agent_id in range(2):
            trainer[agent_id].prep_rollout()
            eval_action, eval_rnn_state = trainer[agent_id].policy.act(
                obs[:, agent_id],
                rnn_states[:, agent_id],
                masks[:, agent_id],
                avail_actions[:, agent_id],
                deterministic=True,
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
        obs = np.stack(obs).reshape(1, 2, layout_w, layout_h, -1)
        avail_actions = avail_actions.reshape(1,2,6)

        rnn_states[dones == True] = np.zeros(
            ((dones == True).sum(), 1, 64),
            dtype=np.float32,
        )
        masks = np.ones((1, 2, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        pos0 = np.argwhere(obs[0,0,:,:,0]==255)[0]
        pos1 = np.argwhere(obs[0,0,:,:,1]==255)[0]
        episode_pos.append(np.concatenate([pos0,pos1]))
        episode_actions.append(np.concatenate(eval_actions))
        episode_obs.append(obs//255)
        episode_ava_act.append(avail_actions)
        onion_pots = obs[0,0,3,0,15]//255 + obs[0,0,4,1,15]//255
        episode_rewards_w0.append(rewards[0,0])
        episode_rewards_w1.append(rewards[0,1])
        

    episode_rewards_sum_w0 = np.sum(np.stack(episode_rewards_w0))
    episode_rewards_sum_w1 = np.sum(np.stack(episode_rewards_w1))
    print(f"episode rewards w0: {episode_rewards_sum_w0}")
    print(f"episode rewards w1: {episode_rewards_sum_w1}")    

    if USE_WANDB:
        traj_gif_path = f"{env.envs[0].run_dir}/gifs/{env.envs[0].layout_name}_seed{env.envs[0].model_seed:02d}/traj_num0/reward_{int(episode_rewards_sum_w1)}.gif"
        # wandb.log({"traj_gif": wandb.Video(traj_gif_path, format="gif")})
    return [episode_rewards_sum_w0, episode_rewards_sum_w1]

def deploy_online_switch(env, controller, eval_trainer, eval_trainer_switch, config, Heps=5, target_reward=80):
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
    for i in range(ctx_rollouts):
        batch = {
            'context_states': context_states.reshape(-1, layout_w, layout_h, layout_c),
            'context_actions': context_actions.reshape(-1, config["action_dim"]),
            # 'context_next_states': context_next_states.reshape(-1, layout_w, layout_h, layout_c),
            'context_rewards': context_rewards.reshape(-1, 1),
        }
        controller.set_batch(batch)

        rewards_sum, states_lnr, actions_lnr, rewards_lnr, next_states_lnr = deploy_eval(env, controller, eval_trainer, config)
        print(f"======== Reward sum: {rewards_sum} ========")
        cum_means.append(rewards_sum)

        context_states = torch.cat((context_states[1:, :, :, :, :], convert_to_tensor([states_lnr])), dim=0)
        context_actions = torch.cat((context_actions[1:, :, :], convert_to_tensor([actions_lnr])), dim=0)
        # context_next_states = torch.cat((context_next_states[1:, :, :, :, :], convert_to_tensor([next_states_lnr])), dim=0)
        context_rewards = torch.cat((context_rewards[1:, :, :], convert_to_tensor([rewards_lnr])), dim=0)

        # # Convert to torch
        # context_states[-1, :, :, :, :] = convert_to_tensor([states_lnr])
        # context_actions[-1, :, :] = convert_to_tensor([actions_lnr])
        # context_next_states[-1, :, :, :, :] = convert_to_tensor([next_states_lnr])
        # context_rewards[-1, :, :] = convert_to_tensor([rewards_lnr])
    
    switch_at = Heps - SWITCH_EPISODES  # first episode with the new partner
    for _ in range(ctx_rollouts, switch_at):
        batch = {
            'context_states': context_states.reshape( -1, layout_w, layout_h, layout_c),
            'context_actions': context_actions.reshape( -1, config["action_dim"]),
            # 'context_next_states': context_next_states.reshape( -1, layout_w, layout_h, layout_c),
            'context_rewards': context_rewards.reshape( -1, 1),
        }
        controller.set_batch(batch)


        if _ == switch_at - 1:
            rewards_sum, states_lnr, actions_lnr, rewards_lnr, next_states_lnr, events_lnr_origin = deploy_eval(env, controller, eval_trainer, config, count_events=True)
        else:
            rewards_sum, states_lnr, actions_lnr, rewards_lnr, next_states_lnr = deploy_eval(env, controller, eval_trainer, config)
        cum_means.append(rewards_sum)

        
        # Convert to torch
        states_lnr = convert_to_tensor([states_lnr])
        actions_lnr = convert_to_tensor([actions_lnr])
        # next_states_lnr = convert_to_tensor([next_states_lnr])
        rewards_lnr = convert_to_tensor([rewards_lnr])

        # Roll in new data by shifting the batch and appending the new data.
        context_states = torch.cat(
            (context_states[1:, :, :], states_lnr), dim=0)
        context_actions = torch.cat(
            (context_actions[1:, :, :], actions_lnr), dim=0)
        # context_next_states = torch.cat(
        #     (context_next_states[1:, :, :], next_states_lnr), dim=0)
        context_rewards = torch.cat(
            (context_rewards[1:, :, :], rewards_lnr), dim=0)
        
    adapted = False
    for ep in range(switch_at, Heps):
        batch = {
            'context_states': context_states.reshape( -1, layout_w, layout_h, layout_c),
            'context_actions': context_actions.reshape( -1, config["action_dim"]),
            'context_rewards': context_rewards.reshape( -1, 1),
        }
        controller.set_batch(batch)


        rewards_sum, states_lnr, actions_lnr, rewards_lnr, _, events_lnr_switch = deploy_eval(env, controller, eval_trainer_switch, config, count_events=True)
        cum_means.append(rewards_sum)
        if not adapted and (rewards_sum >= target_reward or ep == Heps-1):
            # First episode after the switch that reaches the target (Table 4)
            adapted = True
            print(f"CooT can adapt within {ep-switch_at} episodes, with reward={int(rewards_sum)}")

            arr1 = events_lnr_origin
            arr2 = events_lnr_switch
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
            event_names = [
                'put_onion_on_X', 'put_dish_on_X', 'put_soup_on_X', 
                'pickup_onion_from_X', 'pickup_onion_from_O', 
                'pickup_dish_from_X', 'pickup_dish_from_D', 
                'pickup_soup_from_X', 'USEFUL_DISH_PICKUP', 
                'SOUP_PICKUP', 'PLACEMENT_IN_POT', 'delivery', 
                'STAY', 'MOVEMENT', 'IDLE_MOVEMENT', 
                'IDLE_INTERACT_X', 'IDLE_INTERACT_EMPTY'
            ]
            # Create directory if it doesn't exist
            switch_dir = os.path.join(results_dir, "switch", layout_name)
            os.makedirs(switch_dir, exist_ok=True)
            
            # Create filename with timestamp
            # timestamp = datetime.now().strftime("%m%d_%H%M%S")
            csv_filename = f"{switch_dir}/switch_events_{config['model_seed']}_to_{config['switch_seed']}.csv"
            
            with open(csv_filename, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                # Write header
                # writer.writerow(['model_seed', model_seed])
                writer.writerow(['Event']+[event_name for event_name in event_names])
                
                # Write data for each event
                writer.writerow(['distribution1']+[f"{dist1[i]:.6f}" for i in range(len(event_names))])
                writer.writerow(['distribution2']+[f"{dist2[i]:.6f}" for i in range(len(event_names))])
        
        # Convert to torch
        states_lnr = convert_to_tensor([states_lnr])
        actions_lnr = convert_to_tensor([actions_lnr])
        # next_states_lnr = convert_to_tensor([next_states_lnr])
        rewards_lnr = convert_to_tensor([rewards_lnr])

        # Roll in new data by shifting the batch and appending the new data.
        context_states = torch.cat(
            (context_states[1:, :, :], states_lnr), dim=0)
        context_actions = torch.cat(
            (context_actions[1:, :, :], actions_lnr), dim=0)
        # context_next_states = torch.cat(
        #     (context_next_states[1:, :, :], next_states_lnr), dim=0)
        context_rewards = torch.cat(
            (context_rewards[1:, :, :], rewards_lnr), dim=0)

    print(f"======== Reward sum (Heps): {rewards_sum} ========")
    return np.array(cum_means)    

if __name__ == "__main__":
    # Table 4: CooT plays with partner A, then the partner is switched to B for the last
    # --switch_episodes episodes; prints how many episodes it takes to reach --target_reward again.
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
    parser.add_argument('--partner_a', type=int, default=17, help='First partner seed')
    parser.add_argument('--partner_b', type=int, default=31, help='Partner seed after the switch')
    parser.add_argument('--both_directions', action='store_true', help='Also run B -> A')
    parser.add_argument('--heps', type=int, default=25, help='Total episodes per run')
    parser.add_argument('--switch_episodes', type=int, default=15, help='Episodes with the second partner')
    parser.add_argument('--target_reward', type=float, default=60)
    parser.add_argument('--skill_level', type=str, default='final', choices=['final', 'mid'])
    parser.add_argument('--use_render', action='store_true', default=False, help='Save gifs of every episode')
    parser.add_argument('--ckpt_path', type=str, default=None)
    coot_config.add_path_args(parser, ('partner_pool_dir', 'model_dir', 'results_dir'))

    args = vars(parser.parse_args())
    print("Args: ", args)

    USE_WANDB = args['wandb']
    CTX_ROLLOUTS = args['ctx_rollouts']
    HEPS = args['heps']
    SWITCH_EPISODES = args['switch_episodes']
    SKILL_LEVEL = args['skill_level']
    assert HEPS - SWITCH_EPISODES > CTX_ROLLOUTS, "--heps must leave more than ctx_rollouts episodes before the switch"
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
    n_embd = args['embd']
    n_head = args['head']
    n_layer = args['layer']
    lr = args['lr']
    epoch = args['epoch']
    shuffle = args['shuffle']
    dropout = args['dropout']
    envname = args['env']
    ep_hor = args['hor']
    n_eval = args['n_eval']
    seed = args['seed']
    rollin_type = args["rollin_type"]
    wd = args['wd']
    model_subdir = args['model_subdir']
    eval_model_dir = args['eval_model_dir']
    num_query = args['num_query']
    batch_size = args['batch_size']
    transformer = args['transformer']

    layout_cfg = coot_config.load_layout(layout_name)
    state_dim = layout_cfg['state_dim']
    overcooked_version = layout_cfg['overcooked_version']
    args_w0 = layout_cfg['eval_w0']
    args_w1 = layout_cfg['eval_w1']
    action_dim = coot_config.ACTION_DIM # Including 4 directions, stay and interact

    pairs = [(args['partner_a'], args['partner_b'])]
    if args['both_directions']:
        pairs.append((args['partner_b'], args['partner_a']))

    for model_seed, switch_seed in pairs:
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
            'switch_seed': switch_seed,
            'transformer': transformer,
            'ctx_rollouts': CTX_ROLLOUTS,
        }

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
        eval_filename_list = filename.split("_")[1:]
        eval_filename = "_".join([s for s in eval_filename_list])
        render_args = ["--use_render"] if args['use_render'] else []
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
                    "--model_seed_start", "1",
                    "--model_seed_end", "36",
                    "--model_seed", str(MODEL_SEED),
                    "--model_src_dir", eval_model_dir,
                    "--use_hsp",
                    ] + render_args
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
                        return np.linspace(l, r, n).tolist()
                elif s.startswith("["):
                    s = s[1:-1]
                    v_s = list(map(float, s.split(":")))
                    return v_s
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
            w0 = w0_candidates[(env_all_args.seed + env_all_args.w0_offset) % len(w0_candidates)]
            env_all_args.w0 = ""
            for s in w0:
                env_all_args.w0 += str(s) + ","
            env_all_args.w0 = env_all_args.w0[:-1]

            w1 = []
            for s in env_all_args.w1.split(","):
                w1.append(parse_value(s))
            w1_candidates = list(map(list, product(*w1)))
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
            wandb.init(project="ICMARL_eval", group=env_all_args.layout_name+f"_{GROUP_PREFIX}"+f"_ep{epoch}_{SEED_TYPE}_ctx{CTX_ROLLOUTS}_len{EPISODE_LENGTH}", job_type="evaluation", name=f"eval_seed{env_all_args.model_seed}_{SKILL_LEVEL}_{eval_filename}")

        env = make_eval_env(env_all_args, run_dir,seed)
        mep_w0 = env_all_args.w0
        mep_env_args = [
            "--env_name", "Overcooked",
            "--layout_name", layout_name,
            "--algorithm_name", "mep",
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
            "--eval_w0", mep_w0,
            "--model_seed", str(MODEL_SEED),
        ] + render_args
        mep_env_all_args = parse_args(mep_env_args, mep_env_parser)
        mep_run_dir = (
            base_run_dir / mep_env_all_args.env_name / mep_env_all_args.layout_name / "mappo" / mep_env_all_args.experiment_name
        )
        mep_env = make_eval_env(mep_env_all_args, mep_run_dir,seed)

        TrainAlgo, Policy = make_trainer_policy_cls('mappo', use_single_network=False)
        
        eval_policy = []
        for agent_id in range(2):
            po = Policy(
                env_all_args,
                env.observation_space[agent_id],
                env.share_observation_space[agent_id],
                env.action_space[agent_id],
                device=device,
            )
            eval_policy.append(po)
        eval_trainer = []
        # Partner A (w0) with its best response (w1); after the switch, partner B replaces w0.
        model_path0 = coot_config.hsp_actor_path(layout_name, model_seed, SKILL_LEVEL, 0, partner_pool_dir)
        model_path1 = coot_config.hsp_actor_path(layout_name, model_seed, SKILL_LEVEL, 1, partner_pool_dir)
        model_path0_switch = coot_config.hsp_actor_path(layout_name, switch_seed, SKILL_LEVEL, 0, partner_pool_dir)
        for agent_id in range(2):
            tr = TrainAlgo(env_all_args, eval_policy[agent_id], device=device)
            eval_trainer.append(tr)

        eval_model_state_dict = torch.load(model_path0, map_location=device)
        eval_trainer[0].policy.actor.load_state_dict(eval_model_state_dict)
        eval_model_state_dict = torch.load(model_path1, map_location=device)
        eval_trainer[1].policy.actor.load_state_dict(eval_model_state_dict)

        import copy
        eval_trainer_switch = copy.deepcopy(eval_trainer)
        eval_model_state_dict = torch.load(model_path0_switch, map_location=device)
        eval_trainer_switch[0].policy.actor.load_state_dict(eval_model_state_dict)

        lnr_controller = OvercookedController(ego_model, batch_size=n_eval)
        
        Heps = HEPS
        cum_means_lnr = deploy_online_switch(mep_env, lnr_controller, eval_trainer, eval_trainer_switch, config, Heps, target_reward=args['target_reward'])

        # Per-episode returns of this run
        out_dir = os.path.join(results_dir, "switch", layout_name)
        os.makedirs(out_dir, exist_ok=True)
        returns_csv = f"{out_dir}/switch_returns_{model_seed}_to_{switch_seed}.csv"
        with open(returns_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['episode', 'partner', 'return'])
            for ep, ret in enumerate(cum_means_lnr):
                writer.writerow([ep, model_seed if ep < Heps - SWITCH_EPISODES else switch_seed, float(ret)])
        print(f"Per-episode returns saved to {returns_csv}")

        if USE_WANDB:
            wandb.finish()
