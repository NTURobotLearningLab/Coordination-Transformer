import torch
from loguru import logger

from zsceval.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor, R_Critic
from zsceval.utils.util import update_linear_schedule
import torch.nn as nn

class ExDataParallel(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
        
class PEARLContextEncoder(nn.Module):
    def __init__(self, input_dim, z_dim, hidden_dim=128, num_layers=2):
        super().__init__()
        layers = []
        last_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU())
            last_dim = hidden_dim
        self.encoder = nn.Sequential(*layers)
        self.mean = nn.Linear(hidden_dim, z_dim)
        self.logvar = nn.Linear(hidden_dim, z_dim)

    def forward(self, context_batch):
        # context_batch: [N, input_dim]  e.g. obs+action+reward+next_obs
        h = self.encoder(context_batch)
        mean = self.mean(h)
        logvar = self.logvar(h)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z, mean, logvar

class PEARLPolicy:
    def __init__(self, args, obs_space, share_obs_space, act_space, device=torch.device('cpu'), z_dim=0):
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay
        self.device = device
        self.z_dim = z_dim

        self.obs_space = obs_space
        self.share_obs_space = share_obs_space
        self.act_space = act_space
        self.latent_z = None  # Current inferred latent variable
        self.context_buffer = []  # Store past transitions for context inference

        self.actor = R_Actor(args, self.obs_space, self.act_space, self.device)
        self.critic = R_Critic(args, self.share_obs_space, self.device)
        self.context_encoder = PEARLContextEncoder()

        
    def reset_latent(self):
        self.latent_z = None
        self.context_buffer = []

    def store_context(self, obs, act, rew, next_obs):
        # store a single transition
        transition = torch.cat([obs, act, rew, next_obs], dim=-1)
        self.context_buffer.append(transition)

    def sample_z(self):
        if len(self.context_buffer) == 0:
            return torch.zeros(self.z_dim).to(self.device)

        context_batch = torch.stack(self.context_buffer).to(self.device)
        z, mean, logvar = self.context_encoder(context_batch)
        self.latent_z = z.mean(dim=0, keepdim=True)  # or sample multiple z
        return self.latent_z

    def get_actions(
        self,
        share_obs,
        obs,
        rnn_states_actor,
        rnn_states_critic,
        masks,
        available_actions=None,
        deterministic=False,
        task_id=None,
        **kwargs,
    ):
        # Make sure latent z is available
        if self.latent_z is None:
            self.sample_z()

        actions, action_log_probs, rnn_states_actor = self.actor(
            obs, rnn_states_actor, masks, available_actions, deterministic
        )
        values, rnn_states_critic = self.critic(share_obs, rnn_states_critic, masks, task_id=task_id)
        
        # return self.actor(
        #     obs, rnn_states, masks, available_actions, deterministic, latent_z=self.latent_z
        # )
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic
        

    def evaluate_actions(
        self,
        share_obs,
        obs,
        rnn_states_actor,
        rnn_states_critic,
        action,
        masks,
        available_actions=None,
        active_masks=None,
        task_id=None,
        latent_z=None,
    ):
        (
            action_log_probs,
            dist_entropy,
            policy_values,
        ) = self.actor.evaluate_actions(obs, rnn_states_actor, action, masks, available_actions, active_masks)
        values, _ = self.critic(share_obs, rnn_states_critic, masks, task_id=task_id)
        return values, action_log_probs, dist_entropy, policy_values

    def get_values(self, obs, rnn_states, masks):
        return self.critic(obs, rnn_states, masks, latent_z=self.latent_z)

    
class R_MAPPOPolicy:
    def __init__(self, args, obs_space, share_obs_space, act_space, device=torch.device("cpu")):
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.obs_space = obs_space
        self.share_obs_space = share_obs_space
        self.act_space = act_space

        self.data_parallel = getattr(args, "data_parallel", False)

        self.actor = R_Actor(args, self.obs_space, self.act_space, self.device)
        self.critic = R_Critic(args, self.share_obs_space, self.device)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.critic_lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )

    def to_parallel(self):
        if self.data_parallel:
            logger.warning(
                f"Use Data Parallel for Forwarding in devices {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}"
            )
            for name, children in self.actor.named_children():
                setattr(self.actor, name, ExDataParallel(children))
            for name, children in self.critic.named_children():
                setattr(self.critic, name, ExDataParallel(children))

    def lr_decay(self, episode, episodes):
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    def get_actions(
        self,
        share_obs,
        obs,
        rnn_states_actor,
        rnn_states_critic,
        masks,
        available_actions=None,
        deterministic=False,
        task_id=None,
        **kwargs,
    ):
        actions, action_log_probs, rnn_states_actor = self.actor(
            obs, rnn_states_actor, masks, available_actions, deterministic
        )
        values, rnn_states_critic = self.critic(share_obs, rnn_states_critic, masks, task_id=task_id)
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic
        # return actions, action_log_probs, rnn_states_actor

    def get_values(self, share_obs, rnn_states_critic, masks, task_id=None):
        values, _ = self.critic(share_obs, rnn_states_critic, masks, task_id=task_id)
        return values

    def evaluate_actions(
        self,
        share_obs,
        obs,
        rnn_states_actor,
        rnn_states_critic,
        action,
        masks,
        available_actions=None,
        active_masks=None,
        task_id=None,
    ):
        (
            action_log_probs,
            dist_entropy,
            policy_values,
        ) = self.actor.evaluate_actions(obs, rnn_states_actor, action, masks, available_actions, active_masks)
        values, _ = self.critic(share_obs, rnn_states_critic, masks, task_id=task_id)
        return values, action_log_probs, dist_entropy, policy_values

    def evaluate_transitions(
        self,
        share_obs,
        obs,
        rnn_states_actor,
        rnn_states_critic,
        action,
        masks,
        available_actions=None,
        active_masks=None,
        task_id=None,
    ):
        (
            action_log_probs,
            dist_entropy,
            policy_values,
            rnn_states_actor,
        ) = self.actor.evaluate_transitions(obs, rnn_states_actor, action, masks, available_actions, active_masks)
        values, _ = self.critic(share_obs, rnn_states_critic, masks, task_id=task_id)
        return values, action_log_probs, dist_entropy, policy_values, rnn_states_actor

    def act(
        self,
        obs,
        rnn_states_actor,
        masks,
        available_actions=None,
        deterministic=False,
        **kwargs,
    ):
        actions, _, rnn_states_actor = self.actor(obs, rnn_states_actor, masks, available_actions, deterministic)
        return actions, rnn_states_actor

    def get_probs(self, obs, rnn_states_actor, masks, available_actions=None):
        action_probs, rnn_states_actor = self.actor.get_probs(
            obs, rnn_states_actor, masks, available_actions=available_actions
        )
        return action_probs, rnn_states_actor

    def get_action_log_probs(
        self,
        obs,
        rnn_states_actor,
        action,
        masks,
        available_actions=None,
        active_masks=None,
    ):
        action_log_probs, _, _, rnn_states_actor = self.actor.get_action_log_probs(
            obs, rnn_states_actor, action, masks, available_actions, active_masks
        )
        return action_log_probs, rnn_states_actor

    def load_checkpoint(self, ckpt_path):
        if "actor" in ckpt_path:
            self.actor.load_state_dict(torch.load(ckpt_path["actor"], map_location=self.device))
        if "critic" in ckpt_path:
            self.critic.load_state_dict(torch.load(ckpt_path["critic"], map_location=self.device))

    def to(self, device):
        self.actor.to(device)
        self.critic.to(device)

    def prep_rollout(self):
        self.actor.eval()
        self.critic.eval()
