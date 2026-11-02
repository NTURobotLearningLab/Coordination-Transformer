import pickle

import numpy as np
import torch

from utils import convert_to_tensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Dataset(torch.utils.data.Dataset):
    """Dataset class."""

    def __init__(self, path, config, type):
        self.shuffle = config['shuffle']
        self.horizon = config['horizon']
        self.store_gpu = config['store_gpu']
        self.config = config
        self.ctx_rollouts = config['ctx_rollouts']
        self.agents = config['agents']
        self.n_hists = config['n_hists']
        self.n_samples = config['n_samples']
        # Context shuffling: >1 permutes chunks of this many steps (first chunk kept in place),
        # 1 permutes individual steps across the whole context.
        self.chunk_size = config.get('shuffle_chunk_size', 20)
        # Number of validation partners used for the test loss
        self.n_val_agents = config.get('n_val_agents', 5)

        # if path is not a list
        if not isinstance(path, list):
            path = [path]


        ##### New datasets #####
        self.dataset = {
            'query_states': [],
            'optimal_actions': [],
            'context_states': [],
            'context_actions': [],
            'context_next_states': [],
            'context_rewards': [],
        }
        for p in path:
            context_s_p = p.split('.')[0] + '_context_s.pkl'
            context_a_p = p.split('.')[0] + '_context_a.pkl'
            context_r_p = p.split('.')[0] + '_context_r.pkl'
            query_s_p = p.split('.')[0] + '_query_s.pkl'
            optimal_a_p = p.split('.')[0] + '_optimal_a.pkl'
            
            with open(context_s_p, 'rb') as context_s:
                ctx_s = pickle.load(context_s)
            with open(context_a_p, 'rb') as context_a:
                ctx_a = pickle.load(context_a)
            with open(context_r_p, 'rb') as context_r:
                ctx_r = pickle.load(context_r)
            with open(optimal_a_p, 'rb') as optimal_s:
                opt_a = pickle.load(optimal_s)
            with open(query_s_p, 'rb') as query_s:
                qry_s = pickle.load(query_s)            
            
            # Files hold n_hists contexts per partner and n_samples query tuples per context.
            # Each query tuple is paired with its own copy of the context.
            n_partners_in_file = len(ctx_s) // self.n_hists
            if type == "train":
                n_partners = self.agents
            elif type == "test":
                n_partners = min(self.n_val_agents, n_partners_in_file)
            else:
                raise ValueError("Unseen type")
            if n_partners > n_partners_in_file:
                raise ValueError(f"{p}: asked for {n_partners} partners but the file has {n_partners_in_file}")

            H, S = self.n_hists, self.n_samples
            for i in range(n_partners):
                for j in range(H):
                    c = i * H + j
                    self.dataset['context_states'] += [ctx_s[c] for _ in range(S)]
                    self.dataset['context_rewards'] += [ctx_r[c] for _ in range(S)]
                    self.dataset['context_actions'] += [ctx_a[c] for _ in range(S)]
                    self.dataset['query_states'] += qry_s[c * S:(c + 1) * S]
                    self.dataset['optimal_actions'] += opt_a[c * S:(c + 1) * S]


        # # import gc
        # # # import time
        # self.trajs = []
        # context_states = []
        # context_actions = []
        # context_next_states = []
        # # context_rewards = []
        # query_states = []
        # optimal_actions = []
        # for p in path:
        #     p = p.split('.')[0] + ".pkl"
        #     with open(p,"rb") as f:
        #         self.trajs += pickle.load(f)

        #     # for traj in self.trajs:
        #     #     context_states.append(traj['context_states'])
        #     # with np.load(p, allow_pickle=True) as trajs:
        #     # trajs = np.load(p, allow_pickle=True)['data']
        #     # trajs_list = self.trajs.tolist()
        #     self.dataset = {
        #         'query_states': [],
        #         'query_next_states': [],
        #         'optimal_actions': [],
        #         'context_states': [],
        #         'context_actions': [],
        #         'context_next_states': [],
        #         # 'context_rewards': [],
        #     }
        #     i=0
        #     for traj in self.trajs:
        #         self.dataset['context_states'].append(traj['context_states'])
        #         self.dataset['context_actions'].append(traj['context_actions'])
        #         self.dataset['context_rewards'].append(traj['context_rewards'])
        #         # self.dataset['context_next_states'].append(traj['context_states'][1:] + [np.zeros([5,5,20])])
        #         # self.dataset['context_next_states'][i][-1::-200][:] = np.zeros([5,5,20])
        #         # for ctx in range(7):
        #         #     self.dataset['context_next_states'][i][ctx*200-1] = np.zeros([5,5,20])

        #         self.dataset['query_states'].append(traj['query_state'])
        #         self.dataset['optimal_actions'].append(traj['optimal_action'])
        #         # self.dataset['query_next_states'].append(traj['query_state'][1:] + [np.zeros([5,5,20])])
        #         i+=1


        self.zeros = torch.zeros(
            1024 * (config['state_dim'] * 2 + config['action_dim'] + 1)
        )

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.dataset['query_states'])

    def __getitem__(self, index):
        'Generates one sample of data'
        res = {
            'context_states': convert_to_tensor(self.dataset['context_states'][index], store_gpu=self.store_gpu),
            'context_actions': convert_to_tensor(self.dataset['context_actions'][index], store_gpu=self.store_gpu),
            'context_rewards': convert_to_tensor(self.dataset['context_rewards'][index], store_gpu=self.store_gpu),
            # 'context_next_states': convert_to_tensor(self.dataset['context_next_states'][index], store_gpu=self.store_gpu),
            'query_states': convert_to_tensor(self.dataset['query_states'][index], store_gpu=self.store_gpu),
            # 'query_next_states': convert_to_tensor(self.dataset['query_next_states'][index], store_gpu=self.store_gpu),
            'optimal_actions': convert_to_tensor(self.dataset['optimal_actions'][index], store_gpu=self.store_gpu),
            'zeros': self.zeros,
        }

        if self.shuffle:
            # Get the number of chunks to create
            chunk_size = self.chunk_size
            if chunk_size > 1:
                num_chunks = self.horizon // chunk_size
                new_res = res.copy()
                for ctx_rollout in range(self.ctx_rollouts):
                    # Shuffle the chunks
                    shuffled_chunks = torch.randperm(num_chunks-1)
                    shuffled_chunks += 1
                    
                    # Create a new ordering based on shuffled chunks
                    new_ordering = torch.zeros(self.horizon, dtype=torch.long)
                    new_ordering[0:chunk_size] = torch.arange(0, chunk_size)
                    for i, chunk_idx in enumerate(shuffled_chunks):
                        start_pos = (i+1) * chunk_size
                        end_pos = min((i + 2) * chunk_size, self.horizon)
                        orig_start = chunk_idx * chunk_size
                        orig_end = min((chunk_idx + 1) * chunk_size, self.horizon)
                        
                        # Handle the case where the last chunk might be smaller
                        chunk_length = min(end_pos - start_pos, orig_end - orig_start)
                        new_ordering[start_pos:start_pos+chunk_length] = torch.arange(orig_start, orig_start+chunk_length)
                    
                    # Clone and reorder only the tensors that need reordering
                    start_idx = ctx_rollout*self.horizon
                    end_idx = (ctx_rollout+1)*self.horizon
                    new_res['context_states'][start_idx:end_idx] = res['context_states'][start_idx:end_idx].clone()[new_ordering]
                    new_res['context_actions'][start_idx:end_idx] = res['context_actions'][start_idx:end_idx].clone()[new_ordering]
                    # new_res['context_next_states'][start_idx:end_idx] = res['context_next_states'][start_idx:end_idx].clone()[new_ordering]
                    new_res['context_rewards'][start_idx:end_idx] = res['context_rewards'][start_idx:end_idx].clone()[new_ordering]
                return new_res

            perm = torch.randperm(self.horizon*self.ctx_rollouts)
            new_res = res.copy()  # This only copies the dictionary structure, not the tensors

            # Clone and reorder only the tensors that need reordering
            new_res['context_states'] = res['context_states'].clone()[perm]
            new_res['context_actions'] = res['context_actions'].clone()[perm]
            # new_res['context_next_states'] = res['context_next_states'].clone()[perm]
            new_res['context_rewards'] = res['context_rewards'].clone()[perm]
            return new_res
        return res

# class LazyDataset(torch.utils.data.Dataset):
#     def __init__(self, paths, config):
#         self.paths = paths if isinstance(paths, list) else [paths]
#         self.horizon = config['horizon']
#         self.ctx_rollouts = config['ctx_rollouts']
#         self.store_gpu = config['store_gpu']
#         self.shuffle = config['shuffle']
#         self.state_dim = config['state_dim']
#         self.action_dim = config['action_dim']
#         self.zeros = torch.zeros(1024 * (self.state_dim * 2 + self.action_dim + 1))

#         self.index_map = []  # List of (file_idx, traj_idx)
#         self.traj_offsets = []

#         # Build an index of all (file_idx, traj_idx)
#         for file_idx, path in enumerate(self.paths):
#             trajs = np.load(path, allow_pickle=True)['data'].tolist()
#             for traj_idx in range(len(trajs)):
#                 self.index_map.append((file_idx, traj_idx))

#     def __len__(self):
#         return len(self.index_map)

#     def __getitem__(self, idx):
#         file_idx, traj_idx = self.index_map[idx]
#         traj = np.load(self.paths[file_idx], allow_pickle=True)['data'].tolist()[traj_idx]

#         # Convert to tensors
#         context_states = convert_to_tensor(traj['context_states'], store_gpu=self.store_gpu)
#         context_actions = convert_to_tensor(traj['context_actions'], store_gpu=self.store_gpu)
#         query_states = convert_to_tensor(traj['query_state'], store_gpu=self.store_gpu)
#         optimal_actions = convert_to_tensor(traj['optimal_action'], store_gpu=self.store_gpu)

#         res = {
#             'context_states': context_states,
#             'context_actions': context_actions,
#             'query_states': query_states,
#             'optimal_actions': optimal_actions,
#             'zeros': self.zeros,
#         }

#         if self.shuffle:
#             return self.shuffle_context(res)
#         return res

#     def shuffle_context(self, res):
#         # Your existing chunk-based shuffling code here
#         # Adapted from your current Dataset class
#         ...



# class FastDataset(torch.utils.data.Dataset):
#     def __init__(self, path, config):
#         self.shuffle = config['shuffle']
#         self.horizon = config['horizon']
#         self.store_gpu = config['store_gpu']
#         self.config = config

#         if not isinstance(path, list):
#             path = [path]

#         # Load a single sample to determine the shape
#         with open(path[0], 'rb') as f:
#             first_traj = pickle.load(f)[0]

#         num_samples = 5400  # Total number of samples (update if dynamic)
#         state_shape = first_traj['context_states'][0].shape[0]  # e.g., (150, 500)
#         action_shape = first_traj['context_actions'][0].shape[0]
#         reward_shape = first_traj['context_rewards'][0].shape[0]


#         # Create memory-mapped arrays
#         self.query_states = np.memmap("query_states.dat", dtype=np.float32, mode="w+", shape=(num_samples, state_shape))
#         self.optimal_actions = np.memmap("optimal_actions.dat", dtype=np.float32, mode="w+", shape=(num_samples,) + action_shape)
#         self.context_states = np.memmap("context_states.dat", dtype=np.float32, mode="w+", shape=(num_samples, self.horizon) + state_shape)
#         self.context_actions = np.memmap("context_actions.dat", dtype=np.float32, mode="w+", shape=(num_samples, self.horizon) + action_shape)
#         self.context_next_states = np.memmap("context_next_states.dat", dtype=np.float32, mode="w+", shape=(num_samples, self.horizon) + state_shape)
#         self.context_rewards = np.memmap("context_rewards.dat", dtype=np.float32, mode="w+", shape=(num_samples, self.horizon) + reward_shape)


#         # Load data into memory-mapped arrays
#         self._load_data(path)

#     def _load_data(self, paths):
#         """Load data into memory-mapped arrays."""
#         index = 0
#         for p in paths:
#             with open(p, 'rb') as f:
#                 trajs = pickle.load(f)
#                 for traj in trajs:
#                     self.query_states[index] = traj['query_state']
#                     self.optimal_actions[index] = traj['optimal_action']
#                     self.context_states[index] = traj['context_states']
#                     self.context_actions[index] = traj['context_actions']
#                     self.context_next_states[index] = traj['context_next_states']
#                     self.context_rewards[index] = traj['context_rewards']
#                     index += 1

#     def __len__(self):
#         return len(self.query_states)

#     def __getitem__(self, index):
#         """Fetch a single sample"""
#         res = {
#             'query_states': torch.tensor(self.query_states[index]),
#             'optimal_actions': torch.tensor(self.optimal_actions[index]),
#             'context_states': torch.tensor(self.context_states[index]),
#             'context_actions': torch.tensor(self.context_actions[index]),
#             'context_next_states': torch.tensor(self.context_states[index]),
#             'context_rewards': torch.tensor(self.context_rewards[index]),
            
#         }

#         # if self.shuffle:
#         #     perm = torch.randperm(self.horizon)
#         #     res['context_states'] = res['context_states'][perm]
#         #     res['context_actions'] = res['context_actions'][perm]

#         return res


# class ImageDataset(Dataset):
#     """"Dataset class for image-based data."""

#     def __init__(self, paths, config, transform):
#         config['store_gpu'] = False
#         super().__init__(paths, config)
#         self.transform = transform
#         self.config = config

#         context_filepaths = []
#         query_images = []

#         for traj in self.trajs:
#             context_filepaths.append(traj['context_images'])
#             query_image = self.transform(traj['query_image']).float()
#             query_images.append(query_image)

#         self.dataset.update({
#             'context_filepaths': context_filepaths,
#             'query_images': torch.stack(query_images),
#         })

#     def __getitem__(self, index):
#         'Generates one sample of data'
#         filepath = self.dataset['context_filepaths'][index]
#         context_images = np.load(filepath)
#         context_images = [self.transform(images) for images in context_images]
#         context_images = torch.stack(context_images).float()

#         query_images = self.dataset['query_images'][index]

#         res = {
#             'context_images': context_images,#.to(device),
#             'context_states': self.dataset['context_states'][index],
#             'context_actions': self.dataset['context_actions'][index],
#             'context_next_states': self.dataset['context_next_states'][index],
#             'context_rewards': self.dataset['context_rewards'][index],
#             'query_images': query_images,#.to(device),
#             'query_states': self.dataset['query_states'][index],
#             'optimal_actions': self.dataset['optimal_actions'][index],
#             'zeros': self.zeros,
#         }

#         if self.shuffle:
#             perm = torch.randperm(self.horizon)
#             res['context_images'] = res['context_images'][perm]
#             res['context_states'] = res['context_states'][perm]
#             res['context_actions'] = res['context_actions'][perm]
#             res['context_next_states'] = res['context_next_states'][perm]
#             res['context_rewards'] = res['context_rewards'][perm]

#         return res
