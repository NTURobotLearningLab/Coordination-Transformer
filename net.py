import torch
import torch.nn as nn
import transformers
transformers.set_seed(0)
from transformers import Qwen2Config, Qwen2Model, GPT2Config, GPT2Model
from IPython import embed
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Transformer(nn.Module):
    """Transformer class."""

    def __init__(self, config):
        super(Transformer, self).__init__()
        # self.model = "qwen2"
        self.config = config
        self.model = config['transformer'] 
        self.test = config['test']
        self.horizon = self.config['horizon']
        self.n_embd = self.config['n_embd'] # 256
        self.n_layer = self.config['n_layer'] # 4
        self.n_head = self.config['n_head'] # 4
        self.state_dim = self.config['state_dim'] # 500
        self.action_dim = self.config['action_dim'] # 6
        self.dropout = self.config['dropout'] # 0.3
        self.attention_dropout = self.config['attention_dropout']
        self.num_query = self.config['num_query']
        self.ctx_rollouts = self.config['ctx_rollouts']
        self.reward_data = True

        if self.model == "gpt2":
            config = GPT2Config(
                n_positions=self.ctx_rollouts * self.horizon + self.num_query,
                n_embd=self.n_embd,
                n_layer=self.n_layer,
                n_head=self.n_head,
                resid_pdrop=self.dropout,
                embd_pdrop=self.dropout,
                attn_pdrop=self.attention_dropout,
                use_cache=False,
            )
            self.transformer = GPT2Model(config)
        elif self.model == "qwen2":
            # # No tuning
            # config = Qwen2Config(
            #     hidden_size=256,
            #     intermediate_size=1024,
            #     num_hidden_layers=4,
            #     num_attention_heads=4,
            #     num_key_value_heads=4,
            #     hidden_act='gelu',
            #     max_position_embeddings=4 * (1 + self.horizon),
            #     attention_dropout=0.2,
            #     rope_theta=8000.0,
            # )
            
            # Tuning 2
            config = Qwen2Config(
                hidden_size=128,
                intermediate_size=512,
                num_hidden_layers=3,
                num_attention_heads=1,
                num_key_value_heads=1,
                hidden_act='gelu',
                max_position_embeddings=4 * (1 + self.horizon),
                attention_dropout=0.3,
                rope_theta=8000.0,
            )
            self.transformer = Qwen2Model(config)
        if self.reward_data:
            self.embed_transition = nn.Linear(
                self.state_dim + self.action_dim + 1, self.n_embd)
        else:
            self.embed_transition = nn.Linear(
                self.state_dim + self.state_dim + self.action_dim, self.n_embd)
        self.embed_ln = nn.LayerNorm(self.n_embd)
        self.pred_actions = nn.Linear(self.n_embd, self.action_dim)

    def forward(self, x):
        if x['query_states'].dim() == 1:
            query_states = x['query_states'][None, :]
            zeros = x['zeros'][None, :]
            # if not x['context_states'].numel():
            #     x['context_states'] = query_states
            # state_seq = torch.cat([query_states, x['context_states']], dim=0)
            # action_seq = torch.cat(
            #     [zeros[:, :self.action_dim], x['context_actions']], dim=0)
            # next_state_seq = torch.cat(
            #     [zeros[:, :self.state_dim], x['context_next_states']], dim=0)
            # reward_seq = torch.cat([zeros[:, :1], x['context_rewards']], dim=0)
            state_seq = torch.cat([x['context_states'], query_states], dim=0)
            action_seq = torch.cat(
                [x['context_actions'], zeros[:, :self.action_dim]], dim=0)
            next_state_seq = torch.cat(
                [x['context_next_states'], zeros[:, :self.state_dim]], dim=0)
            if self.reward_data:
                reward_seq = torch.cat([x['context_rewards'], zeros[:, :1]], dim=0)
                seq = torch.cat([state_seq, action_seq, reward_seq], dim=1)
            else:
                seq = torch.cat([state_seq, next_state_seq, action_seq], dim=1)
            stacked_inputs = self.embed_transition(seq)
            transformer_outputs = self.transformer(inputs_embeds=stacked_inputs)
            preds = self.pred_actions(transformer_outputs['last_hidden_state'])

            if self.test:
                return preds[-1, :]
            # return preds[:, :], state_seq, action_seq, next_state_seq, reward_seq
            return preds

        else:
            
            # state_max_len = state_len * state_w * state_h * state_c
            # zeros = x['zeros'][:, :state_max_len]
            # return preds[:, -1, :]
            if torch.any(x['context_states'] == 255) or torch.any(x['query_states'] == 255):
                context_states = x['context_states'][None, :]//255
                # context_next_states = x['context_next_states'][None, :]//255
                query_states = x['query_states'][None, :]//255
            else:
                context_states = x['context_states'][None, :]
                context_next_states = x['context_next_states'][None, :]
                query_states = x['query_states'][None, :]

            if self.test:
                batch_size = 1
                state_len, state_w, state_h, state_c = x['context_states'].shape
                context_states = context_states.reshape(batch_size, state_len, -1)
                # context_next_states = context_next_states.reshape(batch_size, state_len, -1)
                context_actions = x['context_actions'][None, :].reshape(batch_size, state_len, self.action_dim)
                if self.reward_data:
                    context_rewards = x['context_rewards'][None, :].reshape(batch_size, state_len, 1)
                query_states = query_states.reshape(batch_size, self.num_query, -1)
                # zeros = x['zeros'].reshape(batch_size, self.num_query, -1)
                zeros = torch.zeros_like(query_states)
                assert query_states.shape[2] == context_states.shape[2]
                
                # if not x['context_states'].numel():
                #     x['context_states'] = query_states
                state_seq = torch.cat([context_states, query_states], dim=1)
                action_seq = torch.cat(
                    [context_actions, zeros[:, :, :self.action_dim]], dim=1)
                # next_state_seq = torch.cat(
                #     [context_next_states, zeros[:, :, :self.state_dim]], dim=1)
                if self.reward_data:
                    reward_seq = torch.cat([context_rewards, zeros[:, :, :1]], dim=1)
                    seq = torch.cat([state_seq, action_seq, reward_seq], dim=2)
                else:
                    seq = torch.cat([state_seq, next_state_seq, action_seq], dim=2)
                stacked_inputs = self.embed_transition(seq)
                transformer_outputs = self.transformer(inputs_embeds=stacked_inputs)
                preds = self.pred_actions(transformer_outputs['last_hidden_state'])

                return preds[:, -1, :]
            else:
                batch_size, state_len, state_w, state_h, state_c = x['context_states'].shape
                # state_max_len = state_len * state_w * state_h * state_c
                # zeros = x['zeros'][:, :state_max_len]
                # context_states = context_states.permute(0, 1, 4, 2, 3)
                # context_next_states = context_next_states.permute(0, 1, 4, 2, 3)
                # query_states = query_states.permute(0, 1, 4, 2, 3)
                context_states = context_states.reshape(batch_size, state_len, -1)
                # context_next_states = context_next_states.reshape(batch_size, state_len, -1)
                query_states = query_states.reshape(batch_size, self.num_query, -1)
                # zeros = x['zeros'].reshape(batch_size, self.num_query, -1)
                zeros = torch.zeros_like(query_states)

                # assert query_states.shape[2] == context_states.shape[2]
                if query_states.shape[2] != context_states.shape[2]:
                    print(f"query state shape: {query_states.shape}")
                    print(f"context state shape: {context_states.shape}")

                # if not x['context_states'].numel():
                #     x['context_states'] = query_states
                state_seq = torch.cat([context_states, query_states], dim=1)
                action_seq = torch.cat(
                    [x['context_actions'], zeros[:, :, :self.action_dim]], dim=1)
                # next_state_seq = torch.cat(
                #     [context_next_states, zeros[:, :, :self.state_dim]], dim=1)
                if self.reward_data:
                    reward_seq = torch.cat([x['context_rewards'], zeros[:, :, :1]], dim=1)
                    seq = torch.cat([state_seq, action_seq, reward_seq], dim=2)
                else:
                    seq = torch.cat([state_seq, next_state_seq, action_seq], dim=2)
                stacked_inputs = self.embed_transition(seq)
                transformer_outputs = self.transformer(inputs_embeds=stacked_inputs)
                preds = self.pred_actions(transformer_outputs['last_hidden_state'])

                return preds[:, :, :]


class ImageTransformer(Transformer):
    """Transformer class for image-based data."""

    def __init__(self, config):
        super().__init__(config)
        self.im_embd = 8

        size = self.config['image_size']
        size = (size - 3) // 2 + 1
        size = (size - 3) // 1 + 1

        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Flatten(start_dim=1),
            nn.Linear(int(16 * size * size), self.im_embd),
            nn.ReLU(),
        )

        new_dim = self.im_embd + self.state_dim + self.action_dim + 1
        self.embed_transition = torch.nn.Linear(new_dim, self.n_embd)
        self.embed_ln = nn.LayerNorm(self.n_embd)

    def forward(self, x):
        query_images = x['query_images'][:, None, :]
        query_states = x['query_states'][:, None, :]
        context_images = x['context_images']
        context_states = x['context_states']
        context_actions = x['context_actions']
        context_rewards = x['context_rewards']

        if len(context_rewards.shape) == 2:
            context_rewards = context_rewards[:, :, None]

        batch_size = query_states.shape[0]

        image_seq = torch.cat([query_images, context_images], dim=1)
        image_seq = image_seq.view(-1, *image_seq.size()[2:])

        image_enc_seq = self.image_encoder(image_seq)
        image_enc_seq = image_enc_seq.view(batch_size, -1, self.im_embd)

        context_states = torch.cat([query_states, context_states], dim=1)
        context_actions = torch.cat([
            torch.zeros(batch_size, 1, self.action_dim).to(device),
            context_actions,
        ], dim=1)
        context_rewards = torch.cat([
            torch.zeros(batch_size, 1, 1).to(device),
            context_rewards,
        ], dim=1)

        stacked_inputs = torch.cat([
            image_enc_seq,
            context_states,
            context_actions,
            context_rewards,
        ], dim=2)
        stacked_inputs = self.embed_transition(stacked_inputs)
        stacked_inputs = self.embed_ln(stacked_inputs)

        transformer_outputs = self.transformer(inputs_embeds=stacked_inputs)
        preds = self.pred_actions(transformer_outputs['last_hidden_state'])

        if self.test:
            return preds[:, -1, :]
        return preds[:, 1:, :]
