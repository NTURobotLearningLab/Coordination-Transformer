import torch.multiprocessing as mp
if mp.get_start_method(allow_none=True) is None:
    mp.set_start_method('spawn', force=True)  # or 'forkserver'

import argparse
import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:16"

import time
from IPython import embed
import wandb

import matplotlib.pyplot as plt
import torch
import gc
gc.collect()
torch.cuda.empty_cache()
from torchvision.transforms import transforms

import numpy as np
import common_args
import random
from dataset import Dataset
from net import Transformer, ImageTransformer
import coot_config
from utils import (
    build_bandit_data_filename,
    build_bandit_model_filename,
    build_linear_bandit_data_filename,
    build_linear_bandit_model_filename,
    build_darkroom_data_filename,
    build_darkroom_model_filename,
    build_miniworld_data_filename,
    build_miniworld_model_filename,
    build_overcooked_data_filename,
    build_overcooked_model_filename,
    worker_init_fn,
)

# os.environ["CUDA_AVAILABLE_DEVICES"] = "1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Context length in episodes; set from --ctx_rollouts in __main__ (read by apply_step_masking)
CTX_ROLLOUTS = 5


def apply_step_masking(batch, args, epoch, schedule='linear'):
    """
    Apply masking to specific steps in episodes, targeting only certain components.
    
    Args:
        batch: Dictionary containing batch data
        args: Dictionary of arguments controlling masking behavior
    
    Returns:
        batch: Dictionary with masked data
    """
    if not args.get('use_step_masking', False):
        return batch
    
    if args.get('use_curriculum_masking', False):
        min_mask_steps = 10 * (CTX_ROLLOUTS)
        max_mask_steps = args['mask_steps_per_episode'] * (CTX_ROLLOUTS)
        progress = epoch / args['num_epochs']
        if schedule == 'linear':
            rate = progress
        elif schedule == 'exponential':
            rate = progress ** 2
        elif schedule == 'logarithmic':
            rate = 1 - (1 - progress) ** 2
        else:
            raise ValueError(f"Invalid schedule: {schedule}")
        
        mask_steps_per_episode = int(min_mask_steps + (max_mask_steps - min_mask_steps) * rate)
    else:
        mask_steps_per_episode = args['mask_steps_per_episode'] * (CTX_ROLLOUTS)
    
    # Get batch dimensions
    batch_size = batch['context_states'].size(0)
    context_length = batch['context_actions'].size(1)  # Number of steps in episode
    
    # Create a copy of the batch to avoid modifying the original
    masked_batch = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    
    # For each sequence in the batch, randomly select steps to mask
    for b in range(batch_size):
        # Randomly select steps to mask (indices in the context sequence)
        steps_to_mask = torch.randperm(context_length)[:mask_steps_per_episode]
        
        # Apply masking to context_actions
        if 'context_actions' in masked_batch:
            masked_batch['context_actions'][b, steps_to_mask] = 0.0
        
        # Apply masking to context_rewards
        if 'context_rewards' in masked_batch:
            masked_batch['context_rewards'][b, steps_to_mask] = 0.0
        
        # Apply masking to context_next_states
        if 'context_next_states' in masked_batch:
            masked_batch['context_next_states'][b, steps_to_mask] = 0.0
    
    return masked_batch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    common_args.add_dataset_args(parser)
    common_args.add_model_args(parser)
    common_args.add_train_args(parser)

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay for regularization')
    parser.add_argument('--dropout_rate', type=float, default=0.25, help='Dropout rate')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='Label smoothing factor')
    parser.add_argument('--grad_clip', type=float, default=0.25, help='Gradient clipping norm')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--prune_dataset', action='store_true', help='Enable dataset pruning')
    parser.add_argument('--prune_ratio', type=float, default=0.1, help='Ratio of data to prune')
    parser.add_argument('--mixup', action='store_true', help='Enable mixup augmentation')
    parser.add_argument('--mixup_alpha', type=float, default=0.2, help='Mixup alpha parameter')
    parser.add_argument('--use_step_masking', action='store_true', help='Enable masking of specific steps in episodes')
    parser.add_argument('--mask_steps_per_episode', type=int, default=5, help='Number of steps to mask in each episode')
    parser.add_argument('--use_curriculum_masking', action='store_true', help='Enable curriculum masking of steps in episodes')
    parser.add_argument('--num_query', type=int, default=1, help='Number of query states in each data')
    parser.add_argument('--mask_schedule', type=str, default='linear', help='Mask schedule')
    parser.add_argument('--transformer', type=str, default='gpt2', help='Transformer model')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--ctx_rollouts', type=int, default=5, help='Context length in episodes (must match the dataset)')
    parser.add_argument('--shuffle_chunk_size', type=int, default=20,
                        help='With --shuffle: permute chunks of this many steps (Table 16 chunk-wise = 20, step-wise = 1)')
    parser.add_argument('--n_val_agents', type=int, default=5, help='Validation partners used for the test loss')
    coot_config.add_path_args(parser, ('dataset_dir', 'model_dir'))

    args = vars(parser.parse_args())
    print("Args: ", args)
    CTX_ROLLOUTS = args['ctx_rollouts']
    dataset_dir = coot_config.resolve_path('dataset_dir', args['dataset_dir'])
    model_dir = coot_config.resolve_path('model_dir', args['model_dir'])

    env = args['env']
    layout = args['layout_name']
    n_envs = args['envs']
    n_agents = args['agents']
    n_hists = args['hists']
    n_samples = args['samples']
    horizon = args['H']
    dim = args['dim']
    state_dim = dim
    action_dim = dim
    n_embd = args['embd']
    n_head = args['head']
    n_layer = args['layer']
    lr = args['lr']
    shuffle = args['shuffle']
    dropout = args['dropout']
    var = args['var']
    cov = args['cov']
    num_epochs = args['num_epochs']
    seed = args['seed']
    lin_d = args['lin_d']
    multi_dataset = args['multi_dataset']
    wd = args['wd']
    batch_size = args['batch_size']
    
    tmp_seed = seed
    if seed == -1:
        tmp_seed = 0


    torch.manual_seed(tmp_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(tmp_seed)
        torch.cuda.manual_seed_all(tmp_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(tmp_seed)
    random.seed(tmp_seed)

    if args['wandb']:
        wandb.init(
            project="ICMARL_train", 
            config=args,
            name= env + "_" + layout + f"_{n_agents}agent_hist{n_hists}_H{horizon}_samp{n_samples}_embd{n_embd}_layer{n_layer}_seed{seed}",
            group=layout,
            job_type="training"
        )  # Add this line

    
    dataset_config = {
        'n_hists': n_hists,
        'n_samples': n_samples,
        'horizon': horizon,
        'dim': dim,
    }
    model_config = {
        'shuffle': shuffle,
        'lr': lr,
        'wd': wd,
        'dropout': dropout,
        'n_embd': n_embd,
        'n_layer': n_layer,
        'n_head': n_head,
        'n_envs': n_envs,
        'n_hists': n_hists,
        'n_samples': n_samples,
        'horizon': horizon,
        'dim': dim,
        'seed': seed,
    }
    if env == 'bandit':
        state_dim = 1

        dataset_config.update({'var': var, 'cov': cov, 'type': 'uniform'})
        path_train = build_bandit_data_filename(
            env, n_envs, dataset_config, mode=0)
        path_test = build_bandit_data_filename(
            env, n_envs, dataset_config, mode=1)

        model_config.update({'var': var, 'cov': cov})
        filename = build_bandit_model_filename(env, model_config)

    elif env == 'overcooked':
        # Set state_dim and action_dim for Overcooked
        state_dim = coot_config.load_layout(layout)['state_dim']
        action_dim = coot_config.ACTION_DIM  # Including 4 directions, stay, and interact
        
        dataset_config.update({
            'rollin_type': 'expert', 
            'dataset_prefix': args['dataset_prefix'],
            'increment': args['increment'],
            'shuffle': False,
            'batch_size': args['batch_size'],
            'ctx_rollouts': CTX_ROLLOUTS,
            # 'augment': args['augment'],  # Add augmentation flag
        })
        
        model_config.update({
            'n_agents': n_agents, 
            'batch_size': args['batch_size'], 
            'use_step_masking': args['use_step_masking'], 
            'mask_steps_per_episode': args['mask_steps_per_episode'],
            'label_smoothing': args['label_smoothing'],
            'num_query': args['num_query'],
        })

        # One train file and one test file covering partners 1..agents
        path_train = build_overcooked_data_filename(
            env, 1, n_agents, dataset_config, layout=layout, mode=0, prefix=dataset_config['dataset_prefix'], root=dataset_dir)
        path_test = build_overcooked_data_filename(
            env, 1, n_agents, dataset_config, layout=layout, mode=1, prefix=dataset_config['dataset_prefix'], root=dataset_dir)
        paths_train = [path_train]
        paths_test = [path_test]
        print(f"train dataset: {path_train}")
        print(f"test dataset: {path_test}")

        filename = build_overcooked_model_filename(env, model_config)
        print(f"Generate filename: {filename}")

    else:
        raise NotImplementedError

    config = {
        'horizon': horizon,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'n_layer': n_layer,
        'n_embd': n_embd,
        'n_head': n_head,
        'shuffle': shuffle,
        'dropout': dropout,
        'test': False,
        'store_gpu': True,
        'layer_norm_eps': 1e-5,
        'use_residual_dropout': True,
        'attention_dropout': dropout * 0.7,
        'num_query': args['num_query'],
        'transformer': args['transformer'],
        'ctx_rollouts': CTX_ROLLOUTS,
        'agents':args['agents'],
        'n_hists': n_hists,
        'n_samples': n_samples,
        'shuffle_chunk_size': args['shuffle_chunk_size'],
        'n_val_agents': args['n_val_agents'],
    }
    if env == 'miniworld':
        config.update({'image_size': 25, 'store_gpu': False})
        model = ImageTransformer(config).to(device)
    else:
        model = Transformer(config).to(device)

    params = {
        'batch_size': args['batch_size'],
        'shuffle': True,
    }
    test_params = {
        'batch_size': args['batch_size'],
        'shuffle': True,
    }

    # Checkpoints, loss log and loss plot go to {model_dir}/{layout}/{model_subdir}/,
    # which is where eval_mean_var.py looks for them.
    save_dir = os.path.join(model_dir, layout, args['model_subdir'])
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving to {save_dir}")
    log_filename = f'{save_dir}/{filename}_logs.txt'
    with open(log_filename, 'w') as f:
        pass
    def printw(string):
        """
        A drop-in replacement for print that also writes to a log file.
        """
        # Use the standard print function to print to the console
        print(string)

        # Write the same output to the log file
        with open(log_filename, 'a') as f:
            print(string, file=f)


    train_dataset = Dataset(paths_train, config, type="train")
    test_dataset = Dataset(paths_test, config, type="test")
    # train_dataset = LazyDataset(path_train, config)
    # test_dataset = LazyDataset(path_test, config)


    # train_loader = torch.utils.data.DataLoader(train_dataset, **params)
    # test_loader = torch.utils.data.DataLoader(test_dataset, **params)

    # optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    # Dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset, **params)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_params)

    # Optimizer with adjusted weight decay and betas
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.95))

    # More sophisticated learning rate scheduler
    num_warmup_steps = int(.1 * num_epochs)  # 10% of total epochs for warmup
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_epochs - current_step) / float(max(1, num_epochs - num_warmup_steps))
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Loss function
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum', label_smoothing=args['label_smoothing'])

    test_loss = []
    train_loss = []

    printw("Num train batches: " + str(len(train_loader)))
    printw("Num test batches: " + str(len(test_loader)))

    torch.cuda.empty_cache()
    # Add early stopping logic
    best_test_loss = float('inf')
    patience_counter = 0
    best_epoch = 0

    for epoch in range(num_epochs):
        # TRAINING
        epoch_train_loss = 0.0
        start_time = time.time()
        # scaler = torch.cuda.amp.GradScaler()

        for i, batch in enumerate(train_loader):
            print(f"Batch {i} of {len(train_loader)}", end='\r')
            batch = {k: v.to(device) for k, v in batch.items()}
            
            
            # Rest of your training code...
            optimal_actions = batch['optimal_actions']
            # optimal_actions = optimal_actions[:, None, :]
            context_actions = batch['context_actions']
            true_actions = torch.cat([context_actions, optimal_actions], dim=1)

            # Apply step masking
            if args.get('use_step_masking', False):
                batch = apply_step_masking(batch, args, epoch, schedule=args['mask_schedule'])
            
            pred_actions = model(batch)
            # if config['num_query'] > 1:
            #     pred_actions = torch.cat([pred_actions[:, :-config['num_query'], :], pred_actions[:, -1:, :]], dim=1)
            
            # true_actions_flat = true_actions.reshape(-1, action_dim)
            # pred_actions_flat = pred_actions.reshape(-1, action_dim)
            # if torch.all(torch.eq(torch.zeros(6).to(device), true_actions[0,800,:])).item():
            #     true_actions_flat = true_actions[:,-1,:].reshape(-1, action_dim)
            #     pred_actions_flat = pred_actions[:,-1,:].reshape(-1, action_dim)
            # elif torch.all(torch.eq(torch.zeros(6).to(device), true_actions[0,600,:])).item():
            #     true_actions_flat = true_actions[:,-201,:].reshape(-1, action_dim)
            #     pred_actions_flat = pred_actions[:,-201,:].reshape(-1, action_dim)
            # elif torch.all(torch.eq(torch.zeros(6).to(device), true_actions[0,400,:])).item():
            #     true_actions_flat = true_actions[:,-401,:].reshape(-1, action_dim)
            #     pred_actions_flat = pred_actions[:,-401,:].reshape(-1, action_dim)
            # elif torch.all(torch.eq(torch.zeros(6).to(device), true_actions[0,200,:])).item():
            #     true_actions_flat = true_actions[:,-601,:].reshape(-1, action_dim)
            #     pred_actions_flat = pred_actions[:,-601,:].reshape(-1, action_dim)
            # elif torch.all(torch.eq(torch.zeros(6).to(device), true_actions[0,0,:])).item():
            #     true_actions_flat = true_actions[:,-801,:].reshape(-1, action_dim)
            #     pred_actions_flat = pred_actions[:,-801,:].reshape(-1, action_dim)
            # else:
            #     true_actions_flat = true_actions.reshape(-1, action_dim)
            #     pred_actions_flat = pred_actions.reshape(-1, action_dim)
            len_query = args['num_query']
            true_actions_flat = true_actions[:,-len_query:,:].reshape(-1, action_dim)
            pred_actions_flat = pred_actions[:,-len_query:,:].reshape(-1, action_dim)

            optimizer.zero_grad()
            loss = loss_fn(pred_actions_flat, true_actions_flat)

            optimizer.zero_grad()
            loss.backward()
            # with torch.cuda.amp.autocast():
            #     loss = loss_fn(pred_actions, true_actions)
            # scaler.scale(loss).backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args['grad_clip'])
            
            optimizer.step()
            # scaler.step(optimizer)
            # scaler.update()
            epoch_train_loss += loss.item() / horizon

        train_loss.append(epoch_train_loss / len(train_dataset))
        end_time = time.time()
        printw(f"\tTrain loss: {train_loss[-1]}")
        printw(f"\tTrain time: {end_time - start_time}")
        
        # Step the scheduler
        scheduler.step()

        if args['wandb']:
            # Log train loss to wandb
            wandb.log({"epoch": epoch + 1, "train_loss": train_loss[-1], "learning_rate": optimizer.param_groups[0]['lr'], "train_time": end_time - start_time})  # Add this line

        # EVALUATION
        printw(f"Epoch: {epoch + 1}")
        start_time = time.time()
        with torch.no_grad():
            epoch_test_loss = 0.0
            for i, batch in enumerate(test_loader):
                print(f"Batch {i} of {len(test_loader)}", end='\r')
                batch = {k: v.to(device) for k, v in batch.items()}
                # true_actions = batch['optimal_actions']
                # pred_actions = model(batch)
                # true_actions = torch.cat()
                # true_actions = true_actions.unsqueeze(
                #     1).repeat(1, pred_actions.shape[1], 1)
                optimal_actions = batch['optimal_actions']
                # optimal_actions = optimal_actions[:, None, :]
                # context_actions = batch['context_actions']
                # true_actions = torch.cat([context_actions, optimal_actions], dim=1)

                # # Apply step masking
                # if args.get('use_step_masking', False):
                #     batch = apply_step_masking(batch, args)
                
                pred_actions_all = model(batch)

                true_actions = optimal_actions[:,-1,:].reshape(-1, action_dim)
                pred_actions = pred_actions_all[:,-1,:].reshape(-1, action_dim)

                loss = loss_fn(pred_actions, true_actions)

                epoch_test_loss += loss.item() / horizon

        test_loss.append(epoch_test_loss / len(test_dataset))
        end_time = time.time()
        printw(f"\tTest loss: {test_loss[-1]}")
        printw(f"\tEval time: {end_time - start_time}")

        if args['wandb']:
            # Log test loss to wandb
            wandb.log({"epoch": epoch + 1, "test_loss": test_loss[-1], "learning_rate": optimizer.param_groups[0]['lr'], "eval_time": end_time - start_time})  # Add this line

        
        # LOGGING
        if (epoch + 1) % 5 == 0 or (env == 'linear_bandit' and (epoch + 1) % 10 == 0):
            torch.save(model.state_dict(), f'{save_dir}/{filename}_epoch{epoch+1}.pt')

        # PLOTTING
        if (epoch + 1) % 5 == 0:
            printw(f"Epoch: {epoch + 1}")
            printw(f"Test Loss:        {test_loss[-1]}")
            printw(f"Train Loss:       {train_loss[-1]}")
            printw("\n")

            plt.yscale('log')
            plt.plot(train_loss[1:], label="Train Loss")
            plt.plot(test_loss[1:], label="Test Loss")
            plt.legend()
            plt.savefig(f"{save_dir}/{filename}_train_loss.png")
            plt.clf()

        if test_loss[-1] < best_test_loss:
            best_test_loss = test_loss[-1]
            patience_counter = 0
            best_epoch = epoch
            # Save best model
            torch.save(model.state_dict(), f'{save_dir}/{filename}_best.pt')
            printw(f"New best model saved at epoch {epoch+1}")
        else:
            patience_counter += 1
            
        if patience_counter >= args['patience']:
            printw(f"Early stopping triggered after {epoch+1} epochs. Best epoch was {best_epoch+1}.")
            break

    torch.save(model.state_dict(), f'{save_dir}/{filename}_last.pt')
    print("Done.")
    if args['wandb']:
        wandb.finish()  # Add this line

    if args['prune_dataset']:
        printw("Pruning dataset...")
        # First pass to compute losses for each sample
        sample_losses = []
        model.eval()
        with torch.no_grad():
            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                optimal_actions = batch['optimal_actions']
                optimal_actions = optimal_actions[:, None, :]
                context_actions = batch['context_actions']
                true_actions = torch.cat([context_actions, optimal_actions], dim=1)
                pred_actions = model(batch)
                
                true_actions = true_actions.reshape(-1, action_dim)
                pred_actions = pred_actions.reshape(-1, action_dim)
                
                # Compute per-sample loss
                losses = torch.nn.functional.cross_entropy(pred_actions.view(-1, action_dim), 
                                                        true_actions.view(-1, action_dim), 
                                                        reduction='none')
                sample_losses.extend(losses.cpu().numpy())
        
        # Find threshold for pruning
        threshold = np.percentile(sample_losses, 100 * (1 - args['prune_ratio']))
        
        # Create pruned dataset
        pruned_indices = [i for i, loss in enumerate(sample_losses) if loss <= threshold]
        train_dataset = torch.utils.data.Subset(train_dataset, pruned_indices)
        train_loader = torch.utils.data.DataLoader(train_dataset, **params)
        printw(f"Pruned dataset from {len(sample_losses)} to {len(pruned_indices)} samples")
