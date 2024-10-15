import gym
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from torch.utils.tensorboard import SummaryWriter

# Import your environment for my case its Microgrid
from env import Microgrid

# Create the environment
env = DummyVecEnv([lambda: Microgrid()])

# Define the PPO model
model = PPO('MlpPolicy', env, verbose=1)

# MAML parameters
meta_lr = 0.0001  # Meta-learning rate
inner_lr = 0.001  # Inner loop learning rate
num_inner_updates = 10  # Number of inner loop updates
meta_batch_size = 20  # Number of tasks in a meta-batch

# TensorBoard writer
writer = SummaryWriter(log_dir='./logs2')

# Function to clone the model
def clone_model(model):
    model_clone = PPO('MlpPolicy', env, verbose=0)
    model_clone.set_parameters(model.get_parameters())
    return model_clone

# Function to perform inner loop updates
def inner_loop_update(model, env, inner_lr, num_inner_updates):
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=inner_lr)
    losses = []
    rewards_list = []
    for update in range(num_inner_updates):
        obs = env.reset()
        total_loss = 0
        total_rewards = 0
        for step in range(1000):  # Collect some data
            action, _ = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            rewards = torch.tensor(rewards, dtype=torch.float32, requires_grad=True)  # Convert rewards to tensor with requires_grad=True
            loss = -torch.mean(rewards)  # Negative reward as loss
            total_loss += loss.item()
            total_rewards += rewards.sum().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        losses.append(total_loss / 1000)
        rewards_list.append(total_rewards / 1000)
    return np.mean(losses), np.std(losses), np.mean(rewards_list), np.std(rewards_list)

# Meta-learning loop
for meta_iteration in range(100):  # Number of meta-iterations
    meta_gradients = None
    meta_losses = []
    meta_rewards = []
    for task in range(meta_batch_size):
        # Clone the model
        model_clone = clone_model(model)
        
        # Perform inner loop updates
        mean_loss, std_loss, mean_reward, std_reward = inner_loop_update(model_clone, env, inner_lr, num_inner_updates)
        
        # Log inner loop metrics
        writer.add_scalar(f'Inner_Loop_Mean_Loss/Iteration_{meta_iteration}', mean_loss, task)
        writer.add_scalar(f'Inner_Loop_Std_Loss/Iteration_{meta_iteration}', std_loss, task)
        writer.add_scalar(f'Inner_Loop_Mean_Reward/Iteration_{meta_iteration}', mean_reward, task)
        writer.add_scalar(f'Inner_Loop_Std_Reward/Iteration_{meta_iteration}', std_reward, task)
        
        # Compute meta-gradients
        obs = env.reset()
        total_loss = 0
        total_rewards = 0
        for step in range(100):  # Collect some data
            action, _ = model_clone.predict(obs)
            obs, rewards, dones, info = env.step(action)
            rewards = torch.tensor(rewards, dtype=torch.float32, requires_grad=True)  # Convert rewards to tensor with requires_grad=True
            loss = -torch.mean(rewards)  # Negative reward as loss
            total_loss += loss.item()
            total_rewards += rewards.sum().item()
            loss.backward()
        
        meta_losses.append(total_loss / 100)
        meta_rewards.append(total_rewards / 100)
        
        # Accumulate meta-gradients
        if meta_gradients is None:
            meta_gradients = [param.grad.clone() if param.grad is not None else torch.zeros_like(param) for param in model.policy.parameters()]
        else:
            for i, param in enumerate(model.policy.parameters()):
                if param.grad is not None:
                    meta_gradients[i] += param.grad.clone()
    
    # Apply meta-gradients
    with torch.no_grad():
        for param, meta_grad in zip(model.policy.parameters(), meta_gradients):
            param -= meta_lr * meta_grad / meta_batch_size

    # Log meta loop metrics
    writer.add_scalar('Meta_Loop_Mean_Loss', np.mean(meta_losses), meta_iteration)
    writer.add_scalar('Meta_Loop_Std_Loss', np.std(meta_losses), meta_iteration)
    writer.add_scalar('Meta_Loop_Mean_Reward', np.mean(meta_rewards), meta_iteration)
    writer.add_scalar('Meta_Loop_Std_Reward', np.std(meta_rewards), meta_iteration)

    # Log meta-gradients
    for i, param in enumerate(model.policy.parameters()):
        writer.add_histogram(f'Meta_Gradients/Param_{i}', meta_gradients[i], meta_iteration)

    # Log parameter values
    for i, param in enumerate(model.policy.parameters()):
        writer.add_histogram(f'Parameters/Param_{i}', param, meta_iteration)

# Save the model
model.save("ppo_microgrid_meta")

# Load the model
model = PPO.load("ppo_microgrid_meta")

# Test the model
obs = env.reset()
episode_rewards = []
episode_lengths = []
episode_reward = 0
episode_length = 0
for step in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    episode_reward += rewards.sum().item()
    episode_length += 1
    if dones.any():
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_reward = 0
        episode_length = 0
        obs = env.reset()

# Log test metrics
writer.add_scalar('Test/Mean_Episode_Reward', np.mean(episode_rewards))
writer.add_scalar('Test/Std_Episode_Reward', np.std(episode_rewards))
writer.add_scalar('Test/Mean_Episode_Length', np.mean(episode_lengths))
writer.add_scalar('Test/Std_Episode_Length', np.std(episode_lengths))

# Close the TensorBoard writer
writer.close()