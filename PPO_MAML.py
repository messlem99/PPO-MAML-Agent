import os
import gym
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter

# Import your corrected environment
from env import Microgrid

def evaluate_meta_model(model, eval_episodes=10):
    """
    Evaluates the current meta-model's zero-shot performance on new tasks.
    """
    eval_env = Microgrid()
    total_rewards = 0
    for _ in range(eval_episodes):
        obs = eval_env.reset()
        done = False
        episode_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = eval_env.step(action)
            episode_reward += reward
        total_rewards += episode_reward
    return total_rewards / eval_episodes

if __name__ == "__main__":
    # --- 1. Define Hyperparameters ---
    PPO_PARAMS = {
        'learning_rate': 4.76e-4,
        'n_steps': 640,
        'batch_size': 256,
        'n_epochs': 10,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.0,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        'policy_kwargs': dict(net_arch=[dict(pi=[64, 64], vf=[64, 64])])
    }

    META_ITERATIONS = 200
    META_LR = 0.001
    META_BATCH_SIZE = 100
    INNER_UPDATES_TIMESTEPS = 72 * 10

    LOG_DIR = "./logs_maml_ppo"
    os.makedirs(LOG_DIR, exist_ok=True)
    writer = SummaryWriter(log_dir=LOG_DIR)

    # --- 2. Setup Environment and Meta-Model ---
    def make_env():
        return Microgrid()

    vec_env = make_vec_env(make_env, n_envs=1)
    meta_model = PPO('MlpPolicy', vec_env, verbose=0, **PPO_PARAMS)

    # --- 3. The MAML-PPO Training Loop ---
    print("Starting MAML-PPO Training...")
    for meta_iter in range(META_ITERATIONS):
        adapted_model_params = []
        
        # --- INNER LOOP ---
        for task_index in range(META_BATCH_SIZE):
            task_model = PPO('MlpPolicy', vec_env, verbose=0, **PPO_PARAMS)
            task_model.set_parameters(meta_model.get_parameters())
            
            # The env resets automatically, creating a new task
            task_model.learn(total_timesteps=INNER_UPDATES_TIMESTEPS)
            
            adapted_model_params.append(deepcopy(task_model.get_parameters()))
        
        # --- OUTER LOOP (Reptile-like FOMAML update) ---
        current_meta_params = meta_model.get_parameters()
        avg_update_direction = {}
        
        for key in current_meta_params['policy']:
            sum_of_adapted_params = sum(p['policy'][key] for p in adapted_model_params)
            avg_adapted_param = sum_of_adapted_params / META_BATCH_SIZE
            avg_update_direction[key] = avg_adapted_param - current_meta_params['policy'][key]

        updated_meta_params = deepcopy(current_meta_params)
        for key in updated_meta_params['policy']:
            updated_meta_params['policy'][key] = current_meta_params['policy'][key] + META_LR * avg_update_direction[key]
        
        meta_model.set_parameters(updated_meta_params)
        
        # --- Logging and Saving ---
        if meta_iter % 10 == 0:
            avg_eval_reward = evaluate_meta_model(meta_model, eval_episodes=20)
            print(f"Meta-Iteration: {meta_iter}/{META_ITERATIONS} | Avg Eval Reward: {avg_eval_reward:.2f}")
            writer.add_scalar('Evaluation/Average_Reward', avg_eval_reward, meta_iter)
            meta_model.save(os.path.join(LOG_DIR, f"maml_ppo_model_{meta_iter}_steps"))

    # --- 4. Final Save ---
    meta_model.save("maml_ppo_microgrid_final")
    writer.close()
    print("Training complete. Final model saved as maml_ppo_microgrid_final.zip")
