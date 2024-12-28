from env import Microgrid
import os
from stable_baselines3 import PPO
from prettytable import PrettyTable
from real_data_sim import RealData
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 18
# Load real data
real_data = RealData('energy_dataset.csv', '2017-1-1', '2017-12-30')

# Load PPO model
PPO_path = os.path.join('PPO_microgrid_meta')
env = Microgrid()
model = PPO.load(PPO_path, env=env)

# Number of episodes to run
episode_count = 365
reward_oa = 0

# Lists to store state values for plotting
soc1_values = []
soc2_values = []
soc3_values = []

# Iterate through episodes
for episode in range(1, episode_count + 1):
    # Reset the environment, but we'll override the state immediately
    _ = env.reset() 

    # Initialize state with real data for the first step
    state = [
        0.5, 0.5, 0.5,  # SoC1, SoC2, SoC3 (keep initial SoC values)
        real_data.bc1.iloc[0], real_data.bc2.iloc[0], real_data.bc3.iloc[0],  # BC1, BC2, BC3
        real_data.pv1.iloc[0], real_data.pv2.iloc[0], real_data.pv3.iloc[0],  # PV1, PV2, PV3
        real_data.wt1.iloc[0], real_data.wt2.iloc[0], real_data.wt3.iloc[0],  # WT1, WT2, WT3
        real_data.load_demand.iloc[0], real_data.FLoad.iloc[0], 
        real_data.data_month[real_data.price_column].iloc[0],  # Price
        real_data.total_forecast.iloc[0],  # FPrice
        real_data.mg1.iloc[0], real_data.mg2.iloc[0], real_data.mg3.iloc[0],  # mg1, mg2, mg3
        real_data.total_non_renewable.iloc[0] 
    ]
    done = False
    total_reward = 0

    # Counter to track the current step in the real data
    data_step = 0 

    # Start the episode
    while not done:
        action, _ = model.predict(state)
        next_state, reward, done, info = env.step(action)

        # Update state with real data for the next step
        data_step += 1
        if data_step < real_data.data_month.shape[0]:  # Use shape[0] to get the number of rows
            state = [
                next_state[0], next_state[1], next_state[2],  # Keep updated SoCs
                real_data.bc1.iloc[data_step], real_data.bc2.iloc[data_step], real_data.bc3.iloc[data_step],
                real_data.pv1.iloc[data_step], real_data.pv2.iloc[data_step], real_data.pv3.iloc[data_step],
                real_data.wt1.iloc[data_step], real_data.wt2.iloc[data_step], real_data.wt3.iloc[data_step],
                real_data.load_demand.iloc[data_step], real_data.FLoad.iloc[data_step],
                real_data.data_month[real_data.price_column].iloc[data_step],
                real_data.total_forecast.iloc[data_step],
                real_data.mg1.iloc[data_step], real_data.mg2.iloc[data_step], real_data.mg3.iloc[data_step],
                real_data.total_non_renewable.iloc[data_step]
            ]
        else:
            # If we've exhausted the real data, end the episode
            done = True 
        
        # Store the state values for plotting
        soc1_values.append(state[0])
        soc2_values.append(state[1])
        soc3_values.append(state[2])
        
        print(f"Episode: {episode}, Step: {env.current_step}\n")
        print(f"Action: {action}")
        print("Done:", done)
        print("-----")
        
        total_reward += reward

    print(f"Episode: {episode} finished with total reward: {total_reward}")
    reward_oa += total_reward

# Apply Gaussian filter to smooth the SoC values
soc1_values_smoothed = gaussian_filter1d(soc1_values, sigma=2)
soc2_values_smoothed = gaussian_filter1d(soc2_values, sigma=2)
soc3_values_smoothed = gaussian_filter1d(soc3_values, sigma=2)

# Plot the smoothed state values
plt.figure(figsize=(10, 6))
plt.plot(soc1_values_smoothed, label='BESS1')
plt.plot(soc2_values_smoothed, label='BESS2')
plt.plot(soc3_values_smoothed, label='BESS3')
plt.xlabel('Steps')
plt.ylabel('SoC')
plt.title('')
plt.legend()
plt.grid()
plt.show()

print(f"Total reward across all episodes: {reward_oa}")