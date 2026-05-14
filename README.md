# Proximal Policy Optimization with Model-Agnostic Meta-Learning for Battery Energy Storage System Management in a Multi-Microgrid
It is hypothesized that the integration of Proximal Policy Optimization (PPO) with Model-Agnostic Meta-Learning (MAML) will enhance the performance and adaptability of reinforcement learning algorithms for a dynamic custom OpenAI Gym environment. Specifically, it is proposed that this novel combination will:
  
   * Improve Adaptability: More rapid and effective adaptation to varying tasks and scenarios will be enabled compared to traditional PPO alone.
   *  Enhance Performance: Better performance in terms of efficiency and policy robustness will be achieved by leveraging MAML’s meta-learning capabilities in conjunction with PPO’s robust policy optimization.
   
The command "tensorboard --logdir=./Logs" should be used to visualize the data with TensorBoard during all the training time.
# Steps to reproduce:

To achieve Proximal Policy Optimization (PPO) with Model-Agnostic Meta-Learning (MAML), begin by setting up the environment and initializing the PPO model. Define the meta-learning parameters such as meta-learning rate, inner loop learning rate, and batch size. Implement functions to clone the PPO model and perform inner loop updates by collecting rewards and optimizing the cloned model. In the meta-learning loop, update the original model by accumulating meta-gradients from multiple tasks and applying them with the meta-learning rate. Log metrics using TensorBoard throughout training, save and load the trained model, and finally test the model to evaluate its performance. Note that in my case, the environment is customized to ensure that each time it is cloned, it provides a new, complete model, effectively treating each clone as a new task.

# This work was conducted in collaboration with the following authors: 
* abdelkader.messlem@univ-tiaret.dz (Abdelkader Messlem), messlem.youcef@univ-tiaret.dz (Youcef Messlem), ahmed.safa@univ-tiaret.dz (Ahmed Safa), kouadria.selman@univ-tiaret.dz (Selman Kouadria), djafar.ould-abdeslam@uha.fr (Djaffar Ould Abdeslam)
# Cite
```bibtex
@article{MESSLEM2026113222,
title = {Meta-reinforcement learning for adaptive control of battery energy storage systems in multi-microgrids},
journal = {Electric Power Systems Research},
volume = {259},
pages = {113222},
year = {2026},
issn = {0378-7796},
doi = {[https://doi.org/10.1016/j.epsr.2026.113222](https://doi.org/10.1016/j.epsr.2026.113222)},
url = {[https://www.sciencedirect.com/science/article/pii/S0378779626005158](https://www.sciencedirect.com/science/article/pii/S0378779626005158)},
author = {Abdelkader Messlem and Youcef Messlem and Ahmed Safa and Selman Kouadria and Djaffar {Ould Abdeslam}}
}
```
