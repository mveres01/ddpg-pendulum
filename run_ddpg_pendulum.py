"""
Script adapted from:
https://github.com/yukezhu/tensorflow-reinforce/blob/master/run_dqn_cartpole.py
"""

from __future__ import print_function
from collections import deque

import numpy as np
import gym

from ddpg import Agent

env_name = 'Pendulum-v0'
env = gym.make(env_name)
env = env.unwrapped
env.seed(1)

state_shape = env.observation_space.shape
num_actions = env.action_space.shape[0]

MAX_EPISODES = 10000
MAX_STEPS = 500
n_iter = 0

action_scale = env.action_space.high[0]
q_learner = Agent(state_shape, num_actions, action_scale)


def exploration(mu, scale, size=None):
    return np.random.normal(mu, scale, size) 


episode_history = deque(maxlen=100)
for i in xrange(MAX_EPISODES):

    # initialize
    state = env.reset()
    total_rewards = 0

    noise = exploration(0.0, 0.2, MAX_STEPS)


    for t in xrange(MAX_STEPS):
        env.render()

        # Add noise and make sure action stays within bounds
        action = q_learner.choose_action(state)
        action = np.clip(action + noise[t], -action_scale, action_scale)

        next_state, reward, done, _ = env.step(action)
        next_state = next_state.flatten()

        # Note how reward is scaled for monitoring purposes
        reward = reward / 10.

        total_rewards += reward

        q_learner.update_buffer(state, action, reward, next_state, done)
       
        state = next_state

        # Fill up some of the experience replay memory before trying to learn
        n_iter += 1
        if n_iter < 10000:
            continue
    
        q_learner.update_policy()


    episode_history.append(total_rewards)

    mean_rewards = np.mean(episode_history)

    print("Episode {}".format(i))
    print("Finished after {} timesteps".format(t+1))
    print("Reward for this episode: {}".format(total_rewards))
    print("Average reward for last 100 episodes: {:.2f}".format(mean_rewards))
