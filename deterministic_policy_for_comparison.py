"""
    Closed-form deterministic solution to compare with. Copied and adapted from
    https://github.com/ZhiqingXiao/OpenAIGymSolution/blob/master/MountainCar-v0_close_form/mountaincar_v0_close_form.ipynb
"""

import itertools
import numpy as np
import gym
from tqdm import tqdm

from utils import plot_best_actions

np.random.seed(0)
env = gym.make('MountainCar-v0')
env.seed(0)

class Agent:

    def select_action(self, observation, epsilon=None, use_target=None):
        try:
            position = observation[0]
            velocity = observation[1]
        except Exception as e:
            position = observation[0][0]
            velocity = observation[0][1]

        lb = min(-0.09 * (position + 0.25) ** 2 + 0.03,
                 0.3 * (position + 0.9) ** 4 - 0.008)
        ub = -0.07 * (position + 0.38) ** 2 + 0.07
        if lb < velocity < ub:
            action = 2  # push right
        else:
            action = 0  # push left
        return action


agent = Agent()


def play_once(env, agent, render=False, verbose=False):
    observation = env.reset()
    episode_reward = 0.

    for step in itertools.count():
        if render:
            env.render()
        action = agent.select_action(observation)
        observation, reward, done, _ = env.step(action)
        episode_reward += reward
        if done:
            break
    if verbose:
        print('get {} rewards in {} steps'.format(
            episode_reward, step + 1))
    return episode_reward


play_once(env, agent, render=True)
plot_best_actions(agent, (-1.2, 0.6), (-0.07, 0.07), 0, marker="DETERMINISTIC_POLICY")

episode_rewards = [play_once(env, agent) for _ in tqdm(range(100))]
print('average episode rewards = {}'.format(np.mean(episode_rewards)))

env.close()
