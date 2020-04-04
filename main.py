# coding: utf-8

import copy
import os
import shutil

import gym
import torch

from memory import Memory
from model import SimpleQModel
from training import hacked_reward, target_run
from utils import plot_learning_curves, plot_best_actions

env = gym.make("MountainCar-v0")
positions_range, velocity_range = (-1.2, 0.6), (-0.07, 0.07)

# `model` updates before we replace target_model with it
target_update = 1000

# update batch size for training
batch_size = 128
memory_capacity = 5000

# environment steps limit
max_steps = 100001

# exploration rate interval
max_epsilon = 0.9
min_epsilon = 0.0

device = torch.device("cpu")


def run(gamma=0.99, device=device):
    # cyclic buffer for experience episodes
    memory = Memory(capacity=memory_capacity)

    # model init
    simple_model = SimpleQModel(device,
                                in_size=2, out_size=3,
                                hidden_size=512, lr=0.00005)

    rewards_by_target_updates, tricky_rewards_by_target_updates = [], []
    state = env.reset()
    current_step_number = 0

    for step in range(max_steps):

        # choosing exploration rate (epsilon-greedy exploration)
        epsilon = max_epsilon - (max_epsilon - min_epsilon) * step / max_steps

        # choosing action
        action = simple_model.select_action(state, epsilon)

        # applying the action and getting the reward
        new_state, reward, done, _ = env.step(action)

        # TRICK: reward shaping (and risking to get a suboptimal solution as well)
        tricky_reward = hacked_reward(state, new_state, step, current_step_number, reward)

        # saving experience
        memory.push((state, action, tricky_reward, new_state, done))

        # # TRICK: the task has a certain degree of symmetry --------------------------
        # # hence adding the mirrored experience (as long as it is not the one close
        # # to the top of the right hill)
        # if current_step_number < 50:
        #     memory.push((-state, 2 - action, tricky_reward, -new_state, done))
        # memory.push((-state, 2 - action, tricky_reward, -new_state, done))

        # # TRICK: adding random jitter -----------------------------------------------
        # # (we're dealing with speeds and distances,
        # # local jitter may provide extra useful data and robustness)
        # for i in range(3):
        #     state += np.random.normal(size=2, scale=0.01)
        #     new_state += np.random.normal(size=2, scale=0.01)
        #     memory.push((state, action, tricky_reward, new_state, done))

        # if done, time to reset, else -- move on
        if done:
            state = env.reset()
            done = False
            current_step_number = 0
        else:
            state = new_state
            current_step_number += 1

        ####### TRAINING #######
        # gradient update based on experience sample from memory
        if step > batch_size:
            simple_model.fit(memory.sample(batch_size), gamma)

        ####### EVALUATION + TARGET UPDATE #######
        if step % target_update == 0:
            target_model = copy.deepcopy(simple_model.model)

            # running one session simulation
            # to find out how we're doing
            total_reward, total_tricky_reward = target_run(env, step, simple_model)

            # saving the score for tracking progress
            rewards_by_target_updates.append(total_reward)
            tricky_rewards_by_target_updates.append(total_tricky_reward)

            if step % 2000 == 0:
                plot_best_actions(simple_model, positions_range, velocity_range, max_steps, marker="step" + str(step))

            simple_model.target_model = target_model

    return rewards_by_target_updates, tricky_rewards_by_target_updates, simple_model


if __name__ == "__main__":
    # reading actions maps
    shutil.rmtree("actions_tracking/", ignore_errors=True)
    os.mkdir("actions_tracking")

    rewards, trewards, simple_model = run()
    plot_learning_curves(rewards, trewards, max_steps, target_update)
    plot_best_actions(simple_model, positions_range, velocity_range, max_steps, marker=None)
    env.close()
