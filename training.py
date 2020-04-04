# coding: utf-8

from math import log


def hacked_reward(old_state, new_state, step, session_step, reward, old_penalty=0.85):
    """ Reward shaping. DANGER! Tricks with rewards! Suboptimal behaviour possible!"""
    new_velocity, old_velocity = new_state[1], old_state[1]
    abs_velocity_diff = abs(new_velocity) - old_penalty * abs(old_velocity)

    # we try not to reward the model too much as it progresses to its destination
    # this helps us to escape training a model rallying from one side of the slope to the other side
    return reward + 100 * abs_velocity_diff / log(session_step + 2)  # / log(log(step + 2) + 1)


def target_run(env, step, simple_model):
    """Running the current model to evaluate"""
    done = False
    state = env.reset()
    total_reward, total_tricky_reward = 0, 0
    current_step_number = 0

    # sample run, stepping-stepping-stepping
    # following what we've learnt at this point
    while not done:
        action = simple_model.select_action(state, 0, use_target=True)
        old_state = state
        state, reward, done, _ = env.step(action)
        total_reward += reward
        # env.render()
        tricky_reward = hacked_reward(old_state, state, step, current_step_number, reward)
        total_tricky_reward += tricky_reward
        current_step_number += 1

    done = False
    state = env.reset()
    print(str(step) + "\ttotal orig.  reward as of now: ", total_reward)
    print(str(step) + "\ttotal hacked reward as of now:", total_tricky_reward)
    print()

    return total_reward, total_tricky_reward
