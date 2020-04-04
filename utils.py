# coding: utf-8

import time

import matplotlib.pyplot as plt
import numpy as np
from numpy import linspace
from torch import nn


class Lambda(nn.Module):
    "An easy way to create a pytorch layer for a simple `func`."

    def __init__(self, func):
        "create a layer that simply calls `func` with `x`"
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class MonoBilinear(nn.Module):
    """
        Too slow?
    """

    def __init__(self, in_features, out_features, bias=True):
        super(MonoBilinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.actual_layer = nn.Bilinear(in_features, in_features, out_features, bias)

    def forward(self, x):
        return self.actual_layer(x, x)


def plot_best_actions(qmodel, ranges_pos, ranges_velocity, maxsteps, marker=None):
    """ Draws the map of actions taken in different states """

    grid = 50
    l = {1: [], 2: [], 0: []}

    for pos in linspace(ranges_pos[0], ranges_pos[1], grid):
        for vel in linspace(ranges_velocity[0], ranges_velocity[1], grid):
            label = qmodel.select_action(np.array([[pos, vel]]), 0, use_target=True)
            l[label].append((pos, vel))

    for label in l:
        x = [a for a, b in l[label]]
        y = [b for a, b in l[label]]
        plt.scatter(x, y, label="action " + str(label))

    plt.title("Actions map: (positions, velocity) -> action")
    leg = plt.legend(loc='best', ncol=2)
    leg.get_frame().set_alpha(0.3)

    if marker:
        plt.savefig("actions_tracking/actions_space_%s_%d.png" % (marker, int(time.time())))
    else:
        plt.savefig("actions_space_%d.png" % int(time.time()))
        plt.show()

    plt.clf()


def plot_learning_curves(rewards, trewards, max_steps, target_update):
    rng = [i * target_update for i in range(max_steps // target_update + 1)]
    plt.plot(rng, trewards, label="hacked reward")
    plt.plot(rng, rewards, label="original reward")
    plt.title("Learning progress, %d steps" % max_steps)
    plt.ylim((-205, -50))

    leg = plt.legend(loc='best', ncol=2)
    leg.get_frame().set_alpha(0.1)
    plt.savefig("results_training_%d.png" % int(time.time()))

    plt.show()
    plt.clf()
