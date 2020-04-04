# coding: utf-8
import copy
import random

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from functools import reduce


def init_weights(layer):
    if type(layer) == nn.Linear:
        # nn.init.xavier_normal(layer.weight)
        nn.init.kaiming_normal_(layer.weight)


class ToPowerAndConcat(nn.Module):

    def __init__(self):
        super(ToPowerAndConcat, self).__init__()

    def forward(self, x):
        x_2 = torch.pow(x, 2)
        x_3 = torch.pow(x, 3)
        x_4 = torch.pow(x, 4)
        x_res = torch.cat((x, x_2, x_3, x_4), dim=-1)
        return x_res


class ActorModel(nn.Module):

    def __init__(self, in_size, out_size, hidden_size):
        super(ActorModel, self).__init__()
        # squared features
        self.sac = ToPowerAndConcat()

        # gate, learning which parts of polynom to disable
        self.l_gate = nn.Linear(in_size, in_size * 4)
        self.sgm = nn.Sigmoid()

        # MLP after the gate
        self.l1 = nn.Linear(in_size * 4, hidden_size)
        self.act = nn.LeakyReLU(0.001) # strictly speaking, can be removed without too much harm
        self.l2 = nn.Linear(hidden_size, out_size)
        print(self)

    def forward(self, x):
        squared_features = self.sac(x)
        gate_applied = squared_features * self.sgm(self.l_gate(x))
        mlp_applied = self.l2(self.act(self.l1(gate_applied)))
        return mlp_applied


class SimpleQModel(object):

    def __init__(self, device,
                 in_size, out_size,
                 hidden_size, hidden_size1=16, lr=1e-3):

        self.device = device
        self.model = ActorModel(in_size, out_size, hidden_size)
        self.model.apply(init_weights)
        self.model.to(device)

        self.target_model = copy.deepcopy(self.model)
        self.target_model.to(device)

        # ancient man's ancestor is almost always a safe choice
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def fit_last(self, state, action, reward, new_state, done, gamma=0.99):

        state = torch.tensor([state]).to(self.device).float()
        new_state = torch.tensor([new_state]).to(self.device).float()
        reward = torch.tensor([reward]).to(self.device).float()
        action = torch.tensor([action]).to(self.device)
        done = torch.tensor(done).to(self.device)

        with torch.no_grad():
            # max Q-value for next step for each state in a batch
            target_q = self.target_model(new_state).max(dim=1)[0]
            target_q[done] = 0

        # rewards NOW are kinda better than the rewards after death
        target_q = reward + target_q * gamma

        # current prediction
        q = self.model(state).gather(dim=1, index=action.unsqueeze(1))
        loss = F.mse_loss(input=q, target=target_q.unsqueeze(1))
        self.online_optimizer.zero_grad()
        loss.backward()
        self.online_optimizer.step()

    def fit(self, batch, gamma=0.99):

        state, action, reward, next_state, done = batch

        # loading batch
        state = torch.tensor(state).to(self.device).float()
        next_state = torch.tensor(next_state).to(self.device).float()
        reward = torch.tensor(reward).to(self.device).float()
        action = torch.tensor(action).to(self.device)
        done = torch.tensor(done).to(self.device)

        with torch.no_grad():
            # max Q-value for next step for each state in a batch
            target_q = self.target_model(next_state).max(dim=1)[0]
            target_q[done] = 0

        # rewards NOW are kinda better than the rewards after death
        target_q = reward + target_q * gamma

        # current prediction
        q = self.model(state).gather(dim=1, index=action.unsqueeze(1))

        # loss = F.mse_loss(input=q, target=target_q.unsqueeze(1))
        # loss = F.smooth_l1_loss(input=q, target=target_q.unsqueeze(1))
        loss = F.l1_loss(input=q, target=target_q.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()

        # for param in self.model.parameters():
        #     param.grad.data.clamp_(-1, 1)

        self.optimizer.step()

    def select_action(self, state, epsilon, use_target=False):
        # exploration happening
        if random.random() < epsilon:
            return random.randint(0, 2)
        # direct exploitation happening
        state_converted = torch.tensor(state).to(self.device).float().unsqueeze(0)
        if not use_target:
            raw_action_predictions = self.model(state_converted)[0]
        else:
            raw_action_predictions = self.target_model(state_converted)[0]

        return torch.argmax(raw_action_predictions).item()
