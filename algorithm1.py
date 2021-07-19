import numpy as np
import torch
import torch.nn as nn
from expectation_model import ExpectationModel, NonLinearExpectationModel
from distribution_model import DistributionModel
from state_value_fn import StateValueFn
from action_value_fn import ActionValueFn
from replay_buffer import Buffer
from copy import deepcopy
from utils import debug_print
import random


class Algorithm1(nn.Module):
    def __init__(self, _opt):
        super(Algorithm1, self).__init__()
        self.opt = deepcopy(_opt)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.rng = np.random.RandomState(seed=self.opt.seed)
        torch.manual_seed(_opt.seed)
        np.random.seed(_opt.seed)
        random.seed(_opt.seed)

        self.rng_planning = np.random.RandomState(seed=_opt.seed)
        self.rng_prob = np.random.RandomState(seed=_opt.seed)

        self.n_actions = self.opt.n_actions
        self.actions = np.arange(self.opt.n_actions)
        self.gamma = self.opt.gamma
        self.buffer = Buffer(self.opt)

        self.model_updates_batch_size = self.opt.batch_size
        self.dynamics_models = []

        for _ in range(self.opt.n_actions):
            if self.opt.linear_exp_model:
                self.dynamics_models.append(ExpectationModel(self.opt).to(self.device))
            else:
                self.dynamics_models.append(
                    NonLinearExpectationModel(self.opt).to(self.device))

        self.policy = None  # Policy is implicit here
        self.action_value_fn = None
        self.state_value_fn = StateValueFn(self.opt).to(self.device)

        self.batch_size = self.opt.batch_size
        self.counter = 0

    def act(self, state):
        backup_values = []
        with torch.no_grad():
            for action in range(self.opt.n_actions):
                pred_reward, pred_next_state = self.model_predict(state, action)
                backup_values.append(pred_reward +
                                     self.state_value_fn(pred_next_state.to(self.device)).item())
        if self.rng_prob.rand() < self.opt.epsilon:
            action = self.rng.randint(self.n_actions)
        else:
            action = np.argmax(backup_values)

        return action


    def model_update(self, states, actions, rewards, next_states):
        for state, action, reward, next_state in zip(states, actions, rewards,next_states):
            self.dynamics_models[action].update(state, reward, next_state)

    def update(self, state, action, reward, next_state, done, sample_env, next_state_position):
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        reward = torch.FloatTensor([reward]).to(self.device)
        done = torch.FloatTensor([done]).to(self.device)

        error = self.td_error(state, reward, next_state, done)
        self.td_update(error)

        self.buffer.add(state, action, reward, next_state, done, sample_env.max_goal_idx, next_state_position)

        self.model_update([state], [action], [reward], [next_state],
                          [sample_env.max_goal_idx], [next_state_position])
        sample_transitions = self.buffer.sample_transitions(
            self.batch_size)
        states, actions, rewards, next_states, _= sample_transitions
        self.model_update(states, actions, rewards, next_states)

        self.planning()
        self.counter += 1

    def v(self, state, with_grad=True):
        if not with_grad:
            with torch.no_grad():
                v = self.state_value_fn(state)
        else:
            v = self.state_value_fn(state)
        return v

    def td_error(self, state, reward, next_state, done):
        return self.td_target(reward, next_state, done) - self.v(state)

    def td_target(self, reward, next_state, done):
        return reward + self.gamma * (1 - done) * self.v(next_state, False)

    def td_update(self, error):
        self.state_value_fn.optimizer.zero_grad()
        loss = error.pow(2).mean()
        loss.backward()
        self.state_value_fn.optimizer.step()

    def model_predict(self, state, action):
        with torch.no_grad():
            pred_reward, pred_next_state = self.dynamics_models[action].forward(state)
        return pred_reward, pred_next_state


    def planning(self):
        if self.batch_size > 0:
            sample_transitions = self.buffer.sample_transitions(self.batch_size)
            states, actions, rewards, next_states, dones = sample_transitions
            states = torch.stack(states)

        for idx in range(self.batch_size):
            state = states[idx]
            done = dones[idx]

            targets_list = []
            for action in range(self.opt.n_actions):
                with torch.no_grad():
                    pred_reward, pred_next_state = self.model_predict(state, action)
                    target = self.td_target(pred_reward, pred_next_state, done)
                    targets_list.append(target)
            targets = torch.stack(targets_list)
            target = torch.max(targets)
            error = target - self.v(state)
            self.td_update(error)