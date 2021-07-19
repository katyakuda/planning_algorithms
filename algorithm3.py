import numpy as np
import torch
import torch.nn as nn
from expectation_model import ExpectationModel, NonLinearExpectationModel
from distribution_model import DistributionModel
from state_value_fn import StateValueFn
from policy import Policy
from replay_buffer import Buffer
from copy import deepcopy
from utils import anneal_epsilon
import torch.nn.functional as F
from policy import Policy
import random


class Algorithm3(nn.Module):
    def __init__(self, _opt):
        super(Algorithm3, self).__init__()
        self.opt = deepcopy(_opt)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.rng = np.random.RandomState(self.opt.seed)
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

        self.state_value_fn = StateValueFn(self.opt).to(self.device)

        self.batch_size = self.opt.batch_size
        self.counter = 0
        self.temperature = 1.

        self.policy = Policy(self.opt)
        self.beta = _opt.entropy_weight

    def act(self, state):
        with torch.no_grad():
            policy_distribution = self.policy.forward(state)
            action = policy_distribution.sample().item()

        if self.rng_prob.rand() < self.opt.epsilon:
            action = self.rng.randint(self.n_actions)

        return action

    def check_and_flip_reward(self, max_goal_idx, reward, next_state_position):
        if self.opt.maze == "corridor":
            if max_goal_idx != self.max_goal_idx:
                if reward == self.opt.r_goal_max:
                    reward = self.opt.r_goal
                    return torch.FloatTensor([reward]).to(self.device)
                if reward == self.opt.r_goal:
                    reward = self.opt.r_goal_max
                return torch.FloatTensor([reward]).to(self.device)
            else:
                return reward
        elif self.opt.maze == "full_maze":
            if max_goal_idx != self.max_goal_idx:
                if reward == self.opt.r_goal or reward == self.opt.r_goal_max:
                    # print("last state pos {} goal pos {}", next_state_position, self.current_max_goal_position)
                    if next_state_position == self.current_max_goal_position:
                        reward = self.opt.r_goal_max
                        # print("\treward flipped", reward)
                    else:
                        reward = self.opt.r_goal
                        # print("\treward flipped", reward)
                    return torch.FloatTensor([reward]).to(self.device)
                else:
                    return reward
            else:
                return reward
        else:
            return reward

    def model_update(self, states, actions, rewards, next_states, max_goal_idxs, next_state_positions):
        for state, action, reward, next_state, max_goal_idx, next_state_position in zip(states, actions, rewards,
                                                                                        next_states, max_goal_idxs,
                                                                                        next_state_positions):
            reward = self.check_and_flip_reward(max_goal_idx, reward, next_state_position)
            self.dynamics_models[action].update(state, reward, next_state)

    def update(self, state, action, reward, next_state, done, sample_env, next_state_position):
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        reward = torch.FloatTensor([reward]).to(self.device)
        done = torch.FloatTensor([done]).to(self.device)

        advantage = self.td_error(state, reward, next_state, done)
        self.td_update(advantage)
        self.policy_update(state, action, advantage.detach())

        self.buffer.add(state, action, reward, next_state, done, sample_env.max_goal_idx, next_state_position)
        self.max_goal_idx = sample_env.max_goal_idx
        self.current_max_goal_position = sample_env.max_goal_position

        self.model_update([state], [action], [reward], [next_state],
                          [sample_env.max_goal_idx], [next_state_position])
        sample_transitions = self.buffer.sample_transitions(
            self.batch_size)
        states, actions, rewards, next_states, _, max_goal_idxs, next_state_positions = sample_transitions
        self.model_update(states, actions, rewards, next_states, max_goal_idxs, next_state_positions)

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
        # advantage = returns - values
        return self.td_target(reward, next_state, done).detach() - self.v(state)

    def td_target(self, reward, next_state, done):
        return reward + self.gamma * (1 - done) * self.v(next_state, False)

    def td_update(self, error):
        self.state_value_fn.optimizer.zero_grad()
        loss = error.pow(2).mean()
        loss.backward()
        self.state_value_fn.optimizer.step()

    def policy_update(self, state, action, td_error):
        policy_distribution = self.policy.forward(state)
        log_probs = policy_distribution.log_prob(torch.FloatTensor([action]))
        # print("log probs", log_probs)

        entropy = policy_distribution.entropy().mean()
        loss = -(log_probs * td_error.detach()).mean() - self.beta * entropy
        # print("loss", loss)
        # print("weight" , self.policy.fc1.weight)
        # print("before step: ", list(self.policy.parameters()))
        # print("grad", self.policy.fc.weight.grad)
        # print("Gradients:")
        # for p in self.policy.parameters():
        #     if p.grad is None:
        #         continue
        #     grad = p.grad.data
        #     print(grad)

        # print(self.policy.fc1.weight.grad)

        self.policy.optimizer.zero_grad()
        loss.backward()
        # print(loss)
        # nn.utils.clip_grad_norm_(self.policy.parameters(), 4)
        # print(self.policy.policy.weight.grad)
        self.policy.optimizer.step()
        # print("weight after", self.policy.fc1.weight)
        # print("after step: ", list(self.policy.parameters()))

    def model_predict(self, state, action):
        with torch.no_grad():
            pred_reward, pred_next_state = self.dynamics_models[action].forward(state)
        return pred_reward, pred_next_state

    # def planning(self):
    #     if self.batch_size > 0:
    #         sample_transitions = self.buffer.sample_transitions(self.batch_size)
    #         states, actions, rewards, next_states, dones, max_goal_idxs = sample_transitions
    #         states = torch.stack(states)
    #     # n-planning steps
    #     for idx in range(self.batch_size):
    #         state = states[idx]
    #         done = dones[idx]
    #
    #         backup_values_list = []  # targets
    #         td_errors = []
    #         for action in range(self.opt.n_actions):
    #             with torch.no_grad():
    #                 pred_reward, pred_next_state = self.dynamics_models[action].forward(state)
    #                 target = self.td_target(pred_reward, pred_next_state, done)  # estimated returns
    #                 td_error = self.td_error(state, pred_reward, pred_next_state, done).detach()
    #
    #             backup_values_list.append(target)
    #             td_errors.append(td_error)
    #
    #         backup_values = torch.stack(backup_values_list)
    #         target = torch.max(backup_values)
    #         action = torch.argmax(backup_values)
    #         error = target - self.v(state)
    #         self.td_update(error)
    #             # print("weight before", self.policy.fc1.weight)
    #         self.policy_update(state, action, error.detach())
                # print("weight after" , self.policy.fc1.weight)

    def planning(self):
        if self.batch_size > 0:
            sample_transitions = self.buffer.sample_transitions(self.batch_size)
            states, actions, rewards, next_states, dones, _, _ = sample_transitions
            states = torch.stack(states)
        # n-planning steps
        for idx in range(self.batch_size):
            state = states[idx]
            done = dones[idx]
            # VAR 1
            # backup_values_list = []  # targets
            # td_errors = []
            # for action in range(self.opt.n_actions):
            #     with torch.no_grad():
            #         pred_reward, pred_next_state = self.dynamics_models[action].forward(state)
            #         target = self.td_target(pred_reward, pred_next_state, done)  # estimated returns
            #         td_error = self.td_error(state, pred_reward, pred_next_state, done).detach()
            #         # self.policy_update(state, action, td_error)
            #     backup_values_list.append(target)
            #     td_errors.append(td_error)
            #
            # backup_values = torch.stack(backup_values_list)
            # target = torch.max(backup_values)
            # error = target - self.v(state)
            # self.td_update(error)
            # self.policy_update(state, action, error.detach())

            # VAR 2
            # with torch.no_grad():
            #     policy_distribution = self.policy.forward(state)
            #     action = policy_distribution.sample().item()
            #
            # with torch.no_grad():
            #     pred_reward, pred_next_state = self.dynamics_models[action].forward(state)
            #     target = self.td_target(pred_reward, pred_next_state, done)  # estimated returns
            #     td_error = self.td_error(state, pred_reward, pred_next_state, done).detach()
            #
            # error = target - self.v(state)
            # self.td_update(error)
            # self.policy_update(state, action, error.detach())
            #
            # Var 3
            backup_values_list = []  # targets
            td_errors = []
            for action in range(self.opt.n_actions):
                with torch.no_grad():
                    pred_reward, pred_next_state = self.dynamics_models[action].forward(state)
                    target = self.td_target(pred_reward, pred_next_state, done)  # estimated returns
                    td_error = self.td_error(state, pred_reward, pred_next_state, done).detach()
                    # self.policy_update(state, action, td_error)
                backup_values_list.append(target)
                td_errors.append(td_error)

            backup_values = torch.stack(backup_values_list)
            target = torch.max(backup_values)
            error = target - self.v(state)
            self.td_update(error)
            with torch.no_grad():
                policy_distribution = self.policy.forward(state)
                action = policy_distribution.sample().item()
            target = backup_values[action]
            error = target - self.v(state)
            self.policy_update(state, action, error.detach())

            # FIXME see VDE_alg3 action = torch.argmax(targets)