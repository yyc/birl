#!/usr/bin/env python
import numpy as np
from pdb import set_trace
import torch
import torch.nn.functional as F


class MDP:
    def __init__(self, transition_weights, transition_beta, env_features, reward_weights, gamma, start=None, goal = None, terminal=None, max_value=1000):
        self.env_features = torch.from_numpy(env_features).cuda()
        self.transitions = self.calculate_tp(torch.from_numpy(transition_weights).cuda(),transition_beta)
        num_actions = self.transitions.size()[1]
        self.rewards = self.calculate_rewards(torch.Tensor(reward_weights).cuda())
        self.gamma = gamma
        self.states = range(np.shape(self.transitions)[0])
        self.actions = range(np.shape(self.transitions)[1])
        self.max_value = max_value
        self.start = start
        self.goal = (goal*torch.ones(1).long()).cuda()
        self.terminal = len(self.states) - 1
        self.check_valid_mdp()
        self.policy = None
        self.Q = None

    def check_valid_mdp(self):
        is_valid = True
        # Need to be of the form S,A,T(S,A)
        if not (len(np.shape(self.transitions)) == 3):
            is_valid = False
        # check that state space size is same in both dims
        if not (np.shape(self.transitions)[0] == np.shape(self.transitions)[2]):
            is_valid = False
        # check that probabilities are valid
        for s in range(np.shape(self.transitions)[0]):
            for a in range(np.shape(self.transitions)[1]):
                prob_sum = 0
                for sprime in range(np.shape(self.transitions)[2]):
                    prob = self.transitions[s][a][sprime]
                    if prob < 0 or prob > 1:
                        is_valid = False
                    prob_sum += prob
                np.testing.assert_almost_equal(1.0, prob_sum)
        if self.gamma < 0 or self.gamma > 1:
            is_valid = False
        assert (is_valid)

    '''
    Policy Iteration from Sutton and Barto
    assumes discount factor of 0.99
    Deterministic policy iteration
    '''

    def policy_iteration(self, policy=None):
        # initialization
        if policy is None:
            policy = self.get_random_policy()

        policy_stable = False
        count = 0
        while not policy_stable:
            # policy evaluation
            V = self.policy_evaluation(policy)
            count += 1
            diff_count = 0
            # policy improvement
            policy_stable = True
            for state in self.states:
                old_action = policy[state]
                action_vals = torch.mm(self.transitions[state, :, :], torch.unsqueeze(self.rewards + self.gamma * V,dim=1)).squeeze(dim=1)
                policy[state] = torch.argmax(action_vals)
                if not old_action == policy[state]:
                    diff_count += 1
                    policy_stable = False
        return (policy, V)

    def get_random_policy(self):
        policy = torch.zeros(len(self.states)).cuda().int()
        for state in self.states:
            policy[state] =torch.randint(0, len(self.actions),(1,)).cuda()
        return policy

    '''
    policy - deterministic policy, maps state to action
    -Deterministic policy evaluation
    '''

    def policy_evaluation(self, policy, theta=0.0001):
        V = torch.zeros(len(self.states)).cuda().double()
        count = 0
        while True:
            delta = torch.zeros(1).cuda().double()
            for state in self.states:
                value = V[state]
                V[state] = torch.dot(self.transitions[state, policy[state], :], self.rewards + self.gamma * V)
                delta = torch.max(delta, torch.abs(value - V[state]))
                # If divergence and policy has value -inf, return value function early
                if V[state] > self.max_value:
                    return V
                if V[state] < -self.max_value:
                    return V
            if delta < theta:
                break
        return V

    def policy_q_evaluation(self, policy):
        V = self.policy_evaluation(policy)
        Q = torch.zeros(self.transitions.size()[0:2]).cuda().double()
        for state in self.states:
            for action in self.actions:
                Q[state, action] = torch.dot(self.transitions[state, action, :], self.rewards + self.gamma * V)
        return Q

    def value_iteration(self, theta=0.0001):
        V = np.zeros(len(self.states))
        while True:
            delta = 0
            for state in self.states:
                v = V[state]
                V[state] = np.amax(np.dot(self.transitions[state, :, :], self.rewards + self.gamma * V))
                delta = max(delta, np.abs(v - V[state]))
            if delta < theta:
                break
        return V

    def q_value_iteration(self, theta=0.0001):
        Q = np.zeros((len(self.states), len(self.actions)))
        while True:
            delta = 0
            for state in self.states:
                for action in self.actions:
                    q = Q[state, action]
                    Q[state, action] = np.dot(self.transitions[state, action, :],
                                              self.rewards + self.gamma * np.amax(Q, axis=1))
                    delta = max(delta, np.abs(q - Q[state, action]))
            if delta < theta:
                break
        return Q

    '''
    Takes a state and action, returns next state
    Chooses a next state according to transition 
    probabilities
    '''

    def act(self, state, action):
        next_state = np.random.choice(self.states, p=self.transitions[state, action, :])
        return self.rewards[next_state], next_state

    def calculate_rewards(self, reward_weights):
        rw = torch.mm(self.env_features, reward_weights.double().unsqueeze(dim=1)).double()
        return rw.squeeze(dim=1)

    def update_rewards(self,reward_weights):
        self.rewards = self.calculate_rewards(reward_weights)

    def update_tp(self,tp_weights,tp_beta):
        self.transitions = self.calculate_tp(tp_weights,tp_beta)

    def update_policy(self,use_policy=False):
        if use_policy:
            self.policy = self.policy_iteration(self.policy)[0]
        else:
            self.policy = self.policy_iteration()[0]

    def do_policy_q_evaluation(self):
        self.Q = self.policy_q_evaluation(self.policy)

    def get_trajectories(self):
        demos = []
        demo = []
        s0 = self.start
        reward_cum = 0.
        go_on = True
        while go_on:
            a0 = self.policy[s0]
            _, s1 = self.act(s0, a0)
            demo.append((s0, a0))
            reward_cum += self.rewards[s0]
            if s0 == self.goal or len(demo) > 30:
                go_on = False
            s0 = s1
        confidence = 0.75
        demos.append((reward_cum, demo, confidence))
        return demos

    def get_reward_of_trajectory(self,trajectory):
        reward_sum = 0.
        for sa in trajectory:
            reward_sum += self.rewards[sa[0]]
        return reward_sum

    def calculate_tp(self, tp_weights, tp_beta):
        n_states,d_states = self.env_features.size()
        n_actions = len(tp_weights)
        tp_normalized = torch.zeros((n_states, n_actions, n_states)).cuda().double()
        for s in range(n_states):
            for a in range(n_actions):
                s_e = self.env_features[s]
                a_weight = tp_weights[a].double()
                res = torch.mm(torch.unsqueeze(s_e,dim=0), a_weight[0:d_states]) + torch.mm(self.env_features, a_weight[d_states:])
                res_mag = torch.norm(res,p=2,dim=1)
                res_q = torch.mul(res_mag,-1.*tp_beta)
                tp_normalized[s,a] = F.softmax(torch.mul(res_q,-1.),dim=0)
        return tp_normalized