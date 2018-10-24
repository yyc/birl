#!/usr/bin/env python
import numpy as np
from pdb import set_trace
import torch
import torch.nn.functional as F
import timeit
from copy import deepcopy


class MDP:
    def __init__(self, transition_weights, transition_beta, env_features, reward_weights, gamma, start=None, goal = None, terminal=None, max_value=1000000):
        self.n_states,self.d_states = env_features.size()
        self.n_action,_,_ = transition_weights.size()
        self.env_features = env_features
        self.transitions = self.calculate_tp(transition_weights.cuda(),transition_beta)
        self.n_states,self.n_action,_ = self.transitions.size()
        num_actions = self.transitions.size()[1]
        self.rewards = self.calculate_rewards(torch.Tensor(reward_weights).cuda())
        self.gamma = gamma
        self.states = range(np.shape(self.transitions)[0])
        self.actions = range(np.shape(self.transitions)[1])
        self.max_value = max_value
        self.start = start
        self.goal = (goal*torch.ones(1).long()).cuda()
        self.terminal = len(self.states) - 1
        assert(self.check_valid_mdp())
        self.policy = None
        self.Q = None
        self.changed = False
        self.gt_reward = torch.Tensor(reward_weights).cuda()
        self.gt_tpweights = transition_weights
        self.gt_tpbeta = transition_beta
        self.init_zero = torch.zeros(len(self.states)).cuda()

    def check_valid_mdp(self):
        # Need to be of the form S,A,T(S,A)
        if not ((self.transitions.size()).__len__() == 3):
            return False
        # check that state space size is same in both dims
        if not self.transitions.size()[0] == self.transitions.size()[2]:
            return False
        # check that probabilities are valid

        if torch.equal(torch.gt(torch.max(self.transitions),1.),torch.ones(1).byte().cuda()):
            return False

        if torch.equal(torch.lt(torch.min(self.transitions),0.),torch.ones(1).byte().cuda()):
            return False

        if not torch.equal(torch.sum(self.transitions, dim=2),
                           torch.ones(self.transitions.size()[0], self.transitions.size()[1]).double().cuda()):
            return False

        if self.gamma < 0 or self.gamma > 1:
            return False

        return True

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
        policy = self.init_zero.int()
        for state in self.states:
            policy[state] =torch.randint(0, len(self.actions),(1,)).cuda()
        return policy

    '''
    policy - deterministic policy, maps state to action
    -Deterministic policy evaluation
    '''

    def policy_evaluation(self, policy, theta=1e-8):
        policy = policy.unsqueeze(dim=1)
        policy = policy.unsqueeze(dim=2)
        policy = policy.expand(-1, self.n_action, self.n_states).long()
        T1 = torch.gather(self.transitions, 1, policy)[:, 0, :]
        T1 = T1.cpu().numpy()
        a1 = np.linalg.inv((np.eye(self.n_states) - self.gamma * T1))
        b1 = torch.from_numpy(a1).cuda()
        V = torch.mm(b1, self.rewards.unsqueeze(dim=1)).squeeze(dim=1)
        return V

    def policy_q_evaluation(self, policy):
        V = self.policy_evaluation(policy)
        Q = self.rewards.unsqueeze(dim=1).expand(-1, self.n_action) + self.gamma * torch.mm(
            self.transitions.view(-1, self.n_states), V.unsqueeze(dim=1)).squeeze(dim=1).view(self.n_states,
                                                                                              self.n_action)
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
        #next_state = np.random.choice(self.states, p=self.transitions[state, action, :])
        next_state = torch.distributions.Categorical(probs=self.transitions[state, action, :]).sample()
        return self.rewards[next_state], next_state.cuda()

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
        return demo,reward_cum

    def get_reward_of_trajectory(self,trajectory):
        reward_sum = 0.
        for saind,sa in enumerate(trajectory):
            reward_sum += self.rewards[sa[0]]
            if (sa[0]==self.goal):
                break
        return reward_sum

    def calculate_tp(self, tp_weights, tp_beta):
        n_states,d_states = self.env_features.size()
        n_actions = len(tp_weights)

        t = tp_weights.permute(1,0,2).double()
        t = t.contiguous().view(2*d_states,-1)
        prod3 = torch.mm(self.env_features,t[0:d_states]).view(-1,n_actions,2).unsqueeze(dim=2)
        prod3 = prod3.expand((-1,-1,n_states,-1))
        prod4 = torch.mm(self.env_features,t[d_states:]).view(-1,n_actions,2).permute(1,0,2).unsqueeze(dim=0)
        prod4 = prod4.expand((n_states,-1,-1,-1))

        fin_prod = prod3 + prod4
        res_mag_out = torch.norm(fin_prod, p=2, dim=3)
        res_q_out = torch.mul(res_mag_out, -1. * tp_beta)
        tp_normalized = F.softmax(res_q_out, dim=2)

        """
        print "prod3: " + str(timeit.default_timer() - start_time)

        #prod2 = torch.bmm(s2, t1[:, d_states:, :])
        #prod2 = prod1.view(n_states, n_actions, -1)


        start_time = timeit.default_timer()
        
        tp2 = torch.zeros(self.n_states,self.n_action,self.n_states)
        for s in range(n_states):
            for a in range(n_actions):
                s_e = self.env_features[s]
                a_weight = tp_weights[a].double()
                res = torch.mm(torch.unsqueeze(s_e,dim=0), a_weight[0:d_states]) + torch.mm(self.env_features, a_weight[d_states:])
                #res = torch.mm(torch.unsqueeze(s_e, dim=0), a_weight[0:d_states])
                res_mag = torch.norm(res,p=2,dim=1)
                res_q = torch.mul(res_mag,-1.*tp_beta)
                tp2[s,a] = F.softmax(res_q,dim=0)
        """
        return tp_normalized

    def update_changed(self):
        self.changed = True

    def restore(self):
        self.update_rewards(self.gt_reward)
        self.update_tp(self.gt_tpweights,self.gt_tpbeta)