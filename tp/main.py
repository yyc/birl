#!/usr/bin/env python
from copy import deepcopy

import gridworld
from gridworld import GridWorld
from mdp import MDP
import numpy as np
from birl import *
from constants import *
from prior import *



def initialize_gridworld(width, height):
    tp_r = np.array([[0., 1., 0., 1., 0., 0., 0., 0., 0., -1., 0., 0., 0., 0., 0., 0.],
                     [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., -1., 0., 0., 0., 0., 0.]])
    tp_r = np.expand_dims(np.transpose(tp_r), axis=0)

    tp_u = np.array([[0., 1., 0., 0., 0., 0., 0., 0., 0., -1., 0., 0., 0., 0., 0., 0.],
                     [0., 0., 1., 0., 1., 0., 0., 0., 0., 0., -1., 0., 0., 0., 0., 0.]])
    tp_u = np.expand_dims(np.transpose(tp_u), axis=0)

    tp_l = np.array([[0., 1., 0., 0., 0., -1., 0., 0., 0., -1., 0., 0., 0., 0., 0., 0.],
                     [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., -1., 0., 0., 0., 0., 0.]])
    tp_l = np.expand_dims(np.transpose(tp_l), axis=0)

    tp_d = np.array([[0., 1., 0., 0., 0., 0., 0., 0., 0., -1., 0., 0., 0., 0., 0., 0.],
                     [0., 0., 1., 0., 0., 0., -1., 0., 0., 0., -1., 0., 0., 0., 0., 0.]])
    tp_d = np.expand_dims(np.transpose(tp_d), axis=0)

    tp_weights = np.row_stack((tp_r, tp_u, tp_l, tp_d))

    tp_beta = 100.
    return tp_weights, tp_beta


def initialize_rewards(dims, num_states):
    weights = np.random.normal(0, 0.25, dims)
    rewards = dict()
    for i in range(num_states):
        rewards[i] = np.dot(weights, np.random.normal(-3, 1, dims))
    # Give goal state higher value
    rewards[num_states - 1] = 10
    return rewards


def get_trajectories(optimal_policy, mdp):
    demos = []
    demo = []
    s0 = 0
    reward_cum = 0.
    go_on = True
    while go_on:
        a0 = optimal_policy[s0]
        _,s1 = mdp.act(s0,a0)
        demo.append((s0,a0))
        reward_cum += mdp.rewards[s0]
        if s0 == 35 or len(demo) > 30:
            go_on = False
        s0 = s1
    confidence = 400.
    demos.append((reward_cum,demo,confidence))
    return demos




if __name__ == '__main__':
    tp_weights, tp_beta = initialize_gridworld(5, 5)
    np.random.seed(3425)
    #Load all the valid dataset
    demos = np.load("valid_trajectories.npy")
    demonstration_list = np.load("valid_demonstration_list.npy")
    env_features = np.load("valid_environments.npy")
    all_transitions = np.load("valid_transitions.npy")
    all_starting_pos = demonstration_list[:,0]
    _,_,d_states = np.shape(env_features)
    limits = np.random.randint(0,len(demos),50)
    test_limits = np.random.randint(0,len(demos),30)
    gt_reward_weight = [1,0,0,0,0,0,0,0]

    unique_envs = np.unique(demonstration_list[:,1])

    mdps = []
    #for _,eind in demonstration_list[limits]:
    for eind,_ in enumerate(env_features):
        goal = np.where(env_features[eind,:,0]==1)[0][0]
        temp_mdp = MDP(tp_weights,tp_beta, env_features[eind],gt_reward_weight,0.99,start=all_starting_pos[eind],goal=goal)
        mdps.append(deepcopy(temp_mdp))

    policy = birl(mdps, 0.02, 100, 1.0, demos[limits], demonstration_list[limits],demos[test_limits],demonstration_list[test_limits], 50, 5, d_states, 0.75, gt_reward_weight, PriorDistribution.UNIFORM)
    print("Finished BIRL")
    print("Agent Playing")
    reward, playout = thing.play(policy)
    print("Reward is " + str(reward))
    print("Playout is " + str(playout))
