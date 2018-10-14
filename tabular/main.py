#!/usr/bin/env python
import gridworld
from gridworld import GridWorld
from mdp import MDP
import numpy as np
from birl import *
from constants import *
from prior import *


def initialize_gridworld(width, height):
    # where 24 is a goal state that always transitions to a
    # special zero-reward terminal state (25) with no available actions
    num_states = 6 * 6
    trans_mat = np.zeros((num_states, 4, num_states))

    # NOTE: the following iterations only happen for states 0-23.
    # This means terminal state 25 has zero probability to transition to any state,
    # even itself, making it terminal, and state 24 is handled specially below.

    # Action 1 = down
    for s in range(num_states):
        if s < num_states - width:
            trans_mat[s, 1, s + width] = 1
        else:
            trans_mat[s, 1, s] = 1

    # Action 0 = up
    for s in range(num_states):
        if s >= width:
            trans_mat[s, 0, s - width] = 1
        else:
            trans_mat[s, 0, s] = 1

    # Action 2 = left
    for s in range(num_states):
        if s % width > 0:
            trans_mat[s, 2, s - 1] = 1
        else:
            trans_mat[s, 2, s] = 1

    # Action 3 = right
    for s in range(num_states):
        if s % width < width - 1:
            trans_mat[s, 3, s + 1] = 1
        else:
            trans_mat[s, 3, s] = 1

    # Finally, goal state always goes to zero reward terminal state
    for a in range(4):
        for s in range(num_states):
            trans_mat[num_states - 1, a, s] = 0
        trans_mat[num_states - 1, a, num_states - 1] = 1

    return trans_mat


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
    transitions = initialize_gridworld(6, 6)
    rewards = -0.01*np.ones(36)
    rewards[np.random.randint(1,35,3)] = -0.1
    rewards[35] = 1.
    #mdp = MDP(transitions, initialize_rewards(5, 36), 0.99)
    mdp = MDP(transitions, rewards, 0.99)
    thing = GridWorld(mdp)
    optimal_policy = mdp.policy_iteration()[0]
    demos = get_trajectories(optimal_policy,mdp)
    #demos = thing.record(1)
    print demos
    _,playout = thing.play(optimal_policy)
    policy = birl(mdp, 0.02, 1000, 1.0, demos, 50, 5, PriorDistribution.UNIFORM)
    print "Finished BIRL"
    print "Agent Playing"
    reward, playout = thing.play(policy)
    print "Reward is " + str(reward)
    print "Playout is " + str(playout)
