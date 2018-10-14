#!/usr/bin/env python
import numpy as np
from copy import deepcopy
import random
import math
from scipy.misc import logsumexp
from constants import *
from prior import *
from pdb import set_trace
import timeit


def get_expected_sor(reward_samples, mdps, demos,demonstration_list,gt):
    cum_rewards = []
    gt_reward_sum = 0.
    for rind,rw in enumerate(reward_samples):
        reward_sum = 0.
        for dind,(_,eind) in enumerate(demonstration_list):
            mdps[eind].update_rewards(rw)
            mdps[eind].update_policy()
            trajectories = mdps[eind].get_trajectories()
            mdps[eind].update_rewards(gt)
            reward_sum += mdps[eind].get_reward_of_trajectory(trajectories[0][1])
            if (rind == 0):
                gt_reward_sum += mdps[eind].get_reward_of_trajectory(demos[dind])
        cum_rewards.append(reward_sum)
    return cum_rewards,gt_reward_sum



def birl(mdps, step_size, iterations, r_max, demos, demonstration_list, test_demos, test_demonstration_list, burn_in, sample_freq, d_states, beta, gt_reward_weight,prior):
    if not isinstance(prior, PriorDistribution):
        print "Invalid Prior"
        raise ValueError
    reward_samples,suboptimal_count = PolicyWalk(mdps, step_size, iterations, burn_in, sample_freq, r_max, demos, demonstration_list, d_states, beta, prior)
    expected_sum_rewards, gt_expected_sum_rewards = get_expected_sor(reward_samples,mdps,test_demos,test_demonstration_list,gt_reward_weight)
    mean_val = np.mean(np.array(expected_sum_rewards), axis=0)
    std_val = np.std(np.array(expected_sum_rewards),axis = 0)
    print "Rewards are "
    print mdp.rewards
    # Optimal deterministic policy
    optimal_policy = mdp.policy_iteration()[0]
    print "Computed Optimal BIRL policy"
    return optimal_policy


# probability distribution P, mdp M, step size delta, and perhaps a previous policy
# Returns : List of Sampled Rewards
def PolicyWalk(mdps, step_size, iterations, burn_in, sample_freq, r_max, demos, demonstration_list, d_states, beta, prior):
    reward_samples = []
    # Step 1 - Pick a random reward vector
    current_reward_weight = select_random_reward(d_states,step_size,r_max)
    valid_envs = np.unique(demonstration_list[:,1])
    for eind in valid_envs:
        mdps[eind].update_rewards(current_reward_weight)
        # Step 2 - Policy Iteration per mdp and store it inside the object
        mdps[eind].update_policy()
        mdps[eind].do_policy_q_evaluation()
    # initialize an original posterior, will be useful later
    post_orig = None
    # Step 3
    suboptimal_count = 0

    for i in range(iterations):
        start_time = timeit.default_timer()
        proposed_mdps = deepcopy(mdps)
        # Step 3a - Pick a reward vector uniformly at random from the neighbors of R
        new_reward_weight = mcmc_step(current_reward_weight, proposed_mdps,step_size, r_max)
        # Step 3b - Compute Q for policy under new reward
        for eind in valid_envs:
            proposed_mdps[eind].do_policy_q_evaluation()

        # Step 3c
        if post_orig is None:
            post_orig = compute_log_posterior(mdps, demos, demonstration_list, beta, prior, d_states, r_max)
        # if policy is suboptimal then proceed to 3ci, 3cii, 3ciii
        if suboptimal(proposed_mdps,demonstration_list):
            suboptimal_count += 1
            # 3ci, do policy iteration under proposed reward function
            for _,eind in demonstration_list:
                proposed_mdps[eind].update_policy(use_policy=True)
                proposed_mdps[eind].do_policy_q_evaluation()
            '''
            Take fraction of posterior probability of proposed reward and policy over 
            posterior probability of original reward and policy
            '''
            post_new = compute_log_posterior(proposed_mdps, demos, demonstration_list, beta, prior, d_states, r_max)
            fraction = np.exp(post_new - post_orig)
            if (random.random() < min(1, fraction)):
                for _,eind in demonstration_list:
                    mdps[eind].rewards = proposed_mdps[eind].rewards
                    mdps[eind].policy = proposed_mdps[eind].policy
                post_orig = post_new
                current_reward_weight = new_reward_weight
        else:
            '''
            Take fraction of the posterior probability of proposed reward under original policy over
            posterior probability of original reward and original policy
            '''
            post_new = compute_log_posterior(proposed_mdps, demos, demonstration_list, beta, prior, d_states, r_max)
            fraction = np.exp(post_new - post_orig)
            if (random.random() < min(1, fraction)):
                for _, eind in demonstration_list:
                    mdps[eind].rewards = proposed_mdps[eind].rewards
                post_orig = post_new
                current_reward_weight = new_reward_weight
        # Store samples
        if i >= burn_in:
            if i % sample_freq == 0:
                print i
                reward_samples.append(current_reward_weight)
        print ("Time taken for iteration %d: %f" %(i,(timeit.default_timer()-start_time)))
    # Step 4 - return the reward samples
    return reward_samples,suboptimal_count


# Demos comes in the form (actual reward, demo, confidence)
def compute_log_posterior(mdps, demos, demonstration_list, beta, prior, d_states,r_max):
    log_exp_val = 0
    # go through each demo
    for d,demo in enumerate(demos):
        mdp = mdps[demonstration_list[d,1]]
        # for each state action pair in the demo
        for sa in demo:
            normalizer = []
            # add to the list of normalization terms
            for a in range(np.shape(mdp.transitions)[1]):
                normalizer.append(beta * mdp.Q[sa[0], a])
            '''
            We take the log of the normalizer, because we take exponent in the calling function,
            which gets rid of the log, and leaves the sum of the exponents. Also, we subtract by the log
            instead of dividing because subtracting logs can be rewritten as division
            '''
            log_exp_val = log_exp_val + beta * mdp.Q[sa[0], sa[1]] - logsumexp(normalizer)
            if sa[0] == mdp.goal:
                break
    # multiply by prior
    return log_exp_val + compute_log_prior(prior, d_states, r_max)


def compute_log_prior(prior, d_states, r_max):
    if prior == PriorDistribution.UNIFORM:
        return (- float(d_states)) * np.log(2 * r_max)


def mcmc_step(current_reward, mdps, step_size, r_max):
    #for each dimension of current reward weight, decide if we should move in that dimension or not
    direction = np.array([pow(-1, random.randint(0, 1)) for _ in range(len(current_reward))])
    '''
    move reward at index either +step_size or -step_size, if reward
    is too large, move it to r_max, and if it too small, move to -_rmax
    '''
    new_reward =  np.array([max(min(current_reward[i] + (direction[i] * step_size), r_max), -r_max) for i in range(len(current_reward))])
    for mdp in mdps:
        mdp.update_rewards(new_reward)
    return new_reward


def suboptimal(mdps,demonstration_list):
    # for every state
    for _,eind in demonstration_list:
        policy = mdps[eind].policy
        Q = mdps[eind].Q
        for s in range(np.shape(Q)[0]):
            for a in range(np.shape(Q)[1]):
                if (Q[s, policy[s]] < Q[s, a]):
                    return True
    return False


# Generates a random reward vector in the grid of reward vectors
def select_random_reward(d_states, step_size, r_max):
    rewards = np.random.uniform(-r_max, r_max, d_states)
    # move theese random rewards to a gridpoint
    for i in range(len(rewards)):
        mod = rewards[i] % step_size
        if mod > (step_size / 2):
            rewards[i] = rewards[i] + (step_size - mod)
        else:
            rewards[i] = rewards[i] - mod
    return rewards
