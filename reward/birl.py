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
from scipy.stats import norm
from scipy.stats import multivariate_normal as mnorm
import torch

def get_expected_sor(samples, mdps, demonstration_list):
    cum_rewards = []
    gt_reward_sum = 0.
    reward_samples = samples
    n_samples = len(reward_samples)
    valid_envs = torch.unique(demonstration_list[:, 1]).cuda()
    for sample in range(n_samples):
        rw = reward_samples[sample]

        for eind in valid_envs:
            mdps[eind].update_rewards(rw)
            #mdps[eind].update_tp(tpweight,tpbeta)
            mdps[eind].update_policy()
        reward_sum = 0.
        for dind,(spos,eind) in enumerate(demonstration_list):
            mdps[eind].restore()
            mdps[eind].start = spos
            trajectories,rtemp = mdps[eind].get_trajectories()
            reward_sum += rtemp
        cum_rewards.append(reward_sum/len(demonstration_list))
    return cum_rewards



def birl(mdps, step_size, iterations, r_max, demos, demonstration_list, test_demos, test_demonstration_list, burn_in, sample_freq, d_states, beta, gt_reward_weight,prior):
    #if not isinstance(prior, PriorDistribution):
    #    print("Invalid Prior")
    #    raise ValueError
    """
    step_size = step_size
    iterations =iterations
    r_max = r_max
    demos = demos
    demonstration_list = torch.from_numpy(demonstration_list)
    test_demos = torch.from_numpy(test_demos)
    test_demonstration_list = torch.from_numpy(test_demonstration_list)
    burn_in = burn_in*torch.ones(1)
    sample_freq = sample_freq*torch.ones(1)
    d_states = d_states*torch.ones(1)
    beta= beta*torch.ones(1)
    gt_reward_weight = torch.from_numpy(np.array(gt_reward_weight))
    """
    samples, suboptimal_count = PolicyWalk(mdps, step_size, iterations, burn_in, sample_freq, r_max, demos,
                                           demonstration_list, d_states, beta, prior)
    cum_exp_sor = get_expected_sor(samples, mdps, test_demonstration_list)
    cum_exp_sor = torch.from_numpy(np.array(cum_exp_sor))
    metric1 = torch.Tensor([torch.mean(cum_exp_sor),torch.std(cum_exp_sor)])
    """
    mean_val = np.mean(np.array(expected_sum_rewards), axis=0)
    std_val = np.std(np.array(expected_sum_rewards),axis = 0)
    print("Rewards are ")
    print(mdp.rewards)
    # Optimal deterministic policy
    optimal_policy = mdp.policy_iteration()[0]
    print("Computed Optimal BIRL policy")
    return optimal_policy
    """
    return metric1,samples
    #return None


# probability distribution P, mdp M, step size delta, and perhaps a previous policy
# Returns : List of Sampled Rewards
def stick_to_grid(uncorrected, step_size):
    return torch.mul(torch.round(torch.div(uncorrected, step_size)), step_size)


def select_random_tpweights(d_states, step_size):
    tp_r = torch.Tensor([[0., 1., 0., 0., 0., 0., 0., 1., 0., -1., 0., 0., 0., 0., 0., 0.],
                     [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., -1., 0., 0., 0., 0., 0.]])
    tp_r = tp_r.view(-1)
    tp_u = torch.Tensor([[0., 1., 0., 0., 0., 0., 0., 0., 0., -1., 0., 0., 0., 0., 0., 0.],
                     [0., 0., 1., 0., 0., 0., 0., 1., 0., 0., -1., 0., 0., 0., 0., 0.]])
    tp_u = tp_u.view(-1)

    tp_l = torch.Tensor([[0., 1., 0., 0., 0., 0., 0., -1., 0., -1., 0., 0., 0., 0., 0., 0.],
                     [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., -1., 0., 0., 0., 0., 0.]])
    tp_l = tp_l.view(-1)

    tp_d = torch.Tensor([[0., 1., 0., 0., 0., 0., 0., 0., 0., -1., 0., 0., 0., 0., 0., 0.],
                     [0., 0., 1., 0., 0., 0., 0., -1., 0., 0., -1., 0., 0., 0., 0., 0.]])
    tp_d = tp_d.view(-1)

    tp_r_new = torch.distributions.MultivariateNormal(tp_r,0.05*torch.eye(len(tp_r))).sample()
    tp_r_new = stick_to_grid(tp_r_new,step_size)

    tp_u_new = torch.distributions.MultivariateNormal(tp_u,0.05*torch.eye(len(tp_u))).sample()
    tp_u_new = stick_to_grid(tp_u_new, step_size)

    tp_l_new = torch.distributions.MultivariateNormal(tp_l,0.05*torch.eye(len(tp_l))).sample()
    tp_l_new = stick_to_grid(tp_l_new, step_size)

    tp_d_new = torch.distributions.MultivariateNormal(tp_d,0.05*torch.eye(len(tp_d))).sample()
    tp_d_new = stick_to_grid(tp_d_new, step_size)

    tp_r_new = torch.unsqueeze(tp_r_new.view(-1,2),dim=0)
    tp_u_new = torch.unsqueeze(tp_u_new.view(-1, 2), dim=0)
    tp_l_new = torch.unsqueeze(tp_l_new.view(-1, 2), dim=0)
    tp_d_new = torch.unsqueeze(tp_d_new.view(-1, 2), dim=0)

    tp_weights = torch.cat((tp_r_new, tp_u_new, tp_l_new, tp_d_new),dim=0).cuda()
    return tp_weights


def select_random_tpbeta(step_size):
    new_beta = torch.distributions.Normal(100.,1.).sample().cuda()
    new_beta = stick_to_grid(new_beta,step_size).double()
    return new_beta



def PolicyWalk(mdps, step_size, iterations, burn_in, sample_freq, r_max, demos, demonstration_list, d_states, beta, prior):
    reward_samples = []
    #tp_weight_samples = []
    #tp_flat_weight_samples =[]
    #tp_beta_samples = []
    # Step 1 - Pick a random reward vector
    current_reward_weight = select_random_reward(d_states,step_size,r_max)
    #current_tp_weights = select_random_tpweights(d_states,step_size)
    #current_tp_beta = select_random_tpbeta(5.)
    valid_envs = torch.unique(demonstration_list[:,1]).cuda()
    #print ("NEED TO PARSE %d ENVS" %len(valid_envs))
    print "UPDATE INITIAL POLICY"
    for eind in valid_envs:
        print ("ENV: %d out of %d" %(eind,len(valid_envs)))
        mdps[eind].update_rewards(current_reward_weight)
        #mdps[eind].update_tp(current_tp_weights,current_tp_beta)

        # Step 2 - Policy Iteration per mdp and store it inside the object
        mdps[eind].update_policy()
        mdps[eind].do_policy_q_evaluation()
    # initialize an original posterior, will be useful later
    post_orig = None
    # Step 3
    suboptimal_count = 0
    print "GETTING SAMPLES"
    for i in range(iterations):
        print ("Iteration %d out of %d" %(i,iterations))
        start_time = timeit.default_timer()
        proposed_mdps = deepcopy(mdps)
        # Step 3a - Pick a reward vector uniformly at random from the neighbors of R
        new_reward_weight = mcmc_step(current_reward_weight, proposed_mdps, valid_envs, step_size, r_max)
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
            fraction = torch.exp(post_new - post_orig)
            prob_of_one = torch.min(torch.ones(1),fraction.cpu().float())
            if torch.equal(torch.distributions.Bernoulli(prob_of_one).sample(),torch.ones(1)):
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
            fraction = torch.exp(post_new - post_orig)
            prob_of_one = torch.min(torch.ones(1), fraction.cpu().float())
            if (torch.distributions.Bernoulli(prob_of_one).sample(),torch.ones(1)):
                for _, eind in demonstration_list:
                    mdps[eind].rewards = proposed_mdps[eind].rewards
                post_orig = post_new
                current_reward_weight = new_reward_weight

        # Store samples
        if i >= burn_in:
            if i % sample_freq == 0:
                #print(i)
                reward_samples.append(current_reward_weight)
        #print ("Time taken for iteration %d: %f" %(i,(timeit.default_timer()-start_time)))
    # Step 4 - return the reward samples
    samples = reward_samples
    return samples,suboptimal_count


# Demos comes in the form (actual reward, demo, confidence)
def compute_log_prior_tpweights(tp_weight):
    mu_tp_r = torch.Tensor([[0., 1., 0., 0., 0., 0., 0., 1., 0., -1., 0., 0., 0., 0., 0., 0.],
                     [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., -1., 0., 0., 0., 0., 0.]])
    mu_tp_r = mu_tp_r.view(-1)

    mu_tp_u = torch.Tensor([[0., 1., 0., 0., 0., 0., 0., 0., 0., -1., 0., 0., 0., 0., 0., 0.],
                     [0., 0., 1., 0., 0., 0., 0., 1., 0., 0., -1., 0., 0., 0., 0., 0.]])
    mu_tp_u = mu_tp_u.view(-1)

    mu_tp_l = torch.Tensor([[0., 1., 0., 0., 0., 0., 0., -1., 0., -1., 0., 0., 0., 0., 0., 0.],
                     [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., -1., 0., 0., 0., 0., 0.]])
    mu_tp_l = mu_tp_l.view(-1)

    mu_tp_d = torch.Tensor([[0., 1., 0., 0., 0., 0., 0., 0., 0., -1., 0., 0., 0., 0., 0., 0.],
                     [0., 0., 1., 0., 0., 0., 0., -1., 0., 0., -1., 0., 0., 0., 0., 0.]])
    mu_tp_d = mu_tp_d.view(-1)

    prob = torch.zeros(1).cuda()

    prob += prob + torch.distributions.MultivariateNormal(mu_tp_r, 0.05 * torch.eye(len(mu_tp_r))).log_prob(
        tp_weight[0].view(-1).cpu()).cuda()
    prob += prob + torch.distributions.MultivariateNormal(mu_tp_u, 0.05 * torch.eye(len(mu_tp_u))).log_prob(
        tp_weight[1].view(-1).cpu()).cuda()
    prob += prob + torch.distributions.MultivariateNormal(mu_tp_l, 0.05 * torch.eye(len(mu_tp_l))).log_prob(
        tp_weight[2].view(-1).cpu()).cuda()
    prob += prob + torch.distributions.MultivariateNormal(mu_tp_d, 0.05 * torch.eye(len(mu_tp_d))).log_prob(
        tp_weight[3].view(-1).cpu()).cuda()
    return prob.double()


def compute_log_prior_tpbeta(tp_beta):
    return torch.distributions.Normal(100. * torch.ones(1).double(), 1. * torch.ones(1).double()).log_prob(
        tp_beta.cpu()).cuda()


def compute_log_posterior(mdps, demos, demonstration_list, beta, prior, d_states,r_max):
    log_exp_val = 0
    # go through each demo
    for d,demo in enumerate(demos):
        mdp = mdps[demonstration_list[d,1]]
        # for each state action pair in the demo
        for sind,sa in enumerate(demo):
            n_actions = mdp.transitions.size()[1]
            normalizer = torch.zeros(n_actions).cuda()
            # add to the list of normalization terms
            for a in range(n_actions):
                normalizer[a] = torch.mul(mdp.Q[sa[0], a],beta)
            '''
            We take the log of the normalizer, because we take exponent in the calling function,
            which gets rid of the log, and leaves the sum of the exponents. Also, we subtract by the log
            instead of dividing because subtracting logs can be rewritten as division
            '''
            log_exp_val = log_exp_val + torch.mul(mdp.Q[sa[0], sa[1]],beta) - torch.logsumexp(normalizer,dim=0).double() #policy
            if sind < len(demos)-1:
                tpval = torch.max(1e-16*torch.ones(1).cuda().double(),mdp.transitions[sa[0],sa[1],demo[sind+1,0]])
                log_exp_val = log_exp_val + torch.log(tpval)
            if torch.equal(sa[0],mdp.goal):
                break
    # multiply by prior
    reward_prior = compute_log_prior(prior, d_states, r_max)
    return log_exp_val + reward_prior


def compute_log_prior(prior, d_states, r_max):
    if prior == PriorDistribution.UNIFORM:
        return torch.mul(torch.log(2. * r_max*torch.ones(1).cuda()),-1.*d_states).double()


def mcmc_step(current_reward, mdps, valid_envs, step_size, r_max):
    possible_dirs = torch.Tensor([-1,1]).cuda()
    indices = torch.randint(0,2,(current_reward.size()[0],)).long().cuda()
    direction = possible_dirs[indices]
    #for each dimension of current reward weight, decide if we should move in that dimension or not
    #direction = np.array([pow(-1, random.randint(0, 1)) for _ in range(len(current_reward))])
    '''
    move reward at index either +step_size or -step_size, if reward
    is too large, move it to r_max, and if it too small, move to -_rmax
    '''
    new_reward = current_reward + torch.mul(direction,step_size)
    new_reward = torch.min(new_reward,torch.mul(torch.ones(new_reward.size()[0]).cuda(),r_max))
    new_reward = torch.max(new_reward, torch.mul(torch.ones(new_reward.size()[0]).cuda(), -1*r_max))


    for eind in valid_envs:
        mdps[eind].update_rewards(new_reward)
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
    rewards = torch.distributions.Uniform(-1*r_max,r_max).sample(torch.Size([d_states])).cuda()
    # move theese random rewards to a gridpoint
    corrected = stick_to_grid(rewards,step_size)
    return corrected
