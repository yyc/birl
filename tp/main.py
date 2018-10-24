#!/usr/bin/env python
from copy import deepcopy
import gridworld
from mdp import MDP
import numpy as np
from birl import *
from constants import *
from prior import *



def initialize_gridworld(width, height):
    tp_r = torch.Tensor([[0., 1., 0., 1., 0., 0., 0., 0., 0., -1., 0., 0., 0., 0., 0., 0.],
                     [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., -1., 0., 0., 0., 0., 0.]])
    tp_r = torch.unsqueeze(torch.t(tp_r), dim=0)

    tp_u = torch.Tensor([[0., 1., 0., 0., 0., 0., 0., 0., 0., -1., 0., 0., 0., 0., 0., 0.],
                     [0., 0., 1., 0., 1., 0., 0., 0., 0., 0., -1., 0., 0., 0., 0., 0.]])
    tp_u = torch.unsqueeze(torch.t(tp_u), dim=0)

    tp_l = torch.Tensor([[0., 1., 0., 0., 0., -1., 0., 0., 0., -1., 0., 0., 0., 0., 0., 0.],
                     [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., -1., 0., 0., 0., 0., 0.]])
    tp_l = torch.unsqueeze(np.transpose(tp_l), dim=0)

    tp_d = torch.Tensor([[0., 1., 0., 0., 0., 0., 0., 0., 0., -1., 0., 0., 0., 0., 0., 0.],
                     [0., 0., 1., 0., 0., 0., -1., 0., 0., 0., -1., 0., 0., 0., 0., 0.]])
    tp_d = torch.unsqueeze(np.transpose(tp_d), dim=0)

    tp_weights = torch.cat((tp_r, tp_u, tp_l, tp_d),dim=0).cuda()

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


""""
reward = np.arange(0,25)
all_transitions = np.round(np.load("valid_transitions.npy")[0],1)
policy = np.random.randint(0,4,25)
gamma = 0.9
print "V calculation"
start_time = timeit.default_timer()
V = np.zeros(25)
while True:
    delta = 0
    for state in range(25):
        v = V[state]
        V[state] = reward[state] + (np.dot(all_transitions[state, policy[state], :], gamma * V))
        delta = max(delta, np.abs(v - V[state]))
    if delta < 1e-32:
        break
print timeit.default_timer() - start_time

start_time = timeit.default_timer()
T = np.zeros((25,25))
for s in range(25):
    T[s] = all_transitions[s,policy[s]]

a = (np.eye(25) - gamma*T)
b =  np.linalg.inv(a)
Vnew = np.matmul(b,reward)
print timeit.default_timer() - start_time
print np.amax(np.abs(V-Vnew))
print "###########################################"

H = np.zeros((25, 4))
for pind in range(25):
    H[pind, policy[pind]] = 1

start_time = timeit.default_timer()
Q = np.zeros((25,4))
while True:
    delta = 0
    for state in range(25):
        for action in range(4):
            q = Q[state, action]
            currentV = np.array([Q[stemp,policy[stemp]] for stemp in range(25)])
            #currentV = np.dot(H[state],Q[state])
            Q[state, action] = reward[state] + np.dot(all_transitions[state, action, :], gamma * currentV)
            delta = max(delta, np.abs(q - Q[state, action]))
    if delta < 1e-32:
        break
print timeit.default_timer() - start_time

start_time = timeit.default_timer()
T = np.zeros((25, 25))
for s in range(25):
    T[s] = all_transitions[s, policy[s]]

a = (np.eye(25) - gamma * T)
b = np.linalg.inv(a)
Vnew = np.matmul(b, reward)
Q2 = np.zeros((25,4))
for ac in range(4):
    T2 = np.zeros((25, 25))
    for s in range(25):
        T2[s] = all_transitions[s, ac]

    Q2[:,ac] = reward + gamma*np.dot(T2,Vnew)
print timeit.default_timer() - start_time

start_time = timeit.default_timer()
T = np.zeros((25, 25))
for s in range(25):
    T[s] = all_transitions[s, policy[s]]

a = (np.eye(25) - gamma * T)
b = np.linalg.inv(a)
Vnew = np.matmul(b, reward)
Q3 = np.expand_dims(reward,axis=1).repeat(4,axis=1) + gamma * np.dot(all_transitions, Vnew)
print timeit.default_timer() - start_time
print np.amax(np.abs(Q - Q2))
print np.amax(np.abs(Q - Q3))
print "###########################################"
"""
if __name__ == '__main__':
    np.random.seed(3425)

    #Get ground truth transition probability parameters and reward parameters used for generating trajectories
    tp_weights, tp_beta = initialize_gridworld(5, 5)
    gt_reward_weight = [1, 0, 0, 0, 0, 0, 0, 0]

    #Load all the valid dataset
    demos = torch.from_numpy(np.load("valid_trajectories.npy")).cuda()
    demonstration_list = torch.from_numpy(np.load("valid_demonstration_list.npy")).cuda()
    env_features = torch.from_numpy(np.load("valid_environments.npy")).cuda()
    all_transitions = torch.from_numpy(np.load("valid_transitions.npy")).cuda()
    all_starting_pos = demonstration_list[:,0] #Start position for each trajectory. State features and environment policies are not affected by this
    _,_,d_states = env_features.size()
    #Use the last environment for test.
    #For all the trajectories in this environment, goal position is fixed. Start position moves around the grid
    test_env = 24
    test_env_start = 576

    #Populate the training dataset with a random trajectory.
    #This stores the indices of the trajectory
    #Need to be removed in the future since we want all trajectories including the first one to be selected by the algorithm
    limits = np.random.randint(0,test_env_start,1)

    #Define all the MDPs used by the trajectories
    mdps = []
    for eind,_ in enumerate(env_features):
        goal = np.where(env_features[eind,:,0]==1)[0][0]
        temp_mdp = MDP(tp_weights,tp_beta, env_features[eind],gt_reward_weight,0.99,start=all_starting_pos[eind],goal=goal) #start can be anything since it is updated later
        mdps.append(deepcopy(temp_mdp))


    #Get the indices and demonstrations for test set
    test_limits = np.arange(test_env_start,len(demos))
    test_demos = demos[test_env_start:]
    #Calculated the ground truth expected sum of reward for test set
    test_exp_sor = 0.
    for td in test_demos:
        test_exp_sor += mdps[test_env].get_reward_of_trajectory(td)
    test_sor = torch.Tensor([test_exp_sor/len(test_demos),torch.zeros(1)])
    #Get ground truth tp_mean and variance. Since it is a single vector, the variance is zero
    test_tpw = torch.cat((tp_weights.view(-1),torch.zeros(len(tp_weights.view(-1))).cuda()))
    test_tpb = torch.Tensor([tp_beta,0.])



    """
    TEST TRAJECTORIES
    
    D = demos
    trajs = np.load("old/AllD9.npy")
    gl = []
    for traj in trajs:
        for dind,d in enumerate(D):
            if np.all(traj == d):
                gl.append(dind)
                break


    _, _, _, samples = birl(mdps, 0.02, 100, 1.0, demos[gl], demonstration_list[gl], demos[test_limits],
                            demonstration_list[test_limits], 50, 5, d_states, 0.75, gt_reward_weight,
                            PriorDistribution.UNIFORM)

    current_reward = torch.unsqueeze(samples['reward'][0],dim = 0)
    current_beta = torch.unsqueeze(samples['tpbeta'][0],dim = 0)
    current_weights = torch.unsqueeze(samples['tpweights'][0],dim = 0)
    for i in range(1,10):
        current_reward = torch.cat((current_reward, torch.unsqueeze(samples['reward'][i],dim = 0)))
        current_beta = torch.cat((current_beta,torch.unsqueeze(samples['tpbeta'][0],dim = 0)))
        current_weights = torch.cat((current_weights, torch.unsqueeze(samples['tpweights'][0],dim = 0)))

    learned_reward = torch.mean(current_reward,dim=0).cpu().numpy()
    learned_beta = torch.mean(current_beta,dim=0).cpu().numpy()
    learned_weights = torch.mean(current_weights,dim=0).cpu().numpy()

    np.save("lreward.npy",learned_reward)
    np.save("lbeta.npy",learned_beta)
    np.save("lweights.npy", learned_weights)

    """
    """
    D = demos
    Dind = np.arange(len(demos))
    eall = []
    metric_sor, metric_tpw, metric_tpb, samples = birl(mdps, 0.02, 100, 1.0, demos, demonstration_list, demos[test_limits],
                            demonstration_list[test_limits], 10, 1, d_states, 0.75, gt_reward_weight,
                            PriorDistribution.UNIFORM)
    metric_compare = torch.norm(metric_sor.cuda() - test_sor.cuda()) + torch.norm(metric_tpw - test_tpw) + torch.norm(
        metric_tpb.cuda() - test_tpb.cuda())

    current_reward = torch.unsqueeze(samples['reward'][0], dim=0)
    current_beta = torch.unsqueeze(samples['tpbeta'][0], dim=0)
    current_weights = torch.unsqueeze(samples['tpweights'][0], dim=0)
    for i in range(1, len(samples['reward'])):
        current_reward = torch.cat((current_reward, torch.unsqueeze(samples['reward'][i], dim=0)))
        current_beta = torch.cat((current_beta, torch.unsqueeze(samples['tpbeta'][0], dim=0)))
        current_weights = torch.cat((current_weights, torch.unsqueeze(samples['tpweights'][0], dim=0)))

    learned_reward = torch.mean(current_reward, dim=0).cpu().numpy()
    learned_beta = torch.mean(current_beta, dim=0).cpu().numpy()
    learned_weights = torch.mean(current_weights, dim=0).cpu().numpy()

    np.save("lreward.npy", learned_reward)
    np.save("lbeta.npy", learned_beta)
    np.save("lweights.npy", learned_weights)
    """

    #Start the training
    D = demos[limits]
    Dind = limits
    eall = []
    add_more = True
    while (add_more):
        print ("Size of dataset: %d" %(len(Dind)))
        #To store the best metric and samples
        best_metric = None
        best_d = None
        best_samples = None
        best_r_samples = None
        best_tw_samples = None
        best_tb_samples = None
        first_time = True

        #Temporary time measurements
        set_start_time = timeit.default_timer()

        #Enumerate through all trajectories in the training set
        for current_d in range(0,test_env_start):
            start_time = timeit.default_timer()
            if not current_d in Dind: #Only consider trajectories that are not already selected
                #Current training set combines previous selected trajectories + trajectory under investigation
                tempDind = deepcopy(Dind)
                tempDind = np.append(tempDind,current_d)

                #BIRL CODE
                metric_sor, metric_tpw, metric_tpb,samples = birl(mdps, 0.02, 110, 1.0, demos[tempDind], demonstration_list[tempDind], demos[test_limits],
                         demonstration_list[test_limits], 10, 1, d_states, 0.75, gt_reward_weight,
                         PriorDistribution.UNIFORM)
                #Comapre the moments of expected sum of reward
                metric_compare = torch.norm(metric_sor.cuda() - test_sor.cuda()) + torch.norm(metric_tpw-test_tpw) + torch.norm(metric_tpb.cuda() - test_tpb.cuda())

                #Compare and store the best metrics
                if first_time:
                    best_metric = metric_compare
                    best_d = current_d
                    first_time = False
                    best_samples = samples
                else:
                    if metric_compare < best_metric:
                        best_metric = metric_compare
                        best_d = current_d
                        best_samples = samples
            print "Time for iteration: " + str(timeit.default_timer()-start_time)

        #Store the best sample for this iteration
        best_r_samples = torch.unsqueeze(samples['reward'][0],dim = 0)
        best_tb_samples = torch.unsqueeze(samples['tpbeta'][0],dim = 0)
        best_tw_samples = torch.unsqueeze(samples['tpweights'][0],dim = 0)

        for i in range(1,len(samples['reward'])):
            best_r_samples = torch.cat((best_r_samples, torch.unsqueeze(samples['reward'][i], dim=0)))
            best_tb_samples = torch.cat((best_tb_samples, torch.unsqueeze(samples['tpbeta'][0], dim=0)))
            best_tw_samples = torch.cat((best_tw_samples, torch.unsqueeze(samples['tpweights'][0], dim=0)))

        best_r_samples_np = best_r_samples.cpu().numpy()
        best_tb_samples_np = best_tb_samples.cpu().numpy()
        best_tw_samples_np = best_tw_samples.cpu().numpy()

        D = torch.cat((D,demos[best_d].unsqueeze(dim=0)),dim=0)
        Dind = np.append(Dind, best_d)
        Dnp = D.cpu().numpy()
        bmnp = best_metric.cpu().numpy()
        np.save("AllD"+str(len(Dind))+".npy",Dnp)
        np.save("BestMetric"+str(len(Dind))+".npy",bmnp)
        np.save("SampleReward"+str(len(Dind))+".npy",best_r_samples_np)
        np.save("SampleTPWeight"+str(len(Dind))+".npy",best_tw_samples_np)
        np.save("SampleTPBeta"+str(len(Dind))+".npy",best_tb_samples_np)

        #Stopping condition
        if len(D) > 100:
            add_more = False
        print ("Time taken for Full set is %f" % ((timeit.default_timer() - set_start_time)))
    print("Finished BIRL")
    print("Agent Playing")