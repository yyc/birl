import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import timeit
from scipy.stats import multivariate_normal as mnorm
from scipy.stats import norm
import CairlTPLikelihood as IRL
from irl.TrajectorySingleDataset import TrajectorySingleDataset as tData
from torch.utils.data import DataLoader
import timeit

class ActiveLearningGreedy():
    def __init__(self, env_features, prior_tp, discount, beta, horizon):
        #super(ActiveLearningGreedy, self).__init__()
        #Save MDP parameters
        self.env_features = env_features
        self.n_envs,_,self.d_states = np.shape(self.env_features)
        self.n_agents = 1
        self.n_states, self.n_actions, _ = np.shape(prior_tp)
        self.discount = discount
        self.horizon = horizon
        self.beta = beta
        self.reward_prior_mu = np.array([99.,0.,0.,0.,0.,0.,0.])

        tp_prior_right_mu = np.array([[0., 1., 0., 1., 0., 0., 0., 0., -1., 0., 0., 0., 0., 0.],
                         [0., 0., 1., 0., 0., 0., 0., 0., 0., -1., 0., 0., 0., 0.]])
        self.tp_prior_right_mu = np.transpose(tp_prior_right_mu)

        tp_prior_up_mu = np.array([[0., 1., 0., 0., 0., 0., 0., 0., -1., 0., 0., 0., 0., 0.],
                         [0., 0., 1., 0., 1., 0., 0., 0., 0., -1., 0., 0., 0., 0.]])
        self.tp_prior_up_mu = np.transpose(tp_prior_up_mu)

        tp_prior_left_mu = np.array([[0., 1., 0., 0., 0., -1., 0., 0., -1., 0., 0., 0., 0., 0.],
                         [0., 0., 1., 0., 0., 0., 0., 0., 0., -1., 0., 0., 0., 0.]])
        self.tp_prior_left_mu = np.transpose(tp_prior_left_mu)

        tp_prior_down_mu = np.array([[0., 1., 0., 0., 0., 0., 0., 0., -1., 0., 0., 0., 0., 0.],
                         [0., 0., 1., 0., 0., 0., -1., 0., 0., -1., 0., 0., 0., 0.]])
        self.tp_prior_down_mu = np.transpose(tp_prior_down_mu)

        self.tp_prior_mu = np.row_stack((np.expand_dims(self.tp_prior_right_mu, axis=0),
                                              np.expand_dims(self.tp_prior_up_mu, axis=0),
                                              np.expand_dims(self.tp_prior_left_mu, axis=0),
                                              np.expand_dims(self.tp_prior_down_mu, axis=0)))

        self.tp_beta = 100.
        """
        self.tp_prior_right_mu = np.array([0.,1.,0.,1.,0.,0.,0.,0.,-1.,0.,0.,0.,0.,0.])
        self.tp_prior_left_mu = np.array([0.,1.,0.,0.,-1.,0.,0.,0.,-1.,0.,0.,0.,0.,0.])
        self.tp_prior_up_mu = np.array([0.,0.,1.,0.,0.,1.,0.,0.,0.,-1.,0.,0.,0.,0.])
        self.tp_prior_down_mu = np.array([0.,0.,1.,0.,0.,-1.,0.,0.,0.,-1.,0.,0.,0.,0.])
        self.tp_prior = np.row_stack((self.tp_prior_right_mu,self.tp_prior_up_mu,self.tp_prior_left_mu,self.tp_prior_down_mu))
        """
        self.parallel_likelihood = IRL.CairlTP(self.env_features, self.n_actions, 1, discount, beta, horizon)



    def active_select(self,trajectories,demonstration_list):
        tpr = np.load("temp_rewards.npy")
        D = []
        EID = []
        while len(trajectories) > 0:
            for tid, trajectory in enumerate(trajectories):
                start_time = timeit.default_timer()
                if len(D) == 0:
                    D1 = np.array(trajectory)
                    D1 = np.expand_dims(D1,axis=0)
                    EID1 = np.array(demonstration_list[tid])
                    EID1 = np.expand_dims(EID1, axis=0)
                else:
                    D1 = np.array(D)
                    D1 = np.append(D1,np.expand_dims(trajectory,axis=0),axis=0)

                    EID1 = np.array(EID)
                    EID1 = np.append(EID1,demonstration_list[tid])
                D1 = trajectories[0:600]
                EID1 = demonstration_list[0:600]
                temp_rewards,temp_tps,temp_tp_beta = self._get_samples(D1,EID1)
                np.save("temp_rewards.npy",temp_rewards)
                np.save("temp_tp.npy",temp_tps)
                np.save("temp_tp_beta.npy",temp_tp_beta)
                print "something"
                print "Time elapsed: " + str(timeit.default_timer() - start_time)
                raw_input("Sampling test done")
                break
            break



    def _get_samples(self,trajectories,demonstration_list,total_samples = 100,proposal_width = 0.5):
        reward_mu_current = np.random.randn(7)
        tp_mu_right_current = np.random.randn(self.d_states*2,2)
        tp_mu_left_current = np.random.randn(self.d_states*2,2)
        tp_mu_up_current = np.random.randn(self.d_states*2,2)
        tp_mu_down_current = np.random.randn(self.d_states*2,2)

        tp_mu_current = np.row_stack((np.expand_dims(tp_mu_right_current, axis=0),
                                      np.expand_dims(tp_mu_up_current, axis=0),
                                      np.expand_dims(tp_mu_left_current, axis=0),
                                      np.expand_dims(tp_mu_down_current, axis=0)))

        tp_beta_mu_current = np.random.randn()

        posterior_reward = [reward_mu_current]
        posterior_tp = [tp_mu_current]
        posterior_tp_beta = [tp_beta_mu_current]

        #Store current likelihood
        current_likelihood = self._get_likelihood(trajectories,demonstration_list,reward_mu_current,tp_mu_current,tp_beta_mu_current)
        #Store current_priors
        current_reward_prior = np.amax((1e-16,self._get_prior(reward_mu_current,mu = self.reward_prior_mu)))
        current_tp_prior = np.amax((1e-16,self._get_prior(tp_mu_current.flatten(), mu = self.tp_prior_mu.flatten())))
        current_tp_beta_prior = np.amax((1e-16,self._get_prior(tp_beta_mu_current,mu=self.tp_beta,single=True)))

        while(len(posterior_reward) < total_samples):
            print "Size of samples: " +str(len(posterior_reward))
            #Get the new proposal by taking a step
            reward_mu_proposal = mnorm(reward_mu_current, proposal_width).rvs()
            tp_mu_proposal = mnorm(tp_mu_current.flatten(), proposal_width).rvs()
            tp_beta_mu_proposal = norm(tp_beta_mu_current, proposal_width).rvs()
            tp_mu_proposal = tp_mu_proposal.reshape((self.n_actions,2*self.d_states,2))

            #Get the likelihood of the new proposal
            proposal_likelihood = self._get_likelihood(trajectories,demonstration_list,reward_mu_proposal,tp_mu_proposal,tp_beta_mu_proposal)

            #Get the prior of the proposed parameters
            proposal_reward_prior = np.amax((1e-16,self._get_prior(reward_mu_proposal,mu = self.reward_prior_mu)))
            proposal_tp_prior = np.amax((1e-16,self._get_prior(tp_mu_proposal.flatten(), mu = self.tp_prior_mu.flatten())))
            proposal_tp_beta_prior = np.amax((1e-16,self._get_prior(tp_beta_mu_proposal,mu = self.tp_beta)))

            #Get the unnormalized posterior for current and proposed parameters
            p_current = current_likelihood + np.log(current_reward_prior) + np.log(current_tp_prior) + np.log(current_tp_beta_prior)
            #p_current = np.amax([p_current,1e-16])
            p_proposal = proposal_likelihood + np.log(proposal_reward_prior) + np.log(proposal_tp_prior) + np.log(proposal_tp_beta_prior)

            # Accept proposal?
            p_accept = p_proposal - p_current

            accept = np.log(np.random.rand()) < p_accept

            if accept:
                # Update position
                reward_mu_current = reward_mu_proposal.copy()
                tp_mu_current = tp_mu_proposal.copy()
                tp_beta_mu_current = tp_beta_mu_proposal.copy()
                current_likelihood = proposal_likelihood.copy()
                current_reward_prior = proposal_reward_prior.copy()
                current_tp_prior = proposal_tp_prior.copy()


            posterior_reward.append(reward_mu_current)
            posterior_tp.append(tp_mu_current)
            posterior_tp_beta.append(tp_beta_mu_current)

        return posterior_reward,posterior_tp,posterior_tp_beta

    def _get_likelihood(self, trajectories, demonstration_list, reward_mu_proposal, tp_mu_proposal,tp_beta):
        batch_size = len(trajectories)
        trajectoryData = tData(trajectories, demonstration_list,self.env_features)
        trajectoryloader = DataLoader(trajectoryData, batch_size=1, shuffle=False)
        actual_tp = np.zeros((self.n_states,self.n_actions,self.n_states))
        for i_batch, sample_batch in enumerate(trajectoryloader):
            pr, pt = self.parallel_likelihood(sample_batch['trajectory'], sample_batch['agent_id'], sample_batch['env_id'], reward_mu_proposal,tp_mu_proposal,tp_beta,True)
            likelihood = -1*(pr + pt)
            #likelihood = np.exp(likelihood)
            #if likelihood >1:
            #    print "WTF"
            return likelihood

    def _get_prior(self, value, mu,single=False):
        if single:
            pr = norm(mu).pdf(value)
        else:
            pr = mnorm(mu).pdf(value)
        return pr
