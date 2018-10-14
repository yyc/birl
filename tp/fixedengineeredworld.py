import gridworld
import irl.reward_value_iteration as vi
import numpy as np
import numpy.random as rn
from scipy.special import expit as sigmoid
import timeit

class Engineeredworld(gridworld.Gridworld):
    def __init__(self, grid_size, wind, discount, n_demonstrators,demonstrator_thresh):
        super(self.__class__, self).__init__(grid_size, wind, discount)

        self.d_states = 8

        self.all_environments, self.all_transitions, self.all_s0, self.all_rewards = self._get_environments(demonstrator_thresh, self.transition_probability)
        self.n_environments = np.shape(self.all_environments)[0]
        self.n_demonstrators = n_demonstrators
        self.demonstrator_thresh = demonstrator_thresh
        ##############################
        ####Comment this out##########
        ##############################
        #self.all_rewards = (self.all_rewards_temp - np.amin(self.all_rewards_temp))/(np.amax(self.all_rewards_temp) - np.amin(self.all_rewards_temp)) - 1

        self.optimizer = None

        self.debug_val = None

        print "initiatied"

    def update_parameters(self,weights):
        self.ground_theta = weights[0:2]
        self.ground_psi = weights[2:5]
        self.ground_sigma = weights[5:6]
        self.ground_eta = weights[6:10]

    def update_reward_value(self,true_r):
        if len(np.shape(true_r)) == 1:
            true_r = np.expand_dims(true_r,axis=0)
            true_r = np.repeat(true_r,self.n_environments,axis=0)
        self.all_rewards = self._get_rewards(true_r, self.capability_cost, self.n_demonstrators)

    def update_tp_to_godmode(self):
        tp = self.all_transitions
        tp = np.expand_dims(tp,axis=0)
        tp = np.repeat(tp,self.n_demonstrators,axis=0)
        self.all_agent_transitions  = tp

    def update_reward(self):
        self.capabilities = self._get_capabilities(self.demonstrator_thresh)
        self.rtask = self._get_rtask(self.all_environments)
        self.capability_cost, self.all_deltas = self._get_cap_cost(self.n_demonstrators, self.all_environments, self.capabilities)
        self.all_rewards = self._get_rewards(self.rtask,self.capability_cost,self.n_demonstrators)

    def update_reward_neldermead(self,weights):
        self.ground_psi = weights
        self.rtask = self._get_rtask(self.all_environments)
        self.all_rewards = self._get_rewards(self.rtask,self.capability_cost,self.n_demonstrators)

    def update_features(self):
        self.all_environments[:, 1, 1] = 75.
        self.all_environments[:, 6, 1] = 59.
        self.all_environments[:, 11, 1] = 72.
        self.all_environments[:, 16, 1] = 58.

        self.all_environments[:, 2, 1] = 59.
        self.all_environments[:, 7, 1] = 59.
        self.all_environments[:, 12, 1] = 59.
        self.all_environments[:, 17, 1] = 59.

    def update_env(self, random_env, random_tps):
        self.all_environments = random_env
        self.all_transitions = random_tps
        self.n_environments = np.shape(self.all_environments)[0]
        self.all_deltas = self._get_deltas(self.n_demonstrators,self.demonstrator_thresh)
        self.update_reward()


    def _get_single_trajectory(self,agent_id,environment_id,trajectory_length, random_start=False):
        trans_prob = self.all_transitions[environment_id]
        policy,v,q = vi.find_policy(n_states=self.n_states,n_actions=self.n_actions,transition_probabilities=trans_prob,reward=self.all_rewards[environment_id],discount=self.discount,stochastic=False)
        #trajectory = self._follow_policy(policy,trajectory_length, random_start)
        trajectory = self._prob_follow_policy(policy, trajectory_length, random_start,trans_prob,self.all_s0[environment_id])
        return np.array(trajectory), np.array(v), np.array(q)

    def get_trajectory(self,agent_id,eid):
        return self._get_single_trajectory(agent_id,eid,15)

    def get_learner_trajectory(self, learn_tp=None, learn_rewards=None, trajectory_length=15,random_start=False):
        policy, v = vi.find_policy(n_states=self.n_states, n_actions=self.n_actions,
                                   transition_probabilities=learn_tp,
                                   reward=learn_rewards, discount=self.discount,
                                   stochastic=False)

        trajectory = self._follow_policy(policy,trajectory_length,random_start)
        return np.array(trajectory),np.array(v)



    def generate_trajectories(self,trajectory_length,random_start=False):
        print "Generating Trajectories"
        """
            Generates trajectories for the current environment using the optimal policy
            Trajectories are of the form [state_ind,action,reward]
            Append to the trajectories array of shape (n_trajectories, trajectory_length, 3)

            Append the current environment to demo_feature_matrices
            Environment is of the form [state_ind, state_features]. Presently of size [25,4]
            demo_feature_matrices of shape:(n_trajectories, n_states, 4)

            Generate a new environment by selecting new goal and flame grids.
            Find the new optimal policy
            Repeat till n_trajectories are found
        """
        trajectories = []
        values = []
        q_values = []
        demonstration_lists =[]
        for d in range(self.n_demonstrators):
            for e in range(self.n_environments):
                d = 0
                #e = 10
                tr_temp,val_temp,q_temp = self._get_single_trajectory(d,e,trajectory_length)
                trajectories.append(tr_temp)
                values.append(val_temp)
                q_values.append(q_temp)
                #demonstration_lists.append([d,e])
                demonstration_lists.append([d, e])
        return np.array(trajectories), np.array(values),np.array(demonstration_lists),np.array(q_values)

    def _get_environments(self, demonstrator_thresh, transition_probs):
        print "Generating Environments"
        """
        #transition parameter
        tp_r = np.array([[0., 1., 0., 1., 0., 0., 0., 0., -1., 0., 0., 0., 0., 0.,0.],
                         [0., 0., 1., 0., 0., 0., 0., 0., 0., -1., 0., 0., 0., 0.,0.]])
        tp_r = np.expand_dims(np.transpose(tp_r),axis=0)

        tp_u = np.array([[0., 1., 0., 0., 0., 0., 0., 0., -1., 0., 0., 0., 0., 0.,0.],
                         [0., 0., 1., 0., 1., 0., 0., 0., 0., -1., 0., 0., 0., 0.,0.]])
        tp_u = np.expand_dims(np.transpose(tp_u), axis=0)

        tp_l = np.array([[0., 1., 0., 0., 0., -1., 0., 0., -1., 0., 0., 0., 0., 0.,0.],
                         [0., 0., 1., 0., 0., 0., 0., 0., 0., -1., 0., 0., 0., 0.,0.]])
        tp_l = np.expand_dims(np.transpose(tp_l), axis=0)


        tp_d = np.array([[0., 1., 0., 0., 0., 0., 0., 0., -1., 0., 0., 0., 0., 0.,0.],
                         [0., 0., 1., 0., 0., 0., -1., 0., 0., -1., 0., 0., 0., 0.,0.]])
        tp_d = np.expand_dims(np.transpose(tp_d), axis=0)
        """
        tp_r = np.array([[0., 1., 0., 1., 0., 0., 0., 0.,      0., -1., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 1., 0., 0., 0., 0., 0.,      0., 0., -1., 0., 0., 0., 0., 0.]])
        tp_r = np.expand_dims(np.transpose(tp_r), axis=0)

        tp_u = np.array([[0., 1., 0., 0., 0., 0., 0., 0.,      0., -1., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 1., 0., 1., 0., 0., 0.,      0., 0., -1., 0., 0., 0., 0., 0.]])
        tp_u = np.expand_dims(np.transpose(tp_u), axis=0)

        tp_l = np.array([[0., 1., 0., 0., 0., -1., 0., 0.,     0., -1., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 1., 0., 0., 0., 0., 0.,      0., 0., -1., 0., 0., 0., 0., 0.]])
        tp_l = np.expand_dims(np.transpose(tp_l), axis=0)

        tp_d = np.array([[0., 1., 0., 0., 0., 0., 0., 0.,      0., -1., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 1., 0., 0., 0., -1., 0.,     0., 0., -1., 0., 0., 0., 0., 0.]])
        tp_d = np.expand_dims(np.transpose(tp_d), axis=0)
        """
        # transition parameter
        tp_r = np.array([[0., 1., 0., 0., 0., 0., 0., 1., 0., -1., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., -1., 0., 0., 0., 0., 0.]])
        tp_r = np.expand_dims(np.transpose(tp_r), axis=0)

        tp_u = np.array([[0., 1., 0., 0., 0., 0., 0., 0., 0., -1., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 1., 0., 0., 0., 0., 1., 0., 0., -1., 0., 0., 0., 0., 0.]])
        tp_u = np.expand_dims(np.transpose(tp_u), axis=0)

        tp_l = np.array([[0., 1., 0., 0., 0., 0., 0., -1., 0., -1., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., -1., 0., 0., 0., 0., 0.]])
        tp_l = np.expand_dims(np.transpose(tp_l), axis=0)

        tp_d = np.array([[0., 1., 0., 0., 0., 0., 0., 0., 0., -1., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 1., 0., 0., 0., 0., -1., 0., 0., -1., 0., 0., 0., 0., 0.]])
        tp_d = np.expand_dims(np.transpose(tp_d), axis=0)
        """
        tp_weights = np.row_stack((tp_r,tp_u,tp_l,tp_d))

        tp_beta = 100.

        #Template for standard environment
        template_env = 1.*np.zeros([self.n_states,self.d_states])
        #red states
        red_states = [7,12,17]
        template_env[red_states,0] = -1.
        template_env[:,7] = 1.
        for s in range(self.n_states):
            x,y = self.int_to_point(s)
            template_env[s,1] = x
            template_env[s,2] = y
        #SET right allow
        template_env[:,3] = 1.
        template_env[:,4] = 1.
        template_env[:,5] = 1.
        template_env[:,6] = 1.

        #Right border sells
        template_env[4, 3] = 0.
        template_env[9, 3] = 0.
        template_env[14,3] = 0.
        template_env[19,3] = 0.
        template_env[24,3] = 0.

        # Up border sells
        template_env[20, 4] = 0.
        template_env[21, 4] = 0.
        template_env[22, 4] = 0.
        template_env[23, 4] = 0.
        template_env[24, 4] = 0.

        #Left border sells
        template_env[0, 5] = 0.
        template_env[5, 5] = 0.
        template_env[10,5] = 0.
        template_env[15,5] = 0.
        template_env[20,5] = 0.


        #Down border sells
        template_env[0, 6] = 0.
        template_env[1, 6] = 0.
        template_env[2, 6] = 0.
        template_env[3, 6] = 0.
        template_env[4, 6] = 0.

        template_transition = transition_probs.copy()

        #for right action
        r_starting = [6,7,11,12,16,17]
        for rs in r_starting:
            template_transition[rs,0,rs] += template_transition[rs,0,rs+1]
            template_transition[rs,0,rs+1] = 0.
            template_env[rs,3] = 0.

        #for up action
        u_starting = [2]
        for us in u_starting:
            template_transition[us,1,us] += template_transition[us,1,us+self.grid_size]
            template_transition[us,1,us+self.grid_size] = 0.
            template_env[us,4] = 0.

        # for left action
        l_starting = [8, 7, 13, 12, 18, 17]
        for ls in l_starting:
            template_transition[ls, 2, ls] += template_transition[ls, 2, ls - 1]
            template_transition[ls, 2, ls - 1] = 0.
            template_env[ls,5] = 0.

        # for down action
        d_starting = [7]
        for ds in d_starting:
            template_transition[ds, 3, ds] += template_transition[ds, 3, ds - self.grid_size]
            template_transition[ds, 3, ds - self.grid_size] = 0.
            template_env[ds,6] = 0.

        all_s0 = []
        all_environments = []
        all_transitions = []
        all_rewards = []

        env_count = 0
        start_time = timeit.default_timer()
        for spos in range(self.n_states):
            for gpos in range(self.n_states):
                if not (gpos == spos):
                    #print "Creating Env: " + str(env_count)
                    env_count += 1
                    current_env = template_env.copy()
                    current_env[gpos,0] = 1.
                    #current_reward = np.dot(current_env,np.array([2.6141,  0.0030,  0.0677]))
                    #current_reward = current_reward + 0.9433
                    #ind = np.where(current_reward <0)
                    #current_reward[ind] = 0.
                    current_reward = 0.*np.ones(self.n_states)
                    current_reward[gpos] = 99.
                    all_environments.append(current_env)
                    all_s0.append(spos)

                    tp_normalized = np.zeros((self.n_states, self.n_actions, self.n_states))
                    for s in range(self.n_states):
                        for a in range(self.n_actions):
                            s_e = current_env[s]
                            a_weight = tp_weights[a]
                            res = np.dot(s_e, a_weight[0:self.d_states]) + np.dot(current_env, a_weight[self.d_states:])
                            res_mag = np.linalg.norm(res,axis=1)
                            res_q = -1 * tp_beta * res_mag
                            max_tp = np.amax(res_q)
                            s_tp = np.sum(np.exp(res_q - max_tp))
                            tp_normalized[s, a] = np.exp(res_q - max_tp) / s_tp

                            """
                            for sprime in range(self.n_states):
                                s_e = current_env[s]
                                sprime_e = current_env[sprime]
                                a_weight = np.transpose(tp_weights[a])
                                res = np.dot(s_e,a_weight[0:7]) + np.dot(sprime_e,a_weight[7:14])
                                res_mag = np.linalg.norm(res)
                                res_q = -1*tp_beta*res_mag
                                tp_unnormalized.append(res_q)
                            tp_unnormalized = np.array(tp_unnormalized)
                            min_tp = np.amin(tp_unnormalized)
                            s_tp = np.sum(np.exp(tp_unnormalized-min_tp))
                            tp_normalized[s,a] = np.exp(tp_unnormalized-min_tp)/s_tp
                            """

                    #all_transitions.append(template_transition.copy())
                    all_transitions.append(tp_normalized)
                    all_rewards.append(current_reward)
                #if env_count > 200:
                #    break
            #if env_count > 200:
            #    break
        print "Time elapsed: " + str(timeit.default_timer() - start_time)
        return np.array(all_environments), np.array(all_transitions), np.array(all_s0), np.array(all_rewards)

    def calculate_rewards(self,curr_env,curr_thresh):
        capabilities = self._get_capabilities(curr_thresh)
        rtask = self._get_rtask(curr_env)
        capability_cost, deltas = self._get_cap_cost(1, curr_env, capabilities)
        all_rewards = self._get_rewards(rtask,capability_cost,1)
        return all_rewards




    def _get_rewards(self,rtask,capability_cost,n_demonstrator):
        rtask_stacked = np.array([rtask]*n_demonstrator)
        #return self.ground_theta[0] * rtask_stacked + self.ground_theta[1] * capability_cost
        return self.ground_theta[0] * rtask_stacked

    def generate_learner_environment(self, demonstrator_thresh,learn_thresh):
        learn_env = np.zeros([self.n_states,self.d_states])
        learn_reward = np.zeros(self.n_states)
        learn_delta = np.zeros((self.n_states,self.d_agents))

        temp_limits = np.sort(np.unique([demonstrator_thresh[x] for x in range(self.n_demonstrators)]))
        temp_limits_extreme = np.concatenate((temp_limits, [np.max(temp_limits) + 10]))
        grid_temperatures = np.array(
            [temp_limits_extreme[ind] + np.random.randint(1, temp_limits_extreme[ind + 1] - temp_limits_extreme[ind], 1)
             for ind
             in range(len(temp_limits))])
        curr_goal_grid = np.random.randint(low=0,high=self.n_states)
        learn_env[curr_goal_grid] = np.array([1.,0.,1.])
        learn_delta[curr_goal_grid] = np.array([0.,learn_thresh])
        for s in range(self.n_states):
            if not s==curr_goal_grid:
                toss = np.random.rand()
                if (toss <= 0.75):
                    fire_temp = 0.
                else:
                    fire_temp = grid_temperatures[np.random.randint(0, len(grid_temperatures), 1)]
                learn_env[s] = np.array([0.,fire_temp,1.])
                learn_delta[s] = np.array([fire_temp,learn_thresh])
        learn_reward = self.calculate_rewards(learn_env,learn_delta)
        learn_tp = self.transition_probability
        return learn_env,learn_reward,learn_delta,learn_tp

    def _follow_policy(self, policy,trajectory_length, random_start):
        if random_start:
            sx, sy = rn.randint(self.grid_size), rn.randint(self.grid_size)
        else:
            sx, sy = 0, 0
        trajectory = []
        for _ in range(trajectory_length):
            action = self.actions[policy[self.point_to_int((sx, sy))]]
            if (0 <= sx + action[0] < self.grid_size and
                            0 <= sy + action[1] < self.grid_size):
                next_sx = sx + action[0]
                next_sy = sy + action[1]
            else:
                next_sx = sx
                next_sy = sy

            state_int = self.point_to_int((sx, sy))
            action_int = self.actions.index(action)
            next_state_int = self.point_to_int((next_sx, next_sy))
            # reward = self.reward(next_state_int)
            #reward = curr_reward[next_state_int]
            trajectory.append((state_int, action_int))
            sx = next_sx
            sy = next_sy
        return trajectory

    def _prob_follow_policy(self, policy,trajectory_length,random_start,trans_prob,starting_state = 0):
        if random_start:
            state_int = rn.randint(self.grid_size)
        else:
            state_int = starting_state
        trajectory = []
        for _ in range(trajectory_length):
            action_int = policy[state_int]
            trajectory.append((state_int,action_int))
            probs = trans_prob[state_int,action_int]
            state_int = np.random.choice(self.n_states,p=probs)
        return trajectory

    def _get_deltas(self, n_demonstrators, environments, capabilities):
        return (np.array(
            [[np.column_stack((current_env, np.array([capabilities[a]] * self.n_states))) for
              current_env in environments] for a in range(n_demonstrators)]))

    def _get_capabilities(self, demonstrator_thresh):
        return np.array([np.dot(self.ground_sigma,dt) for dt in demonstrator_thresh])

    def _get_rtask(self, environment):
        return np.dot(environment,self.ground_psi)

    def _get_cap_cost(self, n_demonstrators, environments, capabilities):
        all_deltas = self._get_deltas(n_demonstrators, environments, capabilities)
        x = np.dot(all_deltas,self.ground_eta)
        cap_cost = sigmoid(x)
        return cap_cost, all_deltas

    def update_network(self, pars):
        self.optimizer = pars

    def update_reward_from_network(self):
        self.rtask = None
        self.capabilities = None
        self.capability_cost = np.zeros(
            (self.n_demonstrators, np.shape(self.all_environments)[0], np.shape(self.all_environments)[1]))
        self.all_rewards = np.zeros(
            (self.n_demonstrators, np.shape(self.all_environments)[0], np.shape(self.all_environments)[1]))
        self.debug_val = np.zeros(
            (self.n_demonstrators, np.shape(self.all_environments)[0], np.shape(self.all_environments)[1]))
        self.rtask = np.zeros(
            (np.shape(self.all_environments)[0], np.shape(self.all_environments)[1]))

        for aind,a in enumerate(self.demonstrator_thresh):
            for eind,e in enumerate(self.all_environments):
                temp_rj, temp_rtask, temp_val = self.calculate_rewards_from_network(e,a,eind)
                if (aind >0) and not(np.all(temp_rtask == self.rtask[eind])):
                    print("WTF")
                self.all_rewards[aind,eind] = temp_rj
                self.rtask[eind] = temp_rtask
                self.capability_cost[aind, eind] = temp_rj - temp_rtask
                self.debug_val[aind,eind] = temp_val

    def calculate_rewards_from_network(self, e,a,eind,transition_prob = None):
        return self.optimizer.get_reward(e, a, eind,transitionProb=transition_prob)

    def _get_agent_transitions(self, all_environments,all_transitions, demonstrator_thresh):
        agent_transitions = np.zeros(np.append(len(demonstrator_thresh),np.shape(all_transitions)))
        for agind, ag in enumerate(demonstrator_thresh):
            for envind, env in enumerate(all_environments):
                mod = ag - env[:,1]
                #mod2 = np.array([mod,np.ones(np.shape(mod)),mod,np.ones(np.shape(mod))])
                mod2 = np.array([mod]*self.n_actions)
                mod3 = np.repeat(mod2[np.newaxis,:,:],self.n_states,axis=0)
                mod4 = sigmoid(100*mod3)
                tp = all_transitions[envind] * mod4
                ut = all_transitions[envind]
                diff = 1. - np.sum(tp,axis=2)
                for s in range(self.n_states):
                    tp[s,:,s] = tp[s,:,s] + diff[s,:]
                agent_transitions[agind,envind] = tp
        return np.array(agent_transitions)




