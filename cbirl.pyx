cdef extern from "birl.c":
	int fib(int n)

import numpy as np
cimport cython
cimport numpy as np
from libc.stdlib cimport malloc, free

def main():
	print fib(5)

def select_random_reward(mdp, int step_size, double r_max):
	rewards = np.random.uniform(-r_max, r_max,np.shape(mdp.transitions)[0])
	#move theese random rewards to a gridpoint
	for i in range(len(rewards)):
		mod = rewards[i] % step_size
		if mod > (step_size/2):
			rewards[i] = rewards[i] + (step_size - mod)
		else:
			rewards[i] = rewards[i] - mod
	return rewards

'''Policy Iteration from Sutton and Barto
   assumes discount factor of 0.99
   Deterministic policy iteration
'''
def policy_iteration(mdp, policy=None):
	#initialization
	if policy is None:
		policy = mdp.get_random_policy()

	policy_stable = False
	count = 0
	while not policy_stable:
		#policy evaluation
		V = policy_evaluation(mdp, policy)
		count += 1
		diff_count = 0
		#policy improvement
		policy_stable = True
		for state in mdp.states:
			old_action = policy[state]
			action_vals = np.dot(mdp.transitions[state,:,:], mdp.rewards + mdp.gamma * V).tolist()
			policy[state] = action_vals.index(max(action_vals))
			if not old_action == policy[state]:
				diff_count += 1
				policy_stable = False
	return (policy, V)

'''
policy - deterministic policy, maps state to action
- Deterministic policy evaluation
'''
def policy_evaluation(mdp, policy, theta=0.0001):
	c_policy_eval(mdp, policy, theta)
	print "CALLED"
	V = np.zeros(len(mdp.states))
	while True:
		delta = 0
		for state in mdp.states:
			value = V[state]
			V[state] = np.dot(mdp.transitions[state, policy[state],:], mdp.rewards + mdp.gamma * V)
			delta = max(delta, np.abs(value - V[state]))
			#If divergence and policy has value -inf, return value function early
			if V[state] > mdp.max_value:
				return V
			if V[state] < -mdp.max_value:
				return V
		if delta < theta:
			break
	return V

def c_policy_eval(mdp, policy, double theta):
	cdef int num_states = len(mdp.states)
	cdef int num_actions = len(mdp.actions)
	cdef int i,j,k = 0
	cdef int state = 0
	# DTYPE = np.float64
	# ctypedef np.float64_t DTYPE_t
	# cdef V = np.zeros(len(mdp.states))
	# # cdef double V[num_states]
	# while i < num_states:
	# 	V[i] = 0
	# 	i = i + 1
	# i = 0
	# cdef int size = num_states * num_actions * num_states
	# cdef double* transitions = <double*> malloc(sizeof(double)*size)
	# # cdef double transitions[num_states, num_actions, num_states]
	# while i < num_states:
	# 	j = 0
	# 	while j < num_actions:
	# 		k = 0
	# 		while k < num_states:
	# 			transitions[i][j][k] = mdp.transitions[i, j, k]
	# 			k = k + 1
	# 		j = j + 1
	# 	i = i + 1
	# i = 0
	# cdef double delta, value
	# while True:
	# 	delta = 0
	# 	state = 0
	# 	while state < num_states:
	# 		value = V[state]
	# 		state = state + 1
	# 	return