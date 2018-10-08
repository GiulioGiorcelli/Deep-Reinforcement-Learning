from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future
#
# Note: gym changed from version 0.7.3 to 0.8.0
# MountainCar episode length is capped at 200 in later versions.
# This means your agent can't learn as much in the earlier episodes
# since they are no longer as long.   
#
# Adapt Q-Learning script to use N-step method instead

import gym
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from gym import wrappers
from datetime import datetime

# code we already wrote
import q_learning_mtn
from q_learning_mtn import plot_cost_to_go, FeatureTransformer, Model, plot_running_avg



class SGDRegressor:
	def __init__(self, **kwargs):
		self.w = None
		self.lr = 10e-3
		
	def partial_fit(self, X, Y):
		if self.w is None:
			D = X.shape[1]
			self.w = np.random.random(D) / np.sqrt(D)
			
		self.w += self.lr*(Y - X.dot(self.w)).dot(X)
		
	def predict(self, X):
		return(X.dot(self.w))
		


# replace SKLearn Regressor with my new class		
q_learning_mtn.SGDRegressor = SGDRegressor



# play one function
def play_one(model, eps, gamma, n=5):

	observation = env.reset()
	done = False
	totalreward = 0
	rewards = []
	states = []
	actions = []
	iters = 0
	
	multiplier = np.array([gamma]*n)**np.arange(n)
	
	while not done and iters < 200:
	
		action = model.sample_action(observation, eps)
		
		states.append(observation)
		actions.append(action)
		
		previous_observation = observation
		observation, reward, done, info = env.step(action)
		
		rewards.append(reward)
		
		#update the model
		if len(rewards) > n:
			return_up_to_predictions = multiplier.dot(rewards[-n:])
			G = return_up_to_predictions + (gamma**n)*np.max(model.predict(observation)[0])
			model.update(states[-n], actions[-n], G)
			
		totalreward += reward
		iters += 1
		
	if n == 1:
		rewards = []
		states = []
		actions = []
	else:
		rewards = rewards[-n+1:]
		states = states[-n+1:]
		actions = actions[-n+1:]
	
	# according to documentation, observation must be >= 0.5 to reach goal	
	if observation[0] >= 0.5:
		print("goal reached")
		while len(rewards) > 0:
			G = multiplier[:len(rewards)].dot(rewards)
			model.update(states[0], actions[0], G)
			rewards.pop(0)
			states.pop(0)
			actions.pop(0)
	else:
		print("goal not reached")
		while len(rewards) > 0:
			guess_rewards = rewards + [-1]*(n - len(rewards))
			G = multiplier.dot(guess_rewards)
			model.update(states[0], actions[0], G)
			rewards.pop(0)
			states.pop(0)
			actions.pop(0)
			
	return totalreward
	

if __name__ == '__main__':
	env = gym.make('MountainCar-v0')
	ft = FeatureTransformer(env)
	model = Model(env, ft, "constant")
	gamma = 0.99
	
	if "monitor" in sys.argv:
		filename = os.path.basename(__file__).split('.')[0]
		monitor_dir = './' + filename + '_' + str(datetime.now())
		env = wrappers.Monitor(env, monitor_dir)
		
	N = 300
	totalrewards = np.empty(N)
	costs = np.empty(N)
	for n in range(N):
		eps = 0.1*(0.97)
		totalreward = play_one(model, eps, gamma)
		totalrewards[n] = totalreward
		print("episode:", n, "total reward:", totalreward)
	print("avg reward for last 100 episodes:", totalrewards[-100:].mean())
	
	plt.plot(totalrewards)
	plt.title("Rewards")
	plt.show()
	
	plot_running_avg(totalrewards)
	
	plot_cost_to_go(env, model)
