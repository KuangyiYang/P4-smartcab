import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator


class QLearningAgent(Agent):
	# An agent using Q learning learns to drive

	def __init__(self, env):
		super(QLearningAgent, self).__init__(env)
		self.color = 'red'
		self.planner = RoutePlanner(self.env, self)
		self.deadline = self.env.get_deadline(self)
		self.next_waypoint = None
		self.moves = 0

		self.qDict = dict()
		self.alpha = 0.9 # learning rate
		self.epsilon = 0.05 # probability of flipping the coin
		self.gamma = 0.2
		
		self.state = None
		self.new_state = None

		self.reward = None
		self.cum_reward = 0

		self.possible_actions = Environment.valid_actions
		self.action = None

	def reset(self, destination = None):
		self.planner.route_to(destination)
		self.next_waypoint = None
		self.moves = 0

		self.state = None
		self.new_state = None

		self.reward = 0
		self.cum_reward = 0

		self.action = None

	def getQvalue(self, state, action):
		key = (state, action)
		return self.qDict.get(key, 5.0)

	def getMaxQ(self, state):
		q = [self.getQvalue(state, a) for a in self.possible_actions]
		return max(q)

	def get_action(self, state):
		"""
		epsilon-greedy approach to choose action given the state 
		"""
		if random.random() < self.epsilon:
			action = random.choice(self.possible_actions)
		else:
			q = [self.getQvalue(state, a) for a in self.possible_actions]
			if q.count(max(q)) > 1: 
				best_actions = [i for i in range(len(self.possible_actions)) if q[i] == max(q)]                       
				index = random.choice(best_actions)

			else:
				index = q.index(max(q))
			action = self.possible_actions[index]

		return action

	def qlearning(self, state, action, nextState, reward):
		"""
		use Qlearning algorithm to update q values
		"""
		key = (state, action)
		if (key not in self.qDict):
			# initialize the q values
			self.qDict[key] = 5.0
		else:
			self.qDict[key] = self.qDict[key] + self.alpha * (reward + self.gamma*self.getMaxQ(nextState) - self.qDict[key])

	def update(self, t):
		self.next_waypoint = self.planner.next_waypoint()
		inputs = self.env.sense(self)
		deadline = self.env.get_deadline(self)

		# take the following variables as states:
		# 	- environ(oncoming, left, right)
		#	- next_waypoint
		#	- traffic light
		self.new_state = inputs
		self.new_state['next_waypoint'] = self.next_waypoint
		self.new_state = tuple(sorted(self.new_state.items()))

		# for the current state, choose an action based on epsilon policy
		action = self.get_action(self.new_state)
		# observe the reward
		new_reward = self.env.act(self, action)
		# update q value based on q learning algorithm
		if self.reward != None:
			self.qlearning(self.state, self.action, self.new_state, self.reward)
		# set the state to the new state
		self.action = action
		self.state = self.new_state
		self.reward = new_reward
		self.cum_reward = self.cum_reward + new_reward
		self.moves = self.moves + 1
		#print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, new_reward)  # [debug]

		
		
def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(QLearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=0.001)  # reduce update_delay to speed up simulation
    sim.run(n_trials=100)  # press Esc or close pygame window to quit


if __name__ == '__main__':
    run()
