### Problem Parameters
import numpy as np


# Define the cost function for each arm
class Values:

    def __init__(self, num_steps: int, num_arms: int, num_states: int, function_type, increasing: bool, num_actions=1):
        self.num_a = num_arms
        self.num_s = num_states
        self.num_act = num_actions
        if num_actions == 1:
            self.vals = np.ones((self.num_s, self.num_a))
            for a in range(self.num_a):
                if function_type[a] > 0:
                    self.vals[:, a] = (np.linspace(0, self.num_s-1, num=self.num_s)) ** function_type[a] / (self.num_s-1) ** function_type[a]
                    if not increasing:
                        self.vals[:, a] = 1 - self.vals[:, a]
        else:
            self.vals = np.ones((self.num_s, self.num_act, self.num_a))
            for a in range(self.num_a):
                for act in range(self.num_act):
                    if function_type[a] > 0:
                        self.vals[:, act, a] = (np.linspace(0, self.num_s-1, num=self.num_s)) ** function_type[a] / (self.num_s-1) ** function_type[a]
                        if not increasing:
                            self.vals[:, act, a] = 1 - self.vals[:, act, a]
        self.vals = np.round(self.vals / num_steps, 3)




# Define the Markov dynamics for each arm
class MarkovDynamics:

    def __init__(self, num_arms: int, num_states: int, prob_remain, transition_type: int, increasing: bool):
        self.num_s = num_states
        self.num_a = num_arms
        self.increasing = increasing
        self.transitions = self.purereset_and_deteriorate(prob_remain, transition_type)

    def purereset_and_deteriorate(self, prob_remain, transition_type):
        transitions = np.zeros((self.num_s, self.num_s, 2, self.num_a))
        for a in range(self.num_a):
            if transition_type == 0:
                transitions[0, 0, 1, a] = 1
                transitions[0, 0, 0, a] = 1
                for s in range(1, self.num_s):
                    transitions[s, s, 1, a] = 1
                    transitions[s, 0, 0, a] = 1 - prob_remain[a]
                    transitions[s, s, 0, a] = prob_remain[a]
            elif transition_type == 1:
                transitions[0, 0, 1, a] = 1
                transitions[0, 0, 0, a] = 1
                for s in range(1, self.num_s):
                    transitions[s, 0, 1, a] = 1 - 2 * prob_remain[a]
                    transitions[s, s, 1, a] = 2 * prob_remain[a]
                    transitions[s, 0, 0, a] = 1 - prob_remain[a]
                    transitions[s, s, 0, a] = prob_remain[a]
            elif transition_type == 2:
                transitions[0, 0, 1, a] = 1
                transitions[0, 0, 0, a] = 1
                for s in range(1, self.num_s):
                    transitions[s, s-1, 1, a] = prob_remain[a]
                    transitions[s, s, 1, a] = 1 - prob_remain[a]
                    transitions[s, s-1, 0, a] = 1 - prob_remain[a]
                    transitions[s, s, 0, a] = prob_remain[a]
            elif transition_type == 3:
                for s in range(self.num_s-1):
                    transitions[s, s, 1, a] = (self.num_s - s - 1) * prob_remain[a]
                    transitions[s, self.num_s-1, 1, a] = 1 - (self.num_s - s - 1) * prob_remain[a]
                transitions[self.num_s-1, self.num_s-1, 1, a] = 1
                transitions[0, 0, 0, a] = 1
                transitions[1:, 0, 0, a] = (1 - (self.num_s - 1) * prob_remain[a]) * np.ones(self.num_s-1)
                transitions[1:, 1:, 0, a] = np.tril(np.full((self.num_s - 1, self.num_s - 1), prob_remain[a]))
                for s in range(1, self.num_s):
                    transitions[s, s, 0, a] = (self.num_s - s) * transitions[s, s, 0, a]
            elif transition_type == 4:
                transitions[:self.num_s-1, :self.num_s-1, 1, a] = prob_remain[a] * np.triu(np.ones((self.num_s-1, self.num_s-1)))
                transitions[:, self.num_s-1, 1, a] = 1 - (self.num_s - np.arange(self.num_s) - 1) * prob_remain[a]
                transitions[0, 0, 0, a] = 1
                transitions[1:, 0, 0, a] = (1 - (self.num_s - 1) * prob_remain[a]) * np.ones(self.num_s-1)
                transitions[1:, 1:, 0, a] = np.tril(np.full((self.num_s - 1, self.num_s - 1), prob_remain[a]))
                for s in range(1, self.num_s):
                    transitions[s, s, 0, a] = (self.num_s - s) * transitions[s, s, 0, a]
            elif transition_type == 5:
                transitions[0, 0, 0, a] = 1
                for s in range(1, self.num_s):
                    transitions[s, 1:s+1, 0, a] = np.round(prob_remain[a] * np.ones(s), 2)
                    transitions[s, 0, 0, a] = 1 - sum(transitions[s, 1:, 0, a])
                for s in range(self.num_s):
                    transitions[s, -1, 1, a] = np.round((s+1) * prob_remain[a], 2)
                    transitions[s, 0, 1, a] = 1 - transitions[s, -1, 1, a]
            elif transition_type == 6:
                transitions[:, :, 0, a] = np.round(((1 - prob_remain[a]) / (self.num_s-1)) * np.ones((self.num_s, self.num_s)), 2)
                for s in range(self.num_s):
                    transitions[s, s, 0, a] = 1 - sum(transitions[0, 1:, 0, a])
                transitions[:, :, 1, a] = np.round(((1 - prob_remain[a]) / (self.num_s-1)) * np.triu(np.ones((self.num_s, self.num_s))), 2)
                for s in range(self.num_s):
                    transitions[s, -1, 1, a] = 1 - sum(transitions[s, :self.num_s-1, 1, a])
            elif transition_type == 14:
                pr01 = prob_remain[0]
                pr02 = prob_remain[1]
                pr11 = prob_remain[2]
                pr12 = prob_remain[3]
                transitions[:, :, 0, a] = np.array([
                    [1, 0, 0],
                    [pr01[a], 1-pr01[a], 0],
                    [0, 1-pr02[a], pr02[a]]
                    ])
                transitions[:, :, 1, a] = np.array([
                    [1, 0, 0],
                    [pr11[a], 1-pr11[a], 0],
                    [0, 1-pr12[a], pr12[a]]
                    ])

        return transitions
