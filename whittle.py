### Risk-Neutral Whittle Index

import numpy as np
from scipy.special import softmax


class Whittle:

    def __init__(self, num_states: int, num_arms: int, reward, transition, horizon):
        self.num_x = num_states
        self.num_a = num_arms
        self.reward = reward
        self.transition = transition
        self.horizon = horizon
        self.digits = 4
        self.w_indices = []

    def get_whittle_indices(self, computation_type, params, n_trials):
        if computation_type == 2:
            l_steps = int(params[1] / n_trials)
            self.whittle_binary_search(params[0], params[1], l_steps)
        else:
            self.whittle_brute_force(params[0], params[1], n_trials)

    def is_equal_mat(self, mat1, mat2):
        for t in range(self.horizon):
            if not np.array_equal(mat1[:, t], mat2[:, t]):
                return False
        return True

    def indexability_check(self, arm_indices, nxt_pol, ref_pol, penalty):
        for t in range(self.horizon):
            if np.any((ref_pol[:, t] == 0) & (nxt_pol[:, t] == 1)):
                print("Not indexable!")
                return False, np.zeros((self.num_x, self.horizon))
            else:
                elements = np.argwhere((ref_pol[:, t] == 1) & (nxt_pol[:, t] == 0))
                for e in elements:
                    arm_indices[e, t] = penalty
        return True, arm_indices

    def whittle_brute_force(self, lower_bound, upper_bound, num_trials):
        for arm in range(self.num_a):
            arm_indices = np.zeros((self.num_x, self.horizon))
            penalty_ref = lower_bound
            ref_pol, _, _ = self.backward(arm, penalty_ref)
            upb_pol, _, _ = self.backward(arm, upper_bound)
            for penalty in np.linspace(lower_bound, upper_bound, num_trials):
                nxt_pol, _, _ = self.backward(arm, penalty)
                if self.is_equal_mat(nxt_pol, upb_pol):
                    flag, arm_indices = self.indexability_check(arm_indices, nxt_pol, ref_pol, penalty)
                    break
                else:
                    if not self.is_equal_mat(nxt_pol, ref_pol):
                        flag, arm_indices = self.indexability_check(arm_indices, nxt_pol, ref_pol, penalty)
                        if flag:
                            ref_pol = np.copy(nxt_pol)
                        else:
                            break
            self.w_indices.append(arm_indices)

    def whittle_binary_search(self, lower_bound, upper_bound, l_steps):
        for arm in range(self.num_a):
            arm_indices = np.zeros((self.num_x, self.horizon))
            penalty_ref = lower_bound
            ref_pol, _, _ = self.backward(arm, penalty_ref)
            ubp_pol, _, _ = self.backward(arm, upper_bound)
            while not self.is_equal_mat(ref_pol, ubp_pol):
                LB_temp = penalty_ref
                UB_temp = upper_bound
                penalty = 0.5 * (LB_temp + UB_temp)
                diff = np.abs(UB_temp - LB_temp)
                while l_steps < diff:
                    som_pol, _, _ = self.backward(arm, penalty)
                    if self.is_equal_mat(som_pol, ref_pol):
                        LB_temp = penalty
                    else:
                        UB_temp = penalty
                    penalty = 0.5 * (LB_temp + UB_temp)
                    diff = np.abs(UB_temp - LB_temp)
                penalty_ref = penalty + l_steps
                nxt_pol, _, _ = self.backward(arm, penalty_ref)
                flag, arm_indices = self.indexability_check(arm_indices, nxt_pol, ref_pol, penalty)
                if flag:
                    ref_pol = np.copy(nxt_pol)
                else:
                    break
            self.w_indices.append(arm_indices)

    def backward(self, arm, penalty):

        # Value function initialization
        V = np.zeros((self.num_x, self.horizon + 1), dtype=np.float32)

        # State-action value function
        Q = np.zeros((self.num_x, self.horizon, 2), dtype=np.float32)

        # Policy function
        pi = np.zeros((self.num_x, self.horizon), dtype=np.int32)

        # Backward induction timing
        t = self.horizon - 1

        # The value iteration loop
        while t >= 0:

            # Loop over the first dimension of the state space
            for x in range(self.num_x):

                # Get the state-action value functions
                for act in range(2):
                    if len(self.reward.shape) == 3:
                        Q[x, t, act] = self.reward[x, act, arm] - penalty * act + np.dot(V[:, t + 1], self.transition[x, :, act, arm])
                    else:
                        Q[x, t, act] = self.reward[x, arm] - penalty * act + np.dot(V[:, t + 1], self.transition[x, :, act, arm])

                # Get the value function and the policy
                if Q[x, t, 1] < Q[x, t, 0]:
                    V[x, t] = Q[x, t, 0]
                    pi[x, t] = 0
                else:
                    V[x, t] = Q[x, t, 1]
                    pi[x, t] = 1

            t = t - 1

        return pi, V, Q

    @staticmethod
    def Whittle_policy(whittle_indices, n_selection, current_x, current_t):
        num_a = len(whittle_indices)

        current_indices = np.zeros(num_a)
        for arm in range(num_a):
            current_indices[arm] = whittle_indices[arm][current_x[arm], current_t]

        # Sort indices based on values and shuffle indices with same values
        sorted_indices = np.argsort(current_indices)[::-1]
        unique_indices, counts = np.unique(current_indices[sorted_indices], return_counts=True)
        top_indices = []
        top_len = 0
        for idx in range(len(unique_indices)):
            indices = np.where(current_indices == unique_indices[len(unique_indices) - idx - 1])[0]
            shuffled_indices = np.random.permutation(indices)
            if top_len + len(shuffled_indices) < n_selection:
                top_indices.extend(list(shuffled_indices))
                top_len += len(shuffled_indices)
            elif top_len + len(shuffled_indices) == n_selection:
                top_indices.extend(list(shuffled_indices))
                top_len += len(shuffled_indices)
                break
            else:
                top_indices.extend(list(shuffled_indices[:n_selection - top_len]))
                top_len += len(shuffled_indices[:n_selection - top_len])
                break

        # Create action vector
        action_vector = np.zeros_like(current_indices, dtype=np.int32)
        action_vector[top_indices] = 1

        return action_vector

    @staticmethod
    def Whittle_softpolicy(whittle_indices, n_selection, current_x, current_t):
        num_a = len(whittle_indices)
        action = np.zeros(num_a, dtype=np.int32)

        current_indices = np.zeros(num_a)
        for arm in range(num_a):
            current_indices[arm] = whittle_indices[arm][current_x[arm], current_t]

        softmax_probs = softmax(current_indices)
        sampled_indices = np.random.choice(range(num_a), n_selection, replace=False, p=softmax_probs)
        action[sampled_indices] = 1

        return action


class WhittleAvg:
    def __init__(self, num_states: int, num_arms: int, reward, transition):
        self.num_x = num_states
        self.num_a = num_arms
        self.reward = reward
        self.transition = transition
        self.digits = 4
        self.w_indices = []

    def get_whittle_indices(self, computation_type, params, n_trials):
        if computation_type == 2:
            l_steps = int(params[1] / n_trials)
            self.whittle_binary_search(params[0], params[1], l_steps)
        else:
            self.whittle_brute_force(params[0], params[1], n_trials)

    def indexability_check(self, arm_indices, nxt_pol, ref_pol, penalty):
        if np.any((ref_pol == 0) & (nxt_pol == 1)):
            print("Not indexable!")
            return False, np.zeros(self.num_x)
        else:
            elements = np.argwhere((ref_pol == 1) & (nxt_pol == 0))
            for e in elements:
                arm_indices[e] = penalty
        return True, arm_indices

    def whittle_brute_force(self, lower_bound, upper_bound, num_trials):
        for arm in range(self.num_a):
            arm_indices = np.zeros(self.num_x)
            penalty_ref = lower_bound
            ref_pol, _, ref_Q = self.bellman_equation(arm, penalty_ref)
            upb_pol, _, _ = self.bellman_equation(arm, upper_bound)
            for penalty in np.linspace(lower_bound, upper_bound, num_trials):
                nxt_pol, _, nxt_Q = self.bellman_equation(arm, penalty)
                if np.array_equal(nxt_pol, upb_pol):
                    flag, arm_indices = self.indexability_check(arm_indices, nxt_pol, ref_pol, penalty)
                    break
                else:
                    if not np.array_equal(nxt_pol, ref_pol):
                        flag, arm_indices = self.indexability_check(arm_indices, nxt_pol, ref_pol, penalty)
                        if flag:
                            ref_pol = np.copy(nxt_pol)
                        else:
                            break
            self.w_indices.append(arm_indices)

    def whittle_binary_search(self, lower_bound, upper_bound, l_steps):
        for arm in range(self.num_a):
            arm_indices = np.zeros(self.num_x)
            penalty_ref = lower_bound
            ref_pol, _, ref_Q = self.bellman_equation(arm, penalty_ref)
            ubp_pol, _, _ = self.bellman_equation(arm, upper_bound)
            while not np.array_equal(ref_pol, ubp_pol):
                LB_temp = penalty_ref
                UB_temp = upper_bound
                penalty = 0.5 * (LB_temp + UB_temp)
                diff = np.abs(UB_temp - LB_temp)
                while l_steps < diff:
                    som_pol, _, _ = self.bellman_equation(arm, penalty)
                    if np.array_equal(som_pol, ref_pol):
                        LB_temp = penalty
                    else:
                        UB_temp = penalty
                    penalty = 0.5 * (LB_temp + UB_temp)
                    diff = np.abs(UB_temp - LB_temp)
                penalty_ref = penalty + l_steps
                nxt_pol, _, nxt_Q = self.bellman_equation(arm, penalty_ref)
                flag, arm_indices = self.indexability_check(arm_indices, nxt_pol, ref_pol, penalty)
                if flag:
                    ref_pol = np.copy(nxt_pol)
                else:
                    break
            self.w_indices.append(arm_indices)

    def bellman_equation(self, arm, penalty):
        # Initialize value function
        V = np.zeros(self.num_x, dtype=np.float32)
        Q = np.zeros((self.num_x, 2), dtype=np.float32)
        pi = np.zeros(self.num_x, dtype=np.int32)

        # Value iteration
        diff = np.inf
        iteration = 0
        while diff > 1e-6 and iteration < 1000:
            V_prev = np.copy(V)
            for x in range(self.num_x):
                for act in range(2):
                    if len(self.reward.shape) == 3:
                        Q[x, act] = self.reward[x, act, arm] - penalty * act + np.dot(V, self.transition[x, :, act, arm])
                    else:
                        Q[x, act] = self.reward[x, arm] - penalty * act + np.dot(V, self.transition[x, :, act, arm])
                V[x] = np.max(Q[x, :])
                pi[x] = np.argmax(Q[x, :])
            diff = np.max(np.abs(V - V_prev))
            iteration += 1

        return pi, V, Q

    @staticmethod
    def Whittle_policy(whittle_indices, n_selection, current_x):
        num_a = len(whittle_indices)

        current_indices = np.zeros(num_a)
        for arm in range(num_a):
            current_indices[arm] = whittle_indices[arm][current_x[arm]]

        # Sort indices based on values and shuffle indices with same values
        sorted_indices = np.argsort(current_indices)[::-1]
        unique_indices, counts = np.unique(current_indices[sorted_indices], return_counts=True)
        top_indices = []
        top_len = 0
        for idx in range(len(unique_indices)):
            indices = np.where(current_indices == unique_indices[len(unique_indices) - idx - 1])[0]
            shuffled_indices = np.random.permutation(indices)
            if top_len + len(shuffled_indices) < n_selection:
                top_indices.extend(list(shuffled_indices))
                top_len += len(shuffled_indices)
            elif top_len + len(shuffled_indices) == n_selection:
                top_indices.extend(list(shuffled_indices))
                top_len += len(shuffled_indices)
                break
            else:
                top_indices.extend(list(shuffled_indices[:n_selection - top_len]))
                top_len += len(shuffled_indices[:n_selection - top_len])
                break

        # Create action vector
        action_vector = np.zeros_like(current_indices, dtype=np.int32)
        action_vector[top_indices] = 1

        return action_vector

    @staticmethod
    def Whittle_softpolicy(whittle_indices, n_selection, current_x):
        num_a = len(whittle_indices)
        action = np.zeros(num_a, dtype=np.int32)

        current_indices = np.zeros(num_a)
        for arm in range(num_a):
            current_indices[arm] = whittle_indices[arm][current_x[arm]]

        softmax_probs = softmax(current_indices)
        sampled_indices = np.random.choice(range(num_a), n_selection, replace=False, p=softmax_probs)
        action[sampled_indices] = 1

        return action
