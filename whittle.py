### Risk-Neutral & Risk-Aware Whittle Index
import numpy as np
from scipy.special import softmax
from itertools import product


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
                lb_temp = penalty_ref
                ub_temp = upper_bound
                penalty = 0.5 * (lb_temp + ub_temp)
                diff = np.abs(ub_temp - lb_temp)
                while l_steps < diff:
                    som_pol, _, _ = self.backward(arm, penalty)
                    if self.is_equal_mat(som_pol, ref_pol):
                        lb_temp = penalty
                    else:
                        ub_temp = penalty
                    penalty = 0.5 * (lb_temp + ub_temp)
                    diff = np.abs(ub_temp - lb_temp)
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


class SafeWhittle:
    
    def __init__(self, num_states: int, num_arms: int, rewards, transition, horizon, u_type, u_order, thresholds):
        self.num_x = num_states
        self.num_a = num_arms
        self.rewards = rewards
        self.transition = transition
        self.horizon = horizon
        self.u_type = u_type
        self.thresholds = thresholds

        self.digits = 3
        self.n_realize = []
        self.n_augment = [0] * num_arms
        self.all_rews = []
        self.all_valus = []

        for a in range(num_arms):

            if len(self.rewards.shape) == 3:
                all_immediate_rew = self.rewards[:, :, a]
            else:
                all_immediate_rew = self.rewards[:, a]
            arm_n_realize = []
            all_total_rewards = []
            for t in range(self.horizon):
                all_total_rewards_by_t = possible_reward_sums(all_immediate_rew.flatten(), t + 1)
                arm_n_realize.append(len(all_total_rewards_by_t))
                if t == self.horizon - 1:
                    all_total_rewards = all_total_rewards_by_t

            self.n_augment[a] = len(all_total_rewards)
            self.all_rews.append(all_total_rewards)
            self.n_realize.append(arm_n_realize)

            arm_valus = []
            for total_rewards in all_total_rewards:
                if u_type == 1:
                    arm_valus.append(np.round(1 - thresholds[a] ** (- 1 / u_order) * (np.maximum(0, thresholds[a] - total_rewards)) ** (1 / u_order), 3))
                elif u_type == 2:
                    arm_valus.append(np.round((1 + np.exp(-u_order * (1 - thresholds[a]))) / (1 + np.exp(-u_order * (total_rewards - thresholds[a]))), 3))
                else:
                    arm_valus.append(1 if total_rewards >= thresholds[a] else 0)

            self.all_valus.append(arm_valus)
            # print(arm_valus)

        # print(self.all_rews[0])
        # print(self.all_valus[0])
        self.w_indices = []

    def get_whittle_indices(self, computation_type, params, n_trials):
        if computation_type == 1:
            l_steps = params[1] / n_trials
            self.whittle_binary_search(params[0], params[1], l_steps)
        else:
            self.whittle_brute_force(params[0], params[1], n_trials)

    def is_equal_mat(self, mat1, mat2, realize_index):
        for t in range(self.horizon):
            mat1_new = mat1[:realize_index[t], :]
            mat2_new = mat2[:realize_index[t], :]
            if not np.array_equal(mat1_new, mat2_new):
                return False
        return True

    def indexability_check(self, arm, arm_indices, realize_index, nxt_pol, ref_pol, penalty, nxt_Q, ref_Q):
        for t in range(self.horizon):
            ref_pol_new = ref_pol[:realize_index[t], :, t]
            nxt_pol_new = nxt_pol[:realize_index[t], :, t]
            ref_Q_new = ref_Q[:realize_index[t], :, t, :]
            nxt_Q_new = nxt_Q[:realize_index[t], :, t, :]
            if np.any((ref_pol_new == 0) & (nxt_pol_new == 1)):
                print("Not indexable!")
                elements = np.argwhere((ref_pol_new == 0) & (nxt_pol_new == 1))
                for e in elements:
                    print(f'element: {[e[0], e[1], t]}')
                    print(f'penalty: {penalty}')
                    print(f'ref policy: {ref_pol_new[:, e[1]]}')
                    print(f'nxt policy: {nxt_pol_new[:, e[1]]}')
                    print(f'ref Q0: {ref_Q_new[e[0], e[1], 0]}')
                    print(f'ref Q1: {ref_Q_new[e[0], e[1], 1]}')
                    print(f'nxt Q0: {nxt_Q_new[e[0], e[1], 0]}')
                    print(f'nxt Q1: {nxt_Q_new[e[0], e[1], 1]}')
                return False, np.zeros((self.n_augment[arm], self.num_x, self.horizon))
            else:
                elements = np.argwhere((ref_pol_new == 1) & (nxt_pol_new == 0))
                for e in elements:
                    arm_indices[e[0], e[1], t] = penalty
        return True, arm_indices

    def whittle_brute_force(self, lower_bound, upper_bound, num_trials):

        for arm in range(self.num_a):
            arm_indices = np.zeros((self.n_augment[arm], self.num_x, self.horizon))
            penalty_ref = lower_bound
            ref_pol, _, ref_Q = self.backward_discreteliftedstate(arm, penalty_ref)
            upb_pol, _, _ = self.backward_discreteliftedstate(arm, upper_bound)
            for penalty in np.linspace(lower_bound, upper_bound, num_trials):
                penalty = np.round(penalty, self.digits)
                nxt_pol, _, nxt_Q = self.backward_discreteliftedstate(arm, penalty)
                if self.is_equal_mat(nxt_pol, upb_pol, self.n_realize[arm]):
                    flag, arm_indices = self.indexability_check(arm, arm_indices, self.n_realize[arm], nxt_pol, ref_pol, penalty, nxt_Q, ref_Q)
                    break
                else:
                    if not self.is_equal_mat(nxt_pol, ref_pol, self.n_realize[arm]):
                        flag, arm_indices = self.indexability_check(arm, arm_indices, self.n_realize[arm], nxt_pol, ref_pol, penalty, nxt_Q, ref_Q)
                        if flag:
                            ref_pol = np.copy(nxt_pol)
                            ref_Q = np.copy(nxt_Q)
                        else:
                            break
            self.w_indices.append(arm_indices)

    def whittle_binary_search(self, lower_bound, upper_bound, l_steps):

        for arm in range(self.num_a):
            arm_indices = np.zeros((self.n_augment[arm], self.num_x, self.horizon))
            penalty_ref = lower_bound
            ref_pol, _, ref_Q = self.backward_discreteliftedstate(arm, penalty_ref)
            ubp_pol, _, _ = self.backward_discreteliftedstate(arm, upper_bound)
            while not self.is_equal_mat(ref_pol, ubp_pol, self.n_realize[arm]):
                lb_temp = penalty_ref
                ub_temp = upper_bound
                penalty = np.round(0.5 * (lb_temp + ub_temp), self.digits)
                diff = np.abs(ub_temp - lb_temp)
                while l_steps < diff:
                    som_pol, _, _ = self.backward_discreteliftedstate(arm, penalty)
                    if self.is_equal_mat(som_pol, ref_pol, self.n_realize[arm]):
                        lb_temp = penalty
                    else:
                        ub_temp = penalty
                    penalty = np.round(0.5 * (lb_temp + ub_temp), self.digits)
                    diff = np.abs(ub_temp - lb_temp)
                penalty_ref = penalty + l_steps
                nxt_pol, _, nxt_Q = self.backward_discreteliftedstate(arm, penalty_ref)
                indexability_flag, arm_indices = self.indexability_check(arm, arm_indices, self.n_realize[arm], nxt_pol, ref_pol, penalty, nxt_Q, ref_Q)
                if indexability_flag:
                    ref_pol = np.copy(nxt_pol)
                    ref_Q = np.copy(nxt_Q)
                else:
                    break
            self.w_indices.append(arm_indices)

    def backward_discreteliftedstate(self, arm, penalty):

        # Value function initialization
        V = np.zeros((self.n_augment[arm], self.num_x, self.horizon + 1), dtype=np.float32)
        for l in range(self.n_augment[arm]):
            V[l, :, self.horizon] = self.all_valus[arm][l] * np.ones(self.num_x)
        # print(V[:, 0, self.horizon])

        # State-action value function
        Q = np.zeros((self.n_augment[arm], self.num_x, self.horizon, 2), dtype=np.float32)

        # Policy function
        pi = np.zeros((self.n_augment[arm], self.num_x, self.horizon), dtype=np.int32)

        # Backward induction timing
        t = self.horizon - 1

        # The value iteration loop
        while t >= 0:

            # Loop over the first dimension of the state space
            for x in range(self.num_x):

                # Loop over the second dimension of the state space
                for l in range(self.n_realize[arm][t]):

                    # Get the state-action value functions
                    # print('**************************************************')
                    # print(f'penalty: {penalty}')
                    # print(f'arm: {arm}')
                    # print(f't: {t}')
                    # print(f'n_augment[arm]: {self.n_augment[arm]}')
                    # print(f'n_realize[arm][t] = {self.n_realize[arm][t]}')
                    # print(f'x: {x}')
                    # print(f'cur_state: {x}')
                    # print(f'reward[x={x}, a={arm}]: {self.rewards[x, arm]}')
                    # print(f'l: {l}')
                    # print(f'nxt_l: {nxt_l}')
                    # print(f'V[nxt_l, :, t+1={t+1}]: {V[nxt_l, :, t+1]}')
                    for act in range(2):

                        # Convert the next state of the second dimension into an index ranged from 1 to L
                        # if len(self.rewards.shape) == 3:
                        #     reward_added = self.rewards[x, act, arm]
                        # else:
                        #     reward_added = self.rewards[x, arm]
                        # total_reward = self.all_rews[arm][l]
                        # next_total_r = total_reward + reward_added
                        # nxt_l = next((i for i, x in enumerate(self.all_rews[arm]) if x == next_total_r), -1)
                        nxt_l = max(0, min(self.n_augment[arm] - 1, l + x))

                        Q[l, x, t, act] = np.round(- penalty * act / self.horizon + np.dot(V[nxt_l, :, t + 1], self.transition[x, :, act, arm]), self.digits + 1)
                        # print('--------------')
                        # print(f'act: {act}')
                        # print(f'- penalty * {act} / horizon: {- penalty * act / self.horizon}')
                        # print(f'transition[x={x}, :, act={act}, arm={arm}]: {self.transition[x, :, act, arm]}')
                        # print(f'np.dot(V[nxt_l={nxt_l}, :, t+1={t+1}], transition): {np.dot(V[nxt_l, :, t+1], self.transition[x, :, act, arm])}')
                        # print(f'Q[l={l}, x={x}, t={t}, act={act}]: {Q[l, x, t, act]}')

                    # Get the value function and the policy
                    if 0 < Q[l, x, t, 0] - Q[l, x, t, 1]:
                        V[l, x, t] = Q[l, x, t, 0]
                        pi[l, x, t] = 0
                    else:
                        V[l, x, t] = Q[l, x, t, 1]
                        pi[l, x, t] = 1

                    # if l==12 and x==1 and t==0:
                    #   print('Exact@')
                    #   print(f'index: {[l, x, t]}')
                    #   print(f'Q0: {Q[l, x, t, 0]}')
                    #   print(f'Q1: {Q[l, x, t, 1]}')
                    #   print(f'po: {pi[l, x, t]}')
                    #   print(f'v_t+1: {V[nxt_l, :, t+1]}')
                    #   print(f'tr0: {self.transition[x, :, 0, arm]}')
                    #   print(f'dot0: {np.dot(V[nxt_l, :, t+1], self.transition[x, :, 0, arm])}')
                    #   print(f'tr1: {self.transition[x, :, 1, arm]}')
                    #   print(f'dot1: {np.dot(V[nxt_l, :, t+1], self.transition[x, :, 1, arm])}')
                    #   print(f'pen: {-penalty}')
                    #   print(f'pen+dot1: {- penalty + np.dot(V[nxt_l, :, t+1], self.transition[x, :, 1, arm])}')

            t = t - 1

        # print(Q[20, :, 4, 0])
        # print(Q[20, :, 4, 1])
        return pi, V, Q

    @staticmethod
    def Whittle_policy(whittle_indices, n_selection, current_x, current_l, current_t):
        num_a = len(whittle_indices)

        current_indices = np.zeros(num_a)
        count_positive = 0
        for arm in range(num_a):
            w_idx = whittle_indices[arm][current_l[arm], current_x[arm], current_t]
            current_indices[arm] = w_idx
            if w_idx >= 0:
                count_positive += 1
        n_selection = np.minimum(n_selection, count_positive)

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
    def Whittle_softpolicy(whittle_indices, n_selection, current_x, current_l, current_t):
        num_a = len(whittle_indices)
        action = np.zeros(num_a, dtype=np.int32)

        current_indices = np.zeros(num_a)
        count_positive = 0
        for arm in range(num_a):
            w_idx = whittle_indices[arm][current_l[arm], current_x[arm], current_t]
            current_indices[arm] = w_idx
            if w_idx >= 0:
                count_positive += 1
        n_selection = np.minimum(n_selection, count_positive)

        softmax_probs = softmax(current_indices)
        sampled_indices = np.random.choice(range(num_a), n_selection, replace=False, p=softmax_probs)
        action[sampled_indices] = 1

        return action


def Whittle_regularpolicy(whittle_indices, n_selection, current_x, current_t):
    num_a = len(whittle_indices)

    current_indices = np.zeros(num_a)
    count_positive = 0
    for arm in range(num_a):
        w_idx = whittle_indices[arm][current_x[arm], current_t]
        current_indices[arm] = w_idx
        if w_idx >= 0:
            count_positive += 1
    n_selection = np.minimum(n_selection, count_positive)

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


def possible_reward_sums(rewards, num_steps):
    reward_combinations = product(rewards, repeat=num_steps)
    sums = set()
    for combination in reward_combinations:
        sums.add(np.round(sum(combination), 3))
    return sorted(sums)


class WhittleNS:

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
                lb_temp = penalty_ref
                ub_temp = upper_bound
                penalty = 0.5 * (lb_temp + ub_temp)
                diff = np.abs(ub_temp - lb_temp)
                while l_steps < diff:
                    som_pol, _, _ = self.backward(arm, penalty)
                    if self.is_equal_mat(som_pol, ref_pol):
                        lb_temp = penalty
                    else:
                        ub_temp = penalty
                    penalty = 0.5 * (lb_temp + ub_temp)
                    diff = np.abs(ub_temp - lb_temp)
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
                    if len(self.reward.shape) == 4:
                        Q[x, t, act] = self.reward[x, act, arm, t] - penalty * act + np.dot(V[:, t + 1], self.transition[x, :, act, arm, t])
                    else:
                        Q[x, t, act] = self.reward[x, arm, t] - penalty * act + np.dot(V[:, t + 1], self.transition[x, :, act, arm, t])

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


class SafeWhittleNS:

    def __init__(self, num_states: int, num_arms: int, rewards, transition, horizon, u_type, u_order, thresholds):
        self.num_x = num_states
        self.num_a = num_arms
        self.rewards = rewards
        self.transition = transition
        self.horizon = horizon
        self.u_type = u_type
        self.thresholds = thresholds

        self.digits = 3
        self.n_realize = []
        self.n_augment = [0] * num_arms
        self.all_rews = []
        self.all_valus = []

        for a in range(num_arms):
            arm_n_realize = []
            all_total_rewards = []
            for t in range(self.horizon):
                if len(self.rewards.shape) == 4:
                    immediate_rews = self.rewards[:, :, a, :t+1] 
                    all_immediate_rews = immediate_rews.reshape(-1, immediate_rews.shape[-1]) 
                else:
                    all_immediate_rews = self.rewards[:, a, :t+1]
                all_total_rewards_by_t = possible_ns_reward_sums(all_immediate_rews)
                arm_n_realize.append(len(all_total_rewards_by_t))
                if t == self.horizon - 1:
                    all_total_rewards = all_total_rewards_by_t

            self.n_augment[a] = len(all_total_rewards)
            self.all_rews.append(all_total_rewards)
            self.n_realize.append(arm_n_realize)

            arm_valus = []
            for total_rewards in all_total_rewards:
                if u_type == 1:
                    arm_valus.append(np.round(1 - thresholds[a] ** (- 1 / u_order) * (np.maximum(0, thresholds[a] - total_rewards)) ** (1 / u_order), 3))
                elif u_type == 2:
                    arm_valus.append(np.round((1 + np.exp(-u_order * (1 - thresholds[a]))) / (1 + np.exp(-u_order * (total_rewards - thresholds[a]))), 3))
                else:
                    arm_valus.append(1 if total_rewards >= thresholds[a] else 0)

            self.all_valus.append(arm_valus)
        self.w_indices = []

    def get_whittle_indices(self, computation_type, params, n_trials):
        if computation_type == 1:
            l_steps = params[1] / n_trials
            self.whittle_binary_search(params[0], params[1], l_steps)
        else:
            self.whittle_brute_force(params[0], params[1], n_trials)

    def is_equal_mat(self, mat1, mat2, realize_index):
        for t in range(self.horizon):
            mat1_new = mat1[:realize_index[t], :]
            mat2_new = mat2[:realize_index[t], :]
            if not np.array_equal(mat1_new, mat2_new):
                return False
        return True

    def indexability_check(self, arm, arm_indices, realize_index, nxt_pol, ref_pol, penalty, nxt_Q, ref_Q):
        for t in range(self.horizon):
            ref_pol_new = ref_pol[:realize_index[t], :, t]
            nxt_pol_new = nxt_pol[:realize_index[t], :, t]
            ref_Q_new = ref_Q[:realize_index[t], :, t, :]
            nxt_Q_new = nxt_Q[:realize_index[t], :, t, :]
            if np.any((ref_pol_new == 0) & (nxt_pol_new == 1)):
                print("Not indexable!")
                elements = np.argwhere((ref_pol_new == 0) & (nxt_pol_new == 1))
                for e in elements:
                    print(f'element: {[e[0], e[1], t]}')
                    print(f'penalty: {penalty}')
                    print(f'ref policy: {ref_pol_new[:, e[1]]}')
                    print(f'nxt policy: {nxt_pol_new[:, e[1]]}')
                    print(f'ref Q0: {ref_Q_new[e[0], e[1], 0]}')
                    print(f'ref Q1: {ref_Q_new[e[0], e[1], 1]}')
                    print(f'nxt Q0: {nxt_Q_new[e[0], e[1], 0]}')
                    print(f'nxt Q1: {nxt_Q_new[e[0], e[1], 1]}')
                return False, np.zeros((self.n_augment[arm], self.num_x, self.horizon))
            else:
                elements = np.argwhere((ref_pol_new == 1) & (nxt_pol_new == 0))
                for e in elements:
                    arm_indices[e[0], e[1], t] = penalty
        return True, arm_indices

    def whittle_brute_force(self, lower_bound, upper_bound, num_trials):

        for arm in range(self.num_a):
            arm_indices = np.zeros((self.n_augment[arm], self.num_x, self.horizon))
            penalty_ref = lower_bound
            ref_pol, _, ref_Q = self.backward_discreteliftedstate(arm, penalty_ref)
            upb_pol, _, _ = self.backward_discreteliftedstate(arm, upper_bound)
            for penalty in np.linspace(lower_bound, upper_bound, num_trials):
                penalty = np.round(penalty, self.digits)
                nxt_pol, _, nxt_Q = self.backward_discreteliftedstate(arm, penalty)
                if self.is_equal_mat(nxt_pol, upb_pol, self.n_realize[arm]):
                    flag, arm_indices = self.indexability_check(arm, arm_indices, self.n_realize[arm], nxt_pol, ref_pol, penalty, nxt_Q, ref_Q)
                    break
                else:
                    if not self.is_equal_mat(nxt_pol, ref_pol, self.n_realize[arm]):
                        flag, arm_indices = self.indexability_check(arm, arm_indices, self.n_realize[arm], nxt_pol, ref_pol, penalty, nxt_Q, ref_Q)
                        if flag:
                            ref_pol = np.copy(nxt_pol)
                            ref_Q = np.copy(nxt_Q)
                        else:
                            break
            self.w_indices.append(arm_indices)

    def whittle_binary_search(self, lower_bound, upper_bound, l_steps):

        for arm in range(self.num_a):
            arm_indices = np.zeros((self.n_augment[arm], self.num_x, self.horizon))
            penalty_ref = lower_bound
            ref_pol, _, ref_Q = self.backward_discreteliftedstate(arm, penalty_ref)
            ubp_pol, _, _ = self.backward_discreteliftedstate(arm, upper_bound)
            while not self.is_equal_mat(ref_pol, ubp_pol, self.n_realize[arm]):
                lb_temp = penalty_ref
                ub_temp = upper_bound
                penalty = np.round(0.5 * (lb_temp + ub_temp), self.digits)
                diff = np.abs(ub_temp - lb_temp)
                while l_steps < diff:
                    som_pol, _, _ = self.backward_discreteliftedstate(arm, penalty)
                    if self.is_equal_mat(som_pol, ref_pol, self.n_realize[arm]):
                        lb_temp = penalty
                    else:
                        ub_temp = penalty
                    penalty = np.round(0.5 * (lb_temp + ub_temp), self.digits)
                    diff = np.abs(ub_temp - lb_temp)
                penalty_ref = penalty + l_steps
                nxt_pol, _, nxt_Q = self.backward_discreteliftedstate(arm, penalty_ref)
                indexability_flag, arm_indices = self.indexability_check(arm, arm_indices, self.n_realize[arm], nxt_pol, ref_pol, penalty, nxt_Q, ref_Q)
                if indexability_flag:
                    ref_pol = np.copy(nxt_pol)
                    ref_Q = np.copy(nxt_Q)
                else:
                    break
            self.w_indices.append(arm_indices)

    def backward_discreteliftedstate(self, arm, penalty):

        # Value function initialization
        V = np.zeros((self.n_augment[arm], self.num_x, self.horizon + 1), dtype=np.float32)
        for l in range(self.n_augment[arm]):
            V[l, :, self.horizon] = self.all_valus[arm][l] * np.ones(self.num_x)

        # State-action value function
        Q = np.zeros((self.n_augment[arm], self.num_x, self.horizon, 2), dtype=np.float32)

        # Policy function
        pi = np.zeros((self.n_augment[arm], self.num_x, self.horizon), dtype=np.int32)

        # Backward induction timing
        t = self.horizon - 1

        # The value iteration loop
        while t >= 0:

            # Loop over the first dimension of the state space
            for x in range(self.num_x):

                # Loop over the second dimension of the state space
                for l in range(self.n_realize[arm][t]):

                    # Get the state-action value functions
                    for act in range(2):

                        # Convert the next state of the second dimension into an index ranged from 1 to L
                        nxt_l = max(0, min(self.n_augment[arm] - 1, l + x))
                        Q[l, x, t, act] = np.round(- penalty * act / self.horizon + np.dot(V[nxt_l, :, t + 1], self.transition[x, :, act, arm, t]), self.digits + 1)

                    # Get the value function and the policy
                    if 0 < Q[l, x, t, 0] - Q[l, x, t, 1]:
                        V[l, x, t] = Q[l, x, t, 0]
                        pi[l, x, t] = 0
                    else:
                        V[l, x, t] = Q[l, x, t, 1]
                        pi[l, x, t] = 1

            t = t - 1

        return pi, V, Q

    @staticmethod
    def Whittle_policy(whittle_indices, n_selection, current_x, current_l, current_t):
        num_a = len(whittle_indices)

        current_indices = np.zeros(num_a)
        count_positive = 0
        for arm in range(num_a):
            w_idx = whittle_indices[arm][current_l[arm], current_x[arm], current_t]
            current_indices[arm] = w_idx
            if w_idx >= 0:
                count_positive += 1
        n_selection = np.minimum(n_selection, count_positive)

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
    def Whittle_softpolicy(whittle_indices, n_selection, current_x, current_l, current_t):
        num_a = len(whittle_indices)
        action = np.zeros(num_a, dtype=np.int32)

        current_indices = np.zeros(num_a)
        count_positive = 0
        for arm in range(num_a):
            w_idx = whittle_indices[arm][current_l[arm], current_x[arm], current_t]
            current_indices[arm] = w_idx
            if w_idx >= 0:
                count_positive += 1
        n_selection = np.minimum(n_selection, count_positive)

        softmax_probs = softmax(current_indices)
        sampled_indices = np.random.choice(range(num_a), n_selection, replace=False, p=softmax_probs)
        action[sampled_indices] = 1

        return action


class SafeWhittleDNS:

    def __init__(self, num_states: int, num_arms: int, rewards, transition, horizon, u_type, u_order, thresholds):
        self.num_x = num_states[0]
        self.num_s = num_states[1]
        self.cutting_points = np.linspace(0, 1, self.num_s+1)
        self.all_total_rewards = [np.median(self.cutting_points[i:i + 2]) for i in range(len(self.cutting_points) - 1)]
        self.num_a = num_arms
        self.rewards = rewards
        self.transition = transition
        self.horizon = horizon
        self.u_type = u_type
        self.thresholds = thresholds

        self.digits = 3
        self.n_realize = []
        self.n_augment = [0] * num_arms
        self.all_rews = []
        self.all_valus = []

        for a in range(num_arms):
            self.n_augment[a] = len(self.all_total_rewards)
            self.all_rews.append(self.all_total_rewards)
            self.n_realize.append([self.num_s] * self.horizon)

            arm_valus = []
            for total_rewards in self.all_total_rewards:
                if u_type == 1:
                    arm_valus.append(np.round(1 - thresholds[a] ** (- 1 / u_order) * (np.maximum(0, thresholds[a] - total_rewards)) ** (1 / u_order), 3))
                elif u_type == 2:
                    arm_valus.append(np.round((1 + np.exp(-u_order * (1 - thresholds[a]))) / (1 + np.exp(-u_order * (total_rewards - thresholds[a]))), 3))
                else:
                    arm_valus.append(1 if total_rewards >= thresholds[a] else 0)

            self.all_valus.append(arm_valus)
        self.w_indices = []

    def get_whittle_indices(self, computation_type, params, n_trials):
        if computation_type == 1:
            l_steps = params[1] / n_trials
            self.whittle_binary_search(params[0], params[1], l_steps)
        else:
            self.whittle_brute_force(params[0], params[1], n_trials)

    def is_equal_mat(self, mat1, mat2, realize_index):
        for t in range(self.horizon):
            mat1_new = mat1[:realize_index[t], :]
            mat2_new = mat2[:realize_index[t], :]
            if not np.array_equal(mat1_new, mat2_new):
                return False
        return True

    def indexability_check(self, arm, arm_indices, realize_index, nxt_pol, ref_pol, penalty, nxt_Q, ref_Q):
        for t in range(self.horizon):
            ref_pol_new = ref_pol[:realize_index[t], :, t]
            nxt_pol_new = nxt_pol[:realize_index[t], :, t]
            ref_Q_new = ref_Q[:realize_index[t], :, t, :]
            nxt_Q_new = nxt_Q[:realize_index[t], :, t, :]
            if np.any((ref_pol_new == 0) & (nxt_pol_new == 1)):
                print("Not indexable!")
                elements = np.argwhere((ref_pol_new == 0) & (nxt_pol_new == 1))
                for e in elements:
                    print(f'element: {[e[0], e[1], t]}')
                    print(f'penalty: {penalty}')
                    print(f'ref policy: {ref_pol_new[:, e[1]]}')
                    print(f'nxt policy: {nxt_pol_new[:, e[1]]}')
                    print(f'ref Q0: {ref_Q_new[e[0], e[1], 0]}')
                    print(f'ref Q1: {ref_Q_new[e[0], e[1], 1]}')
                    print(f'nxt Q0: {nxt_Q_new[e[0], e[1], 0]}')
                    print(f'nxt Q1: {nxt_Q_new[e[0], e[1], 1]}')
                return False, np.zeros((self.n_augment[arm], self.num_x, self.horizon))
            else:
                elements = np.argwhere((ref_pol_new == 1) & (nxt_pol_new == 0))
                for e in elements:
                    arm_indices[e[0], e[1], t] = penalty
        return True, arm_indices

    def whittle_brute_force(self, lower_bound, upper_bound, num_trials):

        for arm in range(self.num_a):
            arm_indices = np.zeros((self.n_augment[arm], self.num_x, self.horizon))
            penalty_ref = lower_bound
            ref_pol, _, ref_Q = self.backward_discreteliftedstate(arm, penalty_ref)
            upb_pol, _, _ = self.backward_discreteliftedstate(arm, upper_bound)
            for penalty in np.linspace(lower_bound, upper_bound, num_trials):
                penalty = np.round(penalty, self.digits)
                nxt_pol, _, nxt_Q = self.backward_discreteliftedstate(arm, penalty)
                if self.is_equal_mat(nxt_pol, upb_pol, self.n_realize[arm]):
                    flag, arm_indices = self.indexability_check(arm, arm_indices, self.n_realize[arm], nxt_pol, ref_pol, penalty, nxt_Q, ref_Q)
                    break
                else:
                    if not self.is_equal_mat(nxt_pol, ref_pol, self.n_realize[arm]):
                        flag, arm_indices = self.indexability_check(arm, arm_indices, self.n_realize[arm], nxt_pol, ref_pol, penalty, nxt_Q, ref_Q)
                        if flag:
                            ref_pol = np.copy(nxt_pol)
                            ref_Q = np.copy(nxt_Q)
                        else:
                            break
            self.w_indices.append(arm_indices)

    def whittle_binary_search(self, lower_bound, upper_bound, l_steps):

        for arm in range(self.num_a):
            arm_indices = np.zeros((self.n_augment[arm], self.num_x, self.horizon))
            penalty_ref = lower_bound
            ref_pol, _, ref_Q = self.backward_discreteliftedstate(arm, penalty_ref)
            ubp_pol, _, _ = self.backward_discreteliftedstate(arm, upper_bound)
            while not self.is_equal_mat(ref_pol, ubp_pol, self.n_realize[arm]):
                lb_temp = penalty_ref
                ub_temp = upper_bound
                penalty = np.round(0.5 * (lb_temp + ub_temp), self.digits)
                diff = np.abs(ub_temp - lb_temp)
                while l_steps < diff:
                    som_pol, _, _ = self.backward_discreteliftedstate(arm, penalty)
                    if self.is_equal_mat(som_pol, ref_pol, self.n_realize[arm]):
                        lb_temp = penalty
                    else:
                        ub_temp = penalty
                    penalty = np.round(0.5 * (lb_temp + ub_temp), self.digits)
                    diff = np.abs(ub_temp - lb_temp)
                penalty_ref = penalty + l_steps
                nxt_pol, _, nxt_Q = self.backward_discreteliftedstate(arm, penalty_ref)
                indexability_flag, arm_indices = self.indexability_check(arm, arm_indices, self.n_realize[arm], nxt_pol, ref_pol, penalty, nxt_Q, ref_Q)
                if indexability_flag:
                    ref_pol = np.copy(nxt_pol)
                    ref_Q = np.copy(nxt_Q)
                else:
                    break
            self.w_indices.append(arm_indices)

    def backward_discreteliftedstate(self, arm, penalty):

        # Value function initialization
        V = np.zeros((self.n_augment[arm], self.num_x, self.horizon + 1), dtype=np.float32)
        for l in range(self.n_augment[arm]):
            V[l, :, self.horizon] = self.all_valus[arm][l] * np.ones(self.num_x)

        # State-action value function
        Q = np.zeros((self.n_augment[arm], self.num_x, self.horizon, 2), dtype=np.float32)

        # Policy function
        pi = np.zeros((self.n_augment[arm], self.num_x, self.horizon), dtype=np.int32)

        # Backward induction timing
        t = self.horizon - 1

        # The value iteration loop
        while t >= 0:

            # Loop over the first dimension of the state space
            for x in range(self.num_x):

                # Loop over the second dimension of the state space
                for l in range(self.n_realize[arm][t]):

                    # Get the state-action value functions
                    for act in range(2):

                        # Convert the next state of the second dimension into an index ranged from 1 to L
                        nxt_l = max(0, min(self.n_augment[arm] - 1, l + x))
                        Q[l, x, t, act] = np.round(- penalty * act / self.horizon + np.dot(V[nxt_l, :, t + 1], self.transition[x, :, act, arm, t]), self.digits + 1)

                    # Get the value function and the policy
                    if 0 < Q[l, x, t, 0] - Q[l, x, t, 1]:
                        V[l, x, t] = Q[l, x, t, 0]
                        pi[l, x, t] = 0
                    else:
                        V[l, x, t] = Q[l, x, t, 1]
                        pi[l, x, t] = 1

            t = t - 1

        return pi, V, Q

    @staticmethod
    def Whittle_policy(whittle_indices, n_selection, current_x, current_l, current_t):
        num_a = len(whittle_indices)

        current_indices = np.zeros(num_a)
        count_positive = 0
        for arm in range(num_a):
            w_idx = whittle_indices[arm][current_l[arm], current_x[arm], current_t]
            current_indices[arm] = w_idx
            if w_idx >= 0:
                count_positive += 1
        n_selection = np.minimum(n_selection, count_positive)

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
    def Whittle_softpolicy(whittle_indices, n_selection, current_x, current_l, current_t):
        num_a = len(whittle_indices)
        action = np.zeros(num_a, dtype=np.int32)

        current_indices = np.zeros(num_a)
        count_positive = 0
        for arm in range(num_a):
            w_idx = whittle_indices[arm][current_l[arm], current_x[arm], current_t]
            current_indices[arm] = w_idx
            if w_idx >= 0:
                count_positive += 1
        n_selection = np.minimum(n_selection, count_positive)

        softmax_probs = softmax(current_indices)
        sampled_indices = np.random.choice(range(num_a), n_selection, replace=False, p=softmax_probs)
        action[sampled_indices] = 1

        return action


class WhittleNSR:

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
                lb_temp = penalty_ref
                ub_temp = upper_bound
                penalty = 0.5 * (lb_temp + ub_temp)
                diff = np.abs(ub_temp - lb_temp)
                while l_steps < diff:
                    som_pol, _, _ = self.backward(arm, penalty)
                    if self.is_equal_mat(som_pol, ref_pol):
                        lb_temp = penalty
                    else:
                        ub_temp = penalty
                    penalty = 0.5 * (lb_temp + ub_temp)
                    diff = np.abs(ub_temp - lb_temp)
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
                    if len(self.reward.shape) == 4:
                        Q[x, t, act] = self.reward[x, act, arm, t] - penalty * act + np.dot(V[:, t + 1], self.transition[x, :, act, arm])
                    else:
                        Q[x, t, act] = self.reward[x, arm, t] - penalty * act + np.dot(V[:, t + 1], self.transition[x, :, act, arm])

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


class SafeWhittleNSR:

    def __init__(self, num_states: int, num_arms: int, rewards, transition, horizon, u_type, u_order, thresholds):
        self.num_x = num_states
        self.num_a = num_arms
        self.rewards = rewards
        self.transition = transition
        self.horizon = horizon
        self.u_type = u_type
        self.thresholds = thresholds

        self.digits = 3
        self.n_realize = []
        self.n_augment = [0] * num_arms
        self.all_rews = []
        self.all_valus = []

        for a in range(num_arms):
            arm_n_realize = []
            all_total_rewards = []
            for t in range(self.horizon):
                if len(self.rewards.shape) == 4:
                    immediate_rews = self.rewards[:, :, a, :t+1] 
                    all_immediate_rews = immediate_rews.reshape(-1, immediate_rews.shape[-1]) 
                else:
                    all_immediate_rews = self.rewards[:, a, :t+1]
                all_total_rewards_by_t = possible_ns_reward_sums(all_immediate_rews)
                arm_n_realize.append(len(all_total_rewards_by_t))
                if t == self.horizon - 1:
                    all_total_rewards = all_total_rewards_by_t

            self.n_augment[a] = len(all_total_rewards)
            self.all_rews.append(all_total_rewards)
            self.n_realize.append(arm_n_realize)

            arm_valus = []
            for total_rewards in all_total_rewards:
                if u_type == 1:
                    arm_valus.append(np.round(1 - thresholds[a] ** (- 1 / u_order) * (np.maximum(0, thresholds[a] - total_rewards)) ** (1 / u_order), 3))
                elif u_type == 2:
                    arm_valus.append(np.round((1 + np.exp(-u_order * (1 - thresholds[a]))) / (1 + np.exp(-u_order * (total_rewards - thresholds[a]))), 3))
                else:
                    arm_valus.append(1 if total_rewards >= thresholds[a] else 0)

            self.all_valus.append(arm_valus)
        self.w_indices = []

    def get_whittle_indices(self, computation_type, params, n_trials):
        if computation_type == 1:
            l_steps = params[1] / n_trials
            self.whittle_binary_search(params[0], params[1], l_steps)
        else:
            self.whittle_brute_force(params[0], params[1], n_trials)

    def is_equal_mat(self, mat1, mat2, realize_index):
        for t in range(self.horizon):
            mat1_new = mat1[:realize_index[t], :]
            mat2_new = mat2[:realize_index[t], :]
            if not np.array_equal(mat1_new, mat2_new):
                return False
        return True

    def indexability_check(self, arm, arm_indices, realize_index, nxt_pol, ref_pol, penalty, nxt_Q, ref_Q):
        for t in range(self.horizon):
            ref_pol_new = ref_pol[:realize_index[t], :, t]
            nxt_pol_new = nxt_pol[:realize_index[t], :, t]
            ref_Q_new = ref_Q[:realize_index[t], :, t, :]
            nxt_Q_new = nxt_Q[:realize_index[t], :, t, :]
            if np.any((ref_pol_new == 0) & (nxt_pol_new == 1)):
                print("Not indexable!")
                elements = np.argwhere((ref_pol_new == 0) & (nxt_pol_new == 1))
                for e in elements:
                    print(f'element: {[e[0], e[1], t]}')
                    print(f'penalty: {penalty}')
                    print(f'ref policy: {ref_pol_new[:, e[1]]}')
                    print(f'nxt policy: {nxt_pol_new[:, e[1]]}')
                    print(f'ref Q0: {ref_Q_new[e[0], e[1], 0]}')
                    print(f'ref Q1: {ref_Q_new[e[0], e[1], 1]}')
                    print(f'nxt Q0: {nxt_Q_new[e[0], e[1], 0]}')
                    print(f'nxt Q1: {nxt_Q_new[e[0], e[1], 1]}')
                return False, np.zeros((self.n_augment[arm], self.num_x, self.horizon))
            else:
                elements = np.argwhere((ref_pol_new == 1) & (nxt_pol_new == 0))
                for e in elements:
                    arm_indices[e[0], e[1], t] = penalty
        return True, arm_indices

    def whittle_brute_force(self, lower_bound, upper_bound, num_trials):

        for arm in range(self.num_a):
            arm_indices = np.zeros((self.n_augment[arm], self.num_x, self.horizon))
            penalty_ref = lower_bound
            ref_pol, _, ref_Q = self.backward_discreteliftedstate(arm, penalty_ref)
            upb_pol, _, _ = self.backward_discreteliftedstate(arm, upper_bound)
            for penalty in np.linspace(lower_bound, upper_bound, num_trials):
                penalty = np.round(penalty, self.digits)
                nxt_pol, _, nxt_Q = self.backward_discreteliftedstate(arm, penalty)
                if self.is_equal_mat(nxt_pol, upb_pol, self.n_realize[arm]):
                    flag, arm_indices = self.indexability_check(arm, arm_indices, self.n_realize[arm], nxt_pol, ref_pol, penalty, nxt_Q, ref_Q)
                    break
                else:
                    if not self.is_equal_mat(nxt_pol, ref_pol, self.n_realize[arm]):
                        flag, arm_indices = self.indexability_check(arm, arm_indices, self.n_realize[arm], nxt_pol, ref_pol, penalty, nxt_Q, ref_Q)
                        if flag:
                            ref_pol = np.copy(nxt_pol)
                            ref_Q = np.copy(nxt_Q)
                        else:
                            break
            self.w_indices.append(arm_indices)

    def whittle_binary_search(self, lower_bound, upper_bound, l_steps):

        for arm in range(self.num_a):
            arm_indices = np.zeros((self.n_augment[arm], self.num_x, self.horizon))
            penalty_ref = lower_bound
            ref_pol, _, ref_Q = self.backward_discreteliftedstate(arm, penalty_ref)
            ubp_pol, _, _ = self.backward_discreteliftedstate(arm, upper_bound)
            while not self.is_equal_mat(ref_pol, ubp_pol, self.n_realize[arm]):
                lb_temp = penalty_ref
                ub_temp = upper_bound
                penalty = np.round(0.5 * (lb_temp + ub_temp), self.digits)
                diff = np.abs(ub_temp - lb_temp)
                while l_steps < diff:
                    som_pol, _, _ = self.backward_discreteliftedstate(arm, penalty)
                    if self.is_equal_mat(som_pol, ref_pol, self.n_realize[arm]):
                        lb_temp = penalty
                    else:
                        ub_temp = penalty
                    penalty = np.round(0.5 * (lb_temp + ub_temp), self.digits)
                    diff = np.abs(ub_temp - lb_temp)
                penalty_ref = penalty + l_steps
                nxt_pol, _, nxt_Q = self.backward_discreteliftedstate(arm, penalty_ref)
                indexability_flag, arm_indices = self.indexability_check(arm, arm_indices, self.n_realize[arm], nxt_pol, ref_pol, penalty, nxt_Q, ref_Q)
                if indexability_flag:
                    ref_pol = np.copy(nxt_pol)
                    ref_Q = np.copy(nxt_Q)
                else:
                    break
            self.w_indices.append(arm_indices)

    def backward_discreteliftedstate(self, arm, penalty):

        # Value function initialization
        V = np.zeros((self.n_augment[arm], self.num_x, self.horizon + 1), dtype=np.float32)
        for l in range(self.n_augment[arm]):
            V[l, :, self.horizon] = self.all_valus[arm][l] * np.ones(self.num_x)

        # State-action value function
        Q = np.zeros((self.n_augment[arm], self.num_x, self.horizon, 2), dtype=np.float32)

        # Policy function
        pi = np.zeros((self.n_augment[arm], self.num_x, self.horizon), dtype=np.int32)

        # Backward induction timing
        t = self.horizon - 1

        # The value iteration loop
        while t >= 0:

            # Loop over the first dimension of the state space
            for x in range(self.num_x):

                # Loop over the second dimension of the state space
                for l in range(self.n_realize[arm][t]):

                    # Get the state-action value functions
                    for act in range(2):

                        # Convert the next state of the second dimension into an index ranged from 1 to L
                        nxt_l = max(0, min(self.n_augment[arm] - 1, l + x))
                        Q[l, x, t, act] = np.round(- penalty * act / self.horizon + np.dot(V[nxt_l, :, t + 1], self.transition[x, :, act, arm]), self.digits + 1)

                    # Get the value function and the policy
                    if 0 < Q[l, x, t, 0] - Q[l, x, t, 1]:
                        V[l, x, t] = Q[l, x, t, 0]
                        pi[l, x, t] = 0
                    else:
                        V[l, x, t] = Q[l, x, t, 1]
                        pi[l, x, t] = 1

            t = t - 1

        return pi, V, Q

    @staticmethod
    def Whittle_policy(whittle_indices, n_selection, current_x, current_l, current_t):
        num_a = len(whittle_indices)

        current_indices = np.zeros(num_a)
        count_positive = 0
        for arm in range(num_a):
            w_idx = whittle_indices[arm][current_l[arm], current_x[arm], current_t]
            current_indices[arm] = w_idx
            if w_idx >= 0:
                count_positive += 1
        n_selection = np.minimum(n_selection, count_positive)

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
    def Whittle_softpolicy(whittle_indices, n_selection, current_x, current_l, current_t):
        num_a = len(whittle_indices)
        action = np.zeros(num_a, dtype=np.int32)

        current_indices = np.zeros(num_a)
        count_positive = 0
        for arm in range(num_a):
            w_idx = whittle_indices[arm][current_l[arm], current_x[arm], current_t]
            current_indices[arm] = w_idx
            if w_idx >= 0:
                count_positive += 1
        n_selection = np.minimum(n_selection, count_positive)

        softmax_probs = softmax(current_indices)
        sampled_indices = np.random.choice(range(num_a), n_selection, replace=False, p=softmax_probs)
        action[sampled_indices] = 1

        return action


class SafeWhittleDNSR:

    def __init__(self, num_states: int, num_arms: int, rewards, transition, horizon, u_type, u_order, thresholds):
        self.num_x = num_states[0]
        self.num_s = num_states[1]
        self.cutting_points = np.linspace(0, 1, self.num_s+1)
        self.all_total_rewards = [np.median(self.cutting_points[i:i + 2]) for i in range(len(self.cutting_points) - 1)]
        self.num_a = num_arms
        self.rewards = rewards
        self.transition = transition
        self.horizon = horizon
        self.u_type = u_type
        self.thresholds = thresholds

        self.digits = 3
        self.n_realize = []
        self.n_augment = [0] * num_arms
        self.all_rews = []
        self.all_valus = []

        for a in range(num_arms):
            self.n_augment[a] = len(self.all_total_rewards)
            self.all_rews.append(self.all_total_rewards)
            self.n_realize.append([self.num_s] * self.horizon)

            arm_valus = []
            for total_rewards in self.all_total_rewards:
                if u_type == 1:
                    arm_valus.append(np.round(1 - thresholds[a] ** (- 1 / u_order) * (np.maximum(0, thresholds[a] - total_rewards)) ** (1 / u_order), 3))
                elif u_type == 2:
                    arm_valus.append(np.round((1 + np.exp(-u_order * (1 - thresholds[a]))) / (1 + np.exp(-u_order * (total_rewards - thresholds[a]))), 3))
                else:
                    arm_valus.append(1 if total_rewards >= thresholds[a] else 0)

            self.all_valus.append(arm_valus)
        self.w_indices = []

    def get_whittle_indices(self, computation_type, params, n_trials):
        if computation_type == 1:
            l_steps = params[1] / n_trials
            self.whittle_binary_search(params[0], params[1], l_steps)
        else:
            self.whittle_brute_force(params[0], params[1], n_trials)

    def is_equal_mat(self, mat1, mat2, realize_index):
        for t in range(self.horizon):
            mat1_new = mat1[:realize_index[t], :]
            mat2_new = mat2[:realize_index[t], :]
            if not np.array_equal(mat1_new, mat2_new):
                return False
        return True

    def indexability_check(self, arm, arm_indices, realize_index, nxt_pol, ref_pol, penalty, nxt_Q, ref_Q):
        for t in range(self.horizon):
            ref_pol_new = ref_pol[:realize_index[t], :, t]
            nxt_pol_new = nxt_pol[:realize_index[t], :, t]
            ref_Q_new = ref_Q[:realize_index[t], :, t, :]
            nxt_Q_new = nxt_Q[:realize_index[t], :, t, :]
            if np.any((ref_pol_new == 0) & (nxt_pol_new == 1)):
                print("Not indexable!")
                elements = np.argwhere((ref_pol_new == 0) & (nxt_pol_new == 1))
                for e in elements:
                    print(f'element: {[e[0], e[1], t]}')
                    print(f'penalty: {penalty}')
                    print(f'ref policy: {ref_pol_new[:, e[1]]}')
                    print(f'nxt policy: {nxt_pol_new[:, e[1]]}')
                    print(f'ref Q0: {ref_Q_new[e[0], e[1], 0]}')
                    print(f'ref Q1: {ref_Q_new[e[0], e[1], 1]}')
                    print(f'nxt Q0: {nxt_Q_new[e[0], e[1], 0]}')
                    print(f'nxt Q1: {nxt_Q_new[e[0], e[1], 1]}')
                return False, np.zeros((self.n_augment[arm], self.num_x, self.horizon))
            else:
                elements = np.argwhere((ref_pol_new == 1) & (nxt_pol_new == 0))
                for e in elements:
                    arm_indices[e[0], e[1], t] = penalty
        return True, arm_indices

    def whittle_brute_force(self, lower_bound, upper_bound, num_trials):

        for arm in range(self.num_a):
            arm_indices = np.zeros((self.n_augment[arm], self.num_x, self.horizon))
            penalty_ref = lower_bound
            ref_pol, _, ref_Q = self.backward_discreteliftedstate(arm, penalty_ref)
            upb_pol, _, _ = self.backward_discreteliftedstate(arm, upper_bound)
            for penalty in np.linspace(lower_bound, upper_bound, num_trials):
                penalty = np.round(penalty, self.digits)
                nxt_pol, _, nxt_Q = self.backward_discreteliftedstate(arm, penalty)
                if self.is_equal_mat(nxt_pol, upb_pol, self.n_realize[arm]):
                    flag, arm_indices = self.indexability_check(arm, arm_indices, self.n_realize[arm], nxt_pol, ref_pol, penalty, nxt_Q, ref_Q)
                    break
                else:
                    if not self.is_equal_mat(nxt_pol, ref_pol, self.n_realize[arm]):
                        flag, arm_indices = self.indexability_check(arm, arm_indices, self.n_realize[arm], nxt_pol, ref_pol, penalty, nxt_Q, ref_Q)
                        if flag:
                            ref_pol = np.copy(nxt_pol)
                            ref_Q = np.copy(nxt_Q)
                        else:
                            break
            self.w_indices.append(arm_indices)

    def whittle_binary_search(self, lower_bound, upper_bound, l_steps):

        for arm in range(self.num_a):
            arm_indices = np.zeros((self.n_augment[arm], self.num_x, self.horizon))
            penalty_ref = lower_bound
            ref_pol, _, ref_Q = self.backward_discreteliftedstate(arm, penalty_ref)
            ubp_pol, _, _ = self.backward_discreteliftedstate(arm, upper_bound)
            while not self.is_equal_mat(ref_pol, ubp_pol, self.n_realize[arm]):
                lb_temp = penalty_ref
                ub_temp = upper_bound
                penalty = np.round(0.5 * (lb_temp + ub_temp), self.digits)
                diff = np.abs(ub_temp - lb_temp)
                while l_steps < diff:
                    som_pol, _, _ = self.backward_discreteliftedstate(arm, penalty)
                    if self.is_equal_mat(som_pol, ref_pol, self.n_realize[arm]):
                        lb_temp = penalty
                    else:
                        ub_temp = penalty
                    penalty = np.round(0.5 * (lb_temp + ub_temp), self.digits)
                    diff = np.abs(ub_temp - lb_temp)
                penalty_ref = penalty + l_steps
                nxt_pol, _, nxt_Q = self.backward_discreteliftedstate(arm, penalty_ref)
                indexability_flag, arm_indices = self.indexability_check(arm, arm_indices, self.n_realize[arm], nxt_pol, ref_pol, penalty, nxt_Q, ref_Q)
                if indexability_flag:
                    ref_pol = np.copy(nxt_pol)
                    ref_Q = np.copy(nxt_Q)
                else:
                    break
            self.w_indices.append(arm_indices)

    def backward_discreteliftedstate(self, arm, penalty):

        # Value function initialization
        V = np.zeros((self.n_augment[arm], self.num_x, self.horizon + 1), dtype=np.float32)
        for l in range(self.n_augment[arm]):
            V[l, :, self.horizon] = self.all_valus[arm][l] * np.ones(self.num_x)

        # State-action value function
        Q = np.zeros((self.n_augment[arm], self.num_x, self.horizon, 2), dtype=np.float32)

        # Policy function
        pi = np.zeros((self.n_augment[arm], self.num_x, self.horizon), dtype=np.int32)

        # Backward induction timing
        t = self.horizon - 1

        # The value iteration loop
        while t >= 0:

            # Loop over the first dimension of the state space
            for x in range(self.num_x):

                # Loop over the second dimension of the state space
                for l in range(self.n_realize[arm][t]):

                    # Get the state-action value functions
                    for act in range(2):

                        # Convert the next state of the second dimension into an index ranged from 1 to L
                        nxt_l = max(0, min(self.n_augment[arm] - 1, l + x))
                        Q[l, x, t, act] = np.round(- penalty * act / self.horizon + np.dot(V[nxt_l, :, t + 1], self.transition[x, :, act, arm]), self.digits + 1)

                    # Get the value function and the policy
                    if 0 < Q[l, x, t, 0] - Q[l, x, t, 1]:
                        V[l, x, t] = Q[l, x, t, 0]
                        pi[l, x, t] = 0
                    else:
                        V[l, x, t] = Q[l, x, t, 1]
                        pi[l, x, t] = 1

            t = t - 1

        return pi, V, Q

    @staticmethod
    def Whittle_policy(whittle_indices, n_selection, current_x, current_l, current_t):
        num_a = len(whittle_indices)

        current_indices = np.zeros(num_a)
        count_positive = 0
        for arm in range(num_a):
            w_idx = whittle_indices[arm][current_l[arm], current_x[arm], current_t]
            current_indices[arm] = w_idx
            if w_idx >= 0:
                count_positive += 1
        n_selection = np.minimum(n_selection, count_positive)

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
    def Whittle_softpolicy(whittle_indices, n_selection, current_x, current_l, current_t):
        num_a = len(whittle_indices)
        action = np.zeros(num_a, dtype=np.int32)

        current_indices = np.zeros(num_a)
        count_positive = 0
        for arm in range(num_a):
            w_idx = whittle_indices[arm][current_l[arm], current_x[arm], current_t]
            current_indices[arm] = w_idx
            if w_idx >= 0:
                count_positive += 1
        n_selection = np.minimum(n_selection, count_positive)

        softmax_probs = softmax(current_indices)
        sampled_indices = np.random.choice(range(num_a), n_selection, replace=False, p=softmax_probs)
        action[sampled_indices] = 1

        return action


def possible_ns_reward_sums(rewards_so_far):
    reward_combinations = product(*rewards_so_far)
    sums = set()
    for combination in reward_combinations:
        sums.add(np.round(sum(combination), 3))
    return sorted(sums)


class WhittleDis:
    def __init__(self, beta, num_states: int, num_arms: int, reward, transition):
        self.beta = beta
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
            ref_pol, _, _ = self.bellman_equation(arm, penalty_ref)
            upb_pol, _, _ = self.bellman_equation(arm, upper_bound)
            for penalty in np.linspace(lower_bound, upper_bound, num_trials):
                nxt_pol, _, _ = self.bellman_equation(arm, penalty)
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
            ref_pol, _, _ = self.bellman_equation(arm, penalty_ref)
            ubp_pol, _, _ = self.bellman_equation(arm, upper_bound)
            while not np.array_equal(ref_pol, ubp_pol):
                lb_temp = penalty_ref
                ub_temp = upper_bound
                penalty = 0.5 * (lb_temp + ub_temp)
                diff = np.abs(ub_temp - lb_temp)
                while l_steps < diff:
                    som_pol, _, _ = self.bellman_equation(arm, penalty)
                    if np.array_equal(som_pol, ref_pol):
                        lb_temp = penalty
                    else:
                        ub_temp = penalty
                    penalty = 0.5 * (lb_temp + ub_temp)
                    diff = np.abs(ub_temp - lb_temp)
                penalty_ref = penalty + l_steps
                nxt_pol, _, _ = self.bellman_equation(arm, penalty_ref)
                flag, arm_indices = self.indexability_check(arm_indices, nxt_pol, ref_pol, penalty)
                if flag:
                    ref_pol = np.copy(nxt_pol)
                else:
                    break
            self.w_indices.append(arm_indices)

    def bellman_equation(self, arm, penalty):
        # Initialize value function
        V = np.zeros(self.num_x, dtype=np.float32)
        W = np.zeros(self.num_x, self.num_y, self.num_z, dtype=np.float32)
        Q = np.zeros((self.num_x, 2), dtype=np.float32)
        pi = np.zeros(self.num_x, dtype=np.int32)

        # Value iteration
        diff = np.inf
        iteration = 0
        while diff > 1e-4 and iteration < 1000:
            V_prev = np.copy(V)
            for x in range(self.num_x):
                for act in range(2):
                    if len(self.reward.shape) == 3:
                        Q[x, act] = (1 - self.beta) * (self.reward[x, act, arm] - penalty * act) \
                                    + self.beta * np.dot(V, self.transition[x, :, act, arm])
                    else:
                        Q[x, act] = (1 - self.beta) * (self.reward[x, arm] - penalty * act) \
                                    + self.beta * np.dot(V, self.transition[x, :, act, arm])
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


class SafeWhittleDis:

    def __init__(self, beta, num_states, num_arms: int, rewards, transition, horizon, u_type, u_order, thresholds):

        self.beta = beta
        self.num_x = num_states[0]
        self.num_s = num_states[1]
        self.cutting_points = np.linspace(0, 1, self.num_s+1)
        self.all_total_rewards = [np.median(self.cutting_points[i:i + 2]) for i in range(len(self.cutting_points) - 1)]
        self.num_a = num_arms
        self.rewards = rewards
        self.transition = transition
        self.horizon = horizon
        self.u_type = u_type
        self.thresholds = thresholds

        self.digits = 3
        self.n_realize = []
        self.n_augment = [0] * num_arms
        self.all_rews = []
        self.all_valus = []

        for a in range(num_arms):
            self.n_augment[a] = len(self.all_total_rewards)
            self.all_rews.append(self.all_total_rewards)
            self.n_realize.append([self.num_s] * self.horizon)

            arm_valus = []
            for total_rewards in self.all_total_rewards:
                if u_type == 1:
                    arm_valus.append(np.round(
                        1 - thresholds[a] ** (- 1 / u_order) * (np.maximum(0, thresholds[a] - total_rewards)) ** (
                                1 / u_order), 3))
                elif u_type == 2:
                    arm_valus.append(np.round((1 + np.exp(-u_order * (1 - thresholds[a]))) / (
                            1 + np.exp(-u_order * (total_rewards - thresholds[a]))), 3))
                else:
                    arm_valus.append(1 if total_rewards >= thresholds[a] else 0)
            self.all_valus.append(arm_valus)

        self.w_indices = []

    def get_reward_partition(self, reward_value):
        index = np.searchsorted(self.cutting_points, reward_value, side='right')
        if index == len(self.cutting_points):
            index -= 1

        return index - 1

    def get_whittle_indices(self, computation_type, params, n_trials):
        if computation_type == 1:
            l_steps = params[1] / n_trials
            self.whittle_binary_search(params[0], params[1], l_steps)
        else:
            self.whittle_brute_force(params[0], params[1], n_trials)

    def indexability_check(self, arm, arm_indices, nxt_pol, ref_pol, penalty, nxt_Q, ref_Q):
        if np.any((ref_pol == 0) & (nxt_pol == 1)):
            print("Not indexable!")
            elements = np.argwhere((ref_pol == 0) & (nxt_pol == 1))
            for e in elements:
                print(f'element: {[e[0], e[1]]}')
                print(f'penalty: {penalty}')
                print(f'ref policy: {ref_pol[:, e[1]]}')
                print(f'nxt policy: {nxt_pol[:, e[1]]}')
                print(f'ref Q0: {ref_Q[e[0], e[1], 0]}')
                print(f'ref Q1: {ref_Q[e[0], e[1], 1]}')
                print(f'nxt Q0: {nxt_Q[e[0], e[1], 0]}')
                print(f'nxt Q1: {nxt_Q[e[0], e[1], 1]}')
            return False, np.zeros((self.n_augment[arm], self.num_x))
        else:
            elements = np.argwhere((ref_pol == 1) & (nxt_pol == 0))
            for e in elements:
                arm_indices[e[0], e[1]] = penalty
        return True, arm_indices

    def whittle_brute_force(self, lower_bound, upper_bound, num_trials):
        for arm in range(self.num_a):
            arm_indices = np.zeros((self.n_augment[arm], self.num_x))
            penalty_ref = lower_bound
            ref_pol, _, ref_Q = self.bellman_equation(arm, penalty_ref)
            upb_pol, _, _ = self.bellman_equation(arm, upper_bound)
            for penalty in np.linspace(lower_bound, upper_bound, num_trials):
                penalty = np.round(penalty, self.digits)
                nxt_pol, _, nxt_Q = self.bellman_equation(arm, penalty)
                if np.array_equal(nxt_pol, upb_pol):
                    flag, arm_indices = self.indexability_check(arm, arm_indices, nxt_pol, ref_pol, penalty, nxt_Q, ref_Q)
                    break
                else:
                    if not np.array_equal(nxt_pol, ref_pol):
                        flag, arm_indices = self.indexability_check(arm, arm_indices, nxt_pol, ref_pol, penalty, nxt_Q, ref_Q)
                        if flag:
                            ref_pol = np.copy(nxt_pol)
                            ref_Q = np.copy(nxt_Q)
                        else:
                            break
            self.w_indices.append(arm_indices)

    def whittle_binary_search(self, lower_bound, upper_bound, l_steps):
        for arm in range(self.num_a):
            arm_indices = np.zeros((self.n_augment[arm], self.num_x))
            penalty_ref = lower_bound
            ref_pol, _, ref_Q = self.bellman_equation(arm, penalty_ref)
            ubp_pol, _, _ = self.bellman_equation(arm, upper_bound)
            while not np.array_equal(ref_pol, ubp_pol):
                lb_temp = penalty_ref
                ub_temp = upper_bound
                penalty = np.round(0.5 * (lb_temp + ub_temp), self.digits)
                diff = np.abs(ub_temp - lb_temp)
                while l_steps < diff:
                    som_pol, _, _ = self.bellman_equation(arm, penalty)
                    if np.array_equal(som_pol, ref_pol):
                        lb_temp = penalty
                    else:
                        ub_temp = penalty
                    penalty = np.round(0.5 * (lb_temp + ub_temp), self.digits)
                    diff = np.abs(ub_temp - lb_temp)
                penalty_ref = penalty + l_steps
                nxt_pol, _, nxt_Q = self.bellman_equation(arm, penalty_ref)
                indexability_flag, arm_indices = self.indexability_check(arm, arm_indices, nxt_pol, ref_pol, penalty, nxt_Q, ref_Q)
                if indexability_flag:
                    ref_pol = np.copy(nxt_pol)
                    ref_Q = np.copy(nxt_Q)
                else:
                    break
            self.w_indices.append(arm_indices)

    def bellman_equation(self, arm, penalty):

        # Value function initialization
        V = np.zeros((self.n_augment[arm], self.num_x, self.horizon + 1), dtype=np.float32)
        for l in range(self.n_augment[arm]):
            V[l, :, self.horizon] = self.all_valus[arm][l] * np.ones(self.num_x)

        # State-action value function
        Q = np.zeros((self.n_augment[arm], self.num_x, self.horizon, 2), dtype=np.float32)

        # Policy function
        pi = np.zeros((self.n_augment[arm], self.num_x, self.horizon), dtype=np.int32)

        # Backward induction timing
        t = self.horizon - 1

        # The value iteration loop
        while t >= 0:

            # Loop over the first dimension of the state space
            for x in range(self.num_x):

                # Loop over the second dimension of the state space
                for l in range(self.n_realize[arm][t]):
                    for act in range(2):

                        if len(self.rewards.shape) == 3:
                            imm_reward = self.rewards[x, act, arm]
                        else:
                            imm_reward = self.rewards[x, arm]

                        nxt_l = self.get_reward_partition(self.all_total_rewards[l] + (1 - self.beta) * (self.beta ** t) * imm_reward)
                        # nxt_l = max(0, min(self.n_augment[arm] - 1, l + x))

                        Q[l, x, t, act] = - (1 - self.beta) * (self.beta ** t) * penalty * act \
                            + np.dot(V[nxt_l, :, t + 1], self.transition[x, :, act, arm])

                    # Get the value function and the policy
                    if 0 < Q[l, x, t, 0] - Q[l, x, t, 1]:
                        V[l, x, t] = Q[l, x, t, 0]
                        pi[l, x, t] = 0
                    else:
                        V[l, x, t] = Q[l, x, t, 1]
                        pi[l, x, t] = 1

            diff = np.max(np.abs(V[:, :, t] - V[:, :, t+1]))

            if diff >= 1e-4:
                t = t - 1
            else:
                return pi[:, :, t], V[:, :, t], Q[:, :, t, :]

        return pi[:, :, 0], V[:, :, 0], Q[:, :, 0, :]

    @staticmethod
    def Whittle_policy(whittle_indices, n_selection, current_x, current_l):
        num_a = len(whittle_indices)

        current_indices = np.zeros(num_a)
        count_positive = 0
        for arm in range(num_a):
            w_idx = whittle_indices[arm][current_l[arm], current_x[arm]]
            current_indices[arm] = w_idx
            if w_idx >= 0:
                count_positive += 1
        n_selection = np.minimum(n_selection, count_positive)

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
    def Whittle_softpolicy(whittle_indices, n_selection, current_x, current_l):
        num_a = len(whittle_indices)
        action = np.zeros(num_a, dtype=np.int32)

        current_indices = np.zeros(num_a)
        count_positive = 0
        for arm in range(num_a):
            w_idx = whittle_indices[arm][current_l[arm], current_x[arm]]
            current_indices[arm] = w_idx
            if w_idx >= 0:
                count_positive += 1
        n_selection = np.minimum(n_selection, count_positive)

        softmax_probs = softmax(current_indices)
        sampled_indices = np.random.choice(range(num_a), n_selection, replace=False, p=softmax_probs)
        action[sampled_indices] = 1

        return action
