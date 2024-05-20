from scipy.stats import dirichlet
import joblib
from Markov import *
from safe_whittle import *
import time


def Process_SafeTSRB(n_iterations, n_episodes, n_steps, n_states, n_arms, n_choices, thresholds, t_type, t_increasing,
                     method, tru_rew, tru_dyn, initial_states, u_type, u_order, save_data):
    n_trials_safety = n_arms * n_states * n_steps

    ##################################################### Process
    totalrewards = np.zeros((n_iterations, n_arms, n_episodes))
    objectives = np.zeros((n_iterations, n_arms, n_episodes))
    sum_wi = np.zeros((n_iterations, n_arms, n_episodes))

    est_prob = np.round((1 / n_states), 2) * np.ones((n_iterations, n_arms, n_episodes))
    counts = np.ones((n_iterations, n_states, n_states, 2, n_arms))

    for i in range(n_iterations):
        start_time = time.time()
        print(f'Iteration {i + 1} out of {n_iterations}')

        M = MarkovDynamics(n_arms, n_states, est_prob[i, :, 0], t_type, t_increasing)
        SafeW = SafeWhittle(n_states, n_arms, tru_rew, M.transitions, n_steps, u_type, u_order, thresholds)
        SafeW.get_whittle_indices(computation_type=method, params=[0, 1], n_trials=n_trials_safety)
        sw_indices = SafeW.w_indices

        for k in range(n_episodes):
            # print(f'Episode {k}')
            states = initial_states.copy()
            _lifted = np.zeros(n_arms, dtype=np.int32)
            for t in range(n_steps):
                _states = np.copy(states)
                for a in range(n_arms):
                    _lifted[a] = max(0, min(SafeW.n_augment[a] - 1, _lifted[a] + _states[a]))
                actions = SafeW.Whittle_policy(sw_indices, n_choices, _states, _lifted, t)
                for a in range(n_arms):
                    if len(tru_rew.shape) == 3:
                        totalrewards[i, a, k] += tru_rew[_states[a], actions[a], a]
                    else:
                        totalrewards[i, a, k] += tru_rew[_states[a], a]
                    if u_type == 1:
                        objectives[i, a, k] = 1 - thresholds[a] ** (- 1 / u_order) * (np.maximum(0, thresholds[a] - totalrewards[i, a, k])) ** (1 / u_order)
                    else:
                        objectives[i, a, k] = (1 + np.exp(-u_order * (1 - thresholds[a]))) / (1 + np.exp(-u_order * (totalrewards[i, a, k] - thresholds[a])))

                    states[a] = np.random.choice(n_states, p=tru_dyn[_states[a], :, actions[a], a])
                    counts[i, _states[a], states[a], actions[a], a] += 1
            # print('Update...')

            est_transitions = np.zeros((n_states, n_states, 2, n_arms))
            for a in range(n_arms):
                for s1 in range(n_states):
                    for act in range(2):
                        est_transitions[s1, :, act, a] = dirichlet.rvs(counts[i, s1, :, act, a])
                if t_type == 5:
                    cnt = [est_transitions[0, -1, 1, a]]
                    for s1 in range(1, n_states):
                        cnt.append((1 / (s1 + 1)) * est_transitions[s1, -1, 1, a])
                        for s2 in range(1, s1):
                            cnt.append(est_transitions[s1, s2, 0, a])
                    est_prob[i, a, k] = np.minimum(np.maximum(0.1 / n_states, np.mean(cnt)), 1 / n_states)
                if t_type == 3:
                    cnt = [(1 / (n_states - 1)) * est_transitions[0, 0, 1, a]]
                    for s1 in range(1, n_states - 1):
                        cnt.append((1 / (n_states - s1 - 1)) * est_transitions[s1, s1, 1, a])
                        cnt.append((1 / (n_states - s1)) * est_transitions[s1, s1, 0, a])
                        for s2 in range(1, s1):
                            cnt.append(est_transitions[s1, s2, 0, a])
                    for s2 in range(1, n_states):
                        cnt.append(est_transitions[n_states - 1, s2, 0, a])
                    est_prob[i, a, k] = np.minimum(np.maximum(0.1 / n_states, np.mean(cnt)), 1 / n_states)

            Mest = MarkovDynamics(n_arms, n_states, est_prob[i, :, k], t_type, t_increasing)
            dynamics = Mest.transitions

            SafeW = SafeWhittle(n_states, n_arms, tru_rew, dynamics, n_steps, u_type, u_order, thresholds)
            SafeW.get_whittle_indices(computation_type=method, params=[0, 1], n_trials=n_trials_safety)
            sw_indices = SafeW.w_indices

            for a in range(n_arms):
                sum_wi[i, a, k] = np.sum(sw_indices[a])

        end_time = time.time()
        print(end_time - start_time)

        if save_data:
            joblib.dump([counts, totalrewards, objectives], f'./output/safetsrb_{n_steps}{n_states}{n_arms}{t_type}{u_type}{n_choices}{thresholds[0]}.joblib')

    return totalrewards, objectives, est_prob, counts, sum_wi
