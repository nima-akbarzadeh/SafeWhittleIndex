import random

from scipy.stats import dirichlet
import joblib
from Markov import *
from safe_whittle import *
from processes import *
import time


def Process_SafeTSRB(n_iterations, l_episodes, n_episodes, n_steps, n_states, n_arms, n_choices, thresholds, t_type, t_increasing,
                     method, tru_rew, tru_dyn, initial_states, u_type, u_order, save_data, max_wi):
    n_trials_safety = n_arms * n_states * n_steps

    ##################################################### Process
    all_rewards = np.zeros((n_iterations, l_episodes, n_arms))
    all_objectives = np.zeros((n_iterations, l_episodes, n_arms))
    all_sumwis = np.zeros((n_iterations, l_episodes, n_arms))
    all_probs = np.round((0.1 / n_states), 2) * np.ones((n_iterations, l_episodes, n_arms))
    duration = 0

    for n in range(n_iterations):

        # print(f'Learning iteration {n + 1} out of {n_iterations}')
        start_time = time.time()
        M = MarkovDynamics(n_arms, n_states, all_probs[n, :, 0], t_type, t_increasing)
        SafeW = SafeWhittle(n_states, n_arms, tru_rew, M.transitions, n_steps, u_type, u_order, thresholds)
        SafeW.get_whittle_indices(computation_type=method, params=[0, max_wi], n_trials=n_trials_safety)
        sw_indices = SafeW.w_indices
        counts = np.ones((n_states, n_states, 2, n_arms))

        for l in range(l_episodes):

            print(f'Episode {l + 1} of {l_episodes} / Iteration {n} / last iteration was {np.round(duration, 1)} seconds')
            totalrewards = np.zeros((n_arms, n_episodes))
            objectives = np.zeros((n_arms, n_episodes))

            for k in range(n_episodes):

                states = initial_states.copy()
                _lifted = np.zeros(n_arms, dtype=np.int32)
                for t in range(n_steps):
                    _states = np.copy(states)
                    for a in range(n_arms):
                        _lifted[a] = max(0, min(SafeW.n_augment[a] - 1, _lifted[a] + _states[a]))
                    actions = SafeW.Whittle_policy(sw_indices, n_choices, _states, _lifted, t)
                    for a in range(n_arms):
                        if len(tru_rew.shape) == 3:
                            totalrewards[a, k] += tru_rew[_states[a], actions[a], a]
                        else:
                            totalrewards[a, k] += tru_rew[_states[a], a]
                        states[a] = np.random.choice(n_states, p=tru_dyn[_states[a], :, actions[a], a])
                        counts[_states[a], states[a], actions[a], a] += 1
                for a in range(n_arms):
                    if u_type == 1:
                        objectives[a, k] = 1 - thresholds[a] ** (- 1 / u_order) * (np.maximum(0, thresholds[a] - totalrewards[a, k])) ** (1 / u_order)
                    elif u_type == 2:
                        objectives[a, k] = (1 + np.exp(-u_order * (1 - thresholds[a]))) / (1 + np.exp(-u_order * (totalrewards[a, k] - thresholds[a])))
                    else:
                        objectives[a, k] = 1 if totalrewards[a, k] >= thresholds[a] else 0

            # print('Update...')
            est_transitions = np.zeros((n_states, n_states, 2, n_arms))
            for a in range(n_arms):
                for s1 in range(n_states):
                    for act in range(2):
                        est_transitions[s1, :, act, a] = dirichlet.rvs(counts[s1, :, act, a])
                if t_type == 5:
                    cnt = [est_transitions[0, -1, 1, a]]
                    for s1 in range(1, n_states):
                        cnt.append((1 / (s1 + 1)) * est_transitions[s1, -1, 1, a])
                        for s2 in range(1, s1):
                            cnt.append(est_transitions[s1, s2, 0, a])
                    all_probs[n, l, a] = np.round(np.minimum(np.maximum(0.1 / n_states, np.mean(cnt)), 1 / n_states), 2)
                if t_type == 3:
                    cnt = [(1 / (n_states - 1)) * est_transitions[0, 0, 1, a]]
                    for s1 in range(1, n_states - 1):
                        cnt.append((1 / (n_states - s1 - 1)) * est_transitions[s1, s1, 1, a])
                        cnt.append((1 / (n_states - s1)) * est_transitions[s1, s1, 0, a])
                        for s2 in range(1, s1):
                            cnt.append(est_transitions[s1, s2, 0, a])
                    for s2 in range(1, n_states):
                        cnt.append(est_transitions[n_states - 1, s2, 0, a])
                    all_probs[n, l, a] = np.round(np.minimum(np.maximum(0.1 / n_states, np.mean(cnt)), 1 / n_states), 2)

            Mest = MarkovDynamics(n_arms, n_states, all_probs[n, l, :], t_type, t_increasing)
            SafeW = SafeWhittle(n_states, n_arms, tru_rew, Mest.transitions, n_steps, u_type, u_order, thresholds)
            SafeW.get_whittle_indices(computation_type=method, params=[0, max_wi], n_trials=n_trials_safety)
            sw_indices = SafeW.w_indices

            for a in range(n_arms):
                all_sumwis[n, l, a] = np.sum(sw_indices[a])
                all_rewards[n, l, a] = np.mean(totalrewards[a, :])
                all_objectives[n, l, a] = np.mean(objectives[a, :])

        end_time = time.time()
        duration = end_time - start_time

    if save_data:
        joblib.dump([all_probs, all_sumwis, all_rewards, all_objectives], f'./output/safetsrb_{n_steps}{n_states}{n_arms}{t_type}{u_type}{n_choices}{thresholds[0]}.joblib')

    return all_probs, all_sumwis, all_rewards, all_objectives


def Process_SafeSoftTSRB(n_iterations, l_episodes, n_episodes, n_steps, n_states, n_arms, n_choices, thresholds, t_type, t_increasing,
                         method, tru_rew, tru_dyn, initial_states, u_type, u_order, save_data, max_wi):

    n_trials_safety = n_arms * n_states * n_steps

    ##################################################### Process
    all_rewards = np.zeros((n_iterations, l_episodes, n_arms))
    all_objectives = np.zeros((n_iterations, l_episodes, n_arms))
    all_sumwis = np.zeros((n_iterations, l_episodes, n_arms))
    all_probs = np.round((0.1 / n_states), 2) * np.ones((n_iterations, l_episodes, n_arms))
    duration = 0

    for n in range(n_iterations):

        # print(f'Iteration {n + 1} / {n_iterations}')
        start_time = time.time()
        M = MarkovDynamics(n_arms, n_states, all_probs[n, :, 0], t_type, t_increasing)
        SafeW = SafeWhittle(n_states, n_arms, tru_rew, M.transitions, n_steps, u_type, u_order, thresholds)
        SafeW.get_whittle_indices(computation_type=method, params=[0, max_wi], n_trials=n_trials_safety)
        sw_indices = SafeW.w_indices
        counts = np.ones((n_states, n_states, 2, n_arms))

        for l in range(l_episodes):

            print(f'Episode {l + 1} of {l_episodes} / Iteration {n} / last iteration was {np.round(duration, 1)} seconds')
            totalrewards = np.zeros((n_arms, n_episodes))
            objectives = np.zeros((n_arms, n_episodes))

            for k in range(n_episodes):

                states = initial_states.copy()
                _lifted = np.zeros(n_arms, dtype=np.int32)
                for t in range(n_steps):
                    _states = np.copy(states)
                    for a in range(n_arms):
                        _lifted[a] = max(0, min(SafeW.n_augment[a] - 1, _lifted[a] + _states[a]))
                    actions = SafeW.Whittle_softpolicy(sw_indices, n_choices, _states, _lifted, t)
                    for a in range(n_arms):
                        if len(tru_rew.shape) == 3:
                            totalrewards[a, k] += tru_rew[_states[a], actions[a], a]
                        else:
                            totalrewards[a, k] += tru_rew[_states[a], a]
                        states[a] = np.random.choice(n_states, p=tru_dyn[_states[a], :, actions[a], a])
                        counts[_states[a], states[a], actions[a], a] += 1
                for a in range(n_arms):
                    if u_type == 1:
                        objectives[a, k] = 1 - thresholds[a] ** (- 1 / u_order) * (np.maximum(0, thresholds[a] - totalrewards[a, k])) ** (1 / u_order)
                    elif u_type == 2:
                        objectives[a, k] = (1 + np.exp(-u_order * (1 - thresholds[a]))) / (1 + np.exp(-u_order * (totalrewards[a, k] - thresholds[a])))
                    else:
                        objectives[a, k] = 1 if totalrewards[a, k] >= thresholds[a] else 0

            # print('Update...')
            est_transitions = np.zeros((n_states, n_states, 2, n_arms))
            for a in range(n_arms):
                for s1 in range(n_states):
                    for act in range(2):
                        est_transitions[s1, :, act, a] = dirichlet.rvs(counts[s1, :, act, a])
                if t_type == 5:
                    cnt = [est_transitions[0, -1, 1, a]]
                    for s1 in range(1, n_states):
                        cnt.append((1 / (s1 + 1)) * est_transitions[s1, -1, 1, a])
                        for s2 in range(1, s1):
                            cnt.append(est_transitions[s1, s2, 0, a])
                    all_probs[n, l, a] = np.round(np.minimum(np.maximum(0.1 / n_states, np.mean(cnt)), 1 / n_states), 2)
                if t_type == 3:
                    cnt = [(1 / (n_states - 1)) * est_transitions[0, 0, 1, a]]
                    for s1 in range(1, n_states - 1):
                        cnt.append((1 / (n_states - s1 - 1)) * est_transitions[s1, s1, 1, a])
                        cnt.append((1 / (n_states - s1)) * est_transitions[s1, s1, 0, a])
                        for s2 in range(1, s1):
                            cnt.append(est_transitions[s1, s2, 0, a])
                    for s2 in range(1, n_states):
                        cnt.append(est_transitions[n_states - 1, s2, 0, a])
                    all_probs[n, l, a] = np.round(np.minimum(np.maximum(0.1 / n_states, np.mean(cnt)), 1 / n_states), 2)

            Mest = MarkovDynamics(n_arms, n_states, all_probs[n, l, :], t_type, t_increasing)
            SafeW = SafeWhittle(n_states, n_arms, tru_rew, Mest.transitions, n_steps, u_type, u_order, thresholds)
            SafeW.get_whittle_indices(computation_type=method, params=[0, max_wi], n_trials=n_trials_safety)
            sw_indices = SafeW.w_indices

            for a in range(n_arms):
                all_sumwis[n, l, a] = np.sum(sw_indices[a])
                all_rewards[n, l, a] = np.mean(totalrewards[a, :])
                all_objectives[n, l, a] = np.mean(objectives[a, :])

        end_time = time.time()
        duration = end_time - start_time

    if save_data:
        joblib.dump([all_probs, all_sumwis, all_rewards, all_objectives], f'./output/safesofttsrb_{n_steps}{n_states}{n_arms}{t_type}{u_type}{n_choices}{thresholds[0]}.joblib')

    return all_probs, all_sumwis, all_rewards, all_objectives


def Process_LearnSafeTSRB(n_iterations, l_episodes, n_episodes, n_steps, n_states, n_arms, n_choices, thresholds, t_type, t_increasing,
                          method, tru_rew, tru_dyn, initial_states, u_type, u_order, save_data, max_wi):

    n_trials_safety = n_states * n_steps
    ##################################################### Process
    all_plan_rewards = np.zeros((n_iterations, l_episodes, n_arms))
    all_plan_objectives = np.zeros((n_iterations, l_episodes, n_arms))
    all_learn_rewards = np.zeros((n_iterations, l_episodes, n_arms))
    all_learn_objectives = np.zeros((n_iterations, l_episodes, n_arms))
    all_learn_sumwis = np.zeros((n_iterations, l_episodes, n_arms))
    all_learn_probs = np.ones((n_iterations, l_episodes, n_arms))
    duration = 0

    PlanW = SafeWhittle(n_states, n_arms, tru_rew, tru_dyn, n_steps, u_type, u_order, thresholds)
    PlanW.get_whittle_indices(computation_type=method, params=[0, max_wi], n_trials=n_trials_safety)
    plan_indices = PlanW.w_indices
    plan_sumwis = np.round([np.sum(plan_indices[a]) for a in range(n_arms)], 2)

    for n in range(n_iterations):

        all_learn_probs[n, 0, :] = np.array([np.round(random.uniform(0.5/n_states, 0.5/n_states), 2) for _ in range(n_arms)])
        print(f'Learning iteration {n + 1} out of {n_iterations}')
        start_time = time.time()

        Mest = MarkovDynamics(n_arms, n_states, all_learn_probs[n, 0, :], t_type, t_increasing)
        LearnW = SafeWhittle(n_states, n_arms, tru_rew, Mest.transitions, n_steps, u_type, u_order, thresholds)
        LearnW.get_whittle_indices(computation_type=method, params=[0, max_wi], n_trials=n_trials_safety)
        learn_indices = LearnW.w_indices

        counts = np.ones((n_states, n_states, 2, n_arms))

        for l in range(l_episodes):

            # print(f'Episode {l + 1} of {l_episodes} / Iteration {n+1} / Last iteration was {np.round(duration, 1)} seconds')
            # totalrewards, objectives, cnts = Process_SingleSoftSafeRB(SafeW, n_episodes, n_steps, n_states, 2, n_choices, thresholds, tru_rew,
            #                                                           tru_dyn, sw_indices, initial_states, u_type, u_order)
            plan_totalrewards, plan_objectives, learn_totalrewards, learn_objectives, cnts = Process_LearnSafeRB(PlanW, plan_indices, LearnW, learn_indices, n_episodes, n_steps, n_states, n_arms,
                                                                                                                 n_choices, thresholds, tru_rew, tru_dyn, initial_states, u_type, u_order)
            counts = counts + cnts

            # print('Update...')
            est_transitions = np.zeros((n_states, n_states, 2, n_arms))
            for a in range(n_arms):
                for s1 in range(n_states):
                    for act in range(2):
                        est_transitions[s1, :, act, a] = dirichlet.rvs(counts[s1, :, act, a])
                if t_type == 5:
                    cnt = [est_transitions[0, -1, 1, a]]
                    for s1 in range(1, n_states):
                        cnt.append((1 / (s1 + 1)) * est_transitions[s1, -1, 1, a])
                        for s2 in range(1, s1):
                            cnt.append(est_transitions[s1, s2, 0, a])
                    all_learn_probs[n, l, a] = np.round(np.minimum(np.maximum(0.1 / n_states, np.mean(cnt)), 1 / n_states), 2)
                if t_type == 3:
                    cnt = [(1 / (n_states - 1)) * est_transitions[0, 0, 1, a]]
                    for s1 in range(1, n_states - 1):
                        cnt.append((1 / (n_states - s1 - 1)) * est_transitions[s1, s1, 1, a])
                        cnt.append((1 / (n_states - s1)) * est_transitions[s1, s1, 0, a])
                        for s2 in range(1, s1):
                            cnt.append(est_transitions[s1, s2, 0, a])
                    for s2 in range(1, n_states):
                        cnt.append(est_transitions[n_states - 1, s2, 0, a])
                    all_learn_probs[n, l, a] = np.round(np.minimum(np.maximum(0.1 / n_states, np.mean(cnt)), 1 / n_states), 2)
            # print(all_learn_probs[n, l, :])
            Mest = MarkovDynamics(n_arms, n_states, all_learn_probs[n, l, :], t_type, t_increasing)
            SafeW = SafeWhittle(n_states, n_arms, tru_rew, Mest.transitions, n_steps, u_type, u_order, thresholds)
            SafeW.get_whittle_indices(computation_type=method, params=[0, max_wi], n_trials=n_trials_safety)
            sw_indices = SafeW.w_indices

            for a in range(n_arms):
                all_learn_sumwis[n, l, a] = np.round(np.sum(sw_indices[a]), 2)
                all_plan_rewards[n, l, a] = np.round(np.mean(plan_totalrewards[a, :]), 2)
                all_plan_objectives[n, l, a] = np.round(np.mean(plan_objectives[a, :]), 2)
                all_learn_rewards[n, l, a] = np.round(np.mean(learn_totalrewards[a, :]), 2)
                all_learn_objectives[n, l, a] = np.round(np.mean(learn_objectives[a, :]), 2)

            # print(f'pr = {np.round(np.mean(all_learn_probs[n, l, :]), 2)}')
            # print(f'true-sw = {np.round(np.mean(plan_sumwis), 2)}')
            # print(f'true-re = {np.round(np.mean(all_plan_rewards[n, l, :]), 2)}')
            # print(f'true-ob = {np.round(np.mean(all_plan_objectives[n, l, :]), 2)}')
            # print(f'sw = {np.round(np.mean(all_learn_sumwis[n, l, :]), 2)}')
            # print(f're = {np.round(np.mean(all_learn_rewards[n, l, :]), 2)}')
            # print(f'ob = {np.round(np.mean(all_learn_objectives[n, l, :]), 2)}')

        end_time = time.time()
        duration = end_time - start_time

    if save_data:
        joblib.dump([all_learn_probs, all_learn_sumwis, all_learn_rewards, all_learn_objectives, all_plan_rewards, all_plan_objectives],
                    f'./output/learnsafetsrb_{n_steps}{n_states}{t_type}{u_type}{n_choices}{thresholds[0]}.joblib')

    return all_learn_probs, all_learn_sumwis, all_learn_rewards, all_learn_objectives, plan_sumwis, all_plan_rewards, all_plan_objectives


def Process_LearnSoftSafeTSRB(n_iterations, l_episodes, n_episodes, n_steps, n_states, n_arms, n_choices, thresholds, t_type, t_increasing,
                              method, tru_rew, tru_dyn, initial_states, u_type, u_order, save_data, max_wi):

    n_trials_safety = n_states * n_steps
    ##################################################### Process
    all_plan_rewards = np.zeros((n_iterations, l_episodes, n_arms))
    all_plan_objectives = np.zeros((n_iterations, l_episodes, n_arms))
    all_learn_rewards = np.zeros((n_iterations, l_episodes, n_arms))
    all_learn_objectives = np.zeros((n_iterations, l_episodes, n_arms))
    all_learn_sumwis = np.zeros((n_iterations, l_episodes, n_arms))
    all_learn_probs = np.ones((n_iterations, l_episodes, n_arms))
    duration = 0

    PlanW = SafeWhittle(n_states, n_arms, tru_rew, tru_dyn, n_steps, u_type, u_order, thresholds)
    PlanW.get_whittle_indices(computation_type=method, params=[0, max_wi], n_trials=n_trials_safety)
    plan_indices = PlanW.w_indices
    plan_sumwis = np.round([np.sum(plan_indices[a]) for a in range(n_arms)], 2)

    for n in range(n_iterations):

        print(f'Learning iteration {n + 1} out of {n_iterations}')
        start_time = time.time()

        all_learn_probs[n, 0, :] = np.array([np.round(random.uniform(0.5/n_states, 0.5/n_states), 2) for _ in range(n_arms)])
        Mest = MarkovDynamics(n_arms, n_states, all_learn_probs[n, 0, :], t_type, t_increasing)
        LearnW = SafeWhittle(n_states, n_arms, tru_rew, Mest.transitions, n_steps, u_type, u_order, thresholds)
        LearnW.get_whittle_indices(computation_type=method, params=[0, max_wi], n_trials=n_trials_safety)
        learn_indices = LearnW.w_indices

        counts = np.ones((n_states, n_states, 2, n_arms))

        for l in range(l_episodes):

            # print(f'Episode {l + 1} of {l_episodes} / Iteration {n+1} / Last iteration was {np.round(duration, 1)} seconds')
            # totalrewards, objectives, cnts = Process_SingleSoftSafeRB(SafeW, n_episodes, n_steps, n_states, 2, n_choices, thresholds, tru_rew,
            #                                                           tru_dyn, sw_indices, initial_states, u_type, u_order)
            plan_totalrewards, plan_objectives, learn_totalrewards, learn_objectives, cnts = Process_LearnSoftSafeRB(PlanW, plan_indices, LearnW, learn_indices, n_episodes, n_steps, n_states, n_arms,
                                                                                                                     n_choices, thresholds, tru_rew, tru_dyn, initial_states, u_type, u_order)
            counts = counts + cnts

            # print('Update...')
            est_transitions = np.zeros((n_states, n_states, 2, n_arms))
            for a in range(n_arms):
                for s1 in range(n_states):
                    for act in range(2):
                        est_transitions[s1, :, act, a] = dirichlet.rvs(counts[s1, :, act, a])
                if t_type == 5:
                    cnt = [est_transitions[0, -1, 1, a]]
                    for s1 in range(1, n_states):
                        cnt.append((1 / (s1 + 1)) * est_transitions[s1, -1, 1, a])
                        for s2 in range(1, s1):
                            cnt.append(est_transitions[s1, s2, 0, a])
                    all_learn_probs[n, l, a] = np.round(np.minimum(np.maximum(0.1 / n_states, np.mean(cnt)), 1 / n_states), 2)
                if t_type == 3:
                    cnt = [(1 / (n_states - 1)) * est_transitions[0, 0, 1, a]]
                    for s1 in range(1, n_states - 1):
                        cnt.append((1 / (n_states - s1 - 1)) * est_transitions[s1, s1, 1, a])
                        cnt.append((1 / (n_states - s1)) * est_transitions[s1, s1, 0, a])
                        for s2 in range(1, s1):
                            cnt.append(est_transitions[s1, s2, 0, a])
                    for s2 in range(1, n_states):
                        cnt.append(est_transitions[n_states - 1, s2, 0, a])
                    all_learn_probs[n, l, a] = np.round(np.minimum(np.maximum(0.1 / n_states, np.mean(cnt)), 1 / n_states), 2)
            # print(all_learn_probs[n, l, :])
            Mest = MarkovDynamics(n_arms, n_states, all_learn_probs[n, l, :], t_type, t_increasing)
            SafeW = SafeWhittle(n_states, n_arms, tru_rew, Mest.transitions, n_steps, u_type, u_order, thresholds)
            SafeW.get_whittle_indices(computation_type=method, params=[0, max_wi], n_trials=n_trials_safety)
            sw_indices = SafeW.w_indices

            for a in range(n_arms):
                all_learn_sumwis[n, l, a] = np.round(np.sum(sw_indices[a]), 2)
                all_plan_rewards[n, l, a] = np.round(np.mean(plan_totalrewards[a, :]), 2)
                all_plan_objectives[n, l, a] = np.round(np.mean(plan_objectives[a, :]), 2)
                all_learn_rewards[n, l, a] = np.round(np.mean(learn_totalrewards[a, :]), 2)
                all_learn_objectives[n, l, a] = np.round(np.mean(learn_objectives[a, :]), 2)

            # print(f'pr = {np.round(np.mean(all_learn_probs[n, l, :]), 2)}')
            # print(f'true-sw = {np.round(np.mean(plan_sumwis), 2)}')
            # print(f'true-re = {np.round(np.mean(all_plan_rewards[n, l, :]), 2)}')
            # print(f'true-ob = {np.round(np.mean(all_plan_objectives[n, l, :]), 2)}')
            # print(f'sw = {np.round(np.mean(all_learn_sumwis[n, l, :]), 2)}')
            # print(f're = {np.round(np.mean(all_learn_rewards[n, l, :]), 2)}')
            # print(f'ob = {np.round(np.mean(all_learn_objectives[n, l, :]), 2)}')

        end_time = time.time()
        duration = end_time - start_time

    if save_data:
        joblib.dump([all_learn_probs, all_learn_sumwis, all_learn_rewards, all_learn_objectives, all_plan_rewards, all_plan_objectives],
                    f'./output/learnsoftsafetsrb_{n_steps}{n_states}{t_type}{u_type}{n_choices}{thresholds[0]}.joblib')

    return all_learn_probs, all_learn_sumwis, all_learn_rewards, all_learn_objectives, plan_sumwis, all_plan_rewards, all_plan_objectives


def Process_SingleSafeTSRB(n_iterations, l_episodes, n_episodes, n_steps, n_states, fixed_wi, n_choices, thresholds, t_type, t_increasing,
                           method, tru_rew, tru_dyn, initial_states, u_type, u_order, save_data, max_wi):
    n_trials_safety = n_states * n_steps

    ##################################################### Process
    all_rewards = np.zeros((n_iterations, l_episodes, 2))
    all_objectives = np.zeros((n_iterations, l_episodes, 2))
    all_sumwis = np.zeros((n_iterations, l_episodes))
    all_probs = np.round((0.1 / n_states), 2) * np.ones((n_iterations, l_episodes))
    duration = 0

    for n in range(n_iterations):

        # print(f'Learning iteration {n + 1} out of {n_iterations}')
        start_time = time.time()

        M = MarkovDynamics(1, n_states, [all_probs[n, 0]], t_type, t_increasing)
        SafeW = SafeWhittle(n_states, 1, tru_rew, M.transitions, n_steps, u_type, u_order, thresholds)
        SafeW.get_whittle_indices(computation_type=method, params=[0, max_wi], n_trials=n_trials_safety)
        sw_indices = SafeW.w_indices
        sw_indices.append(fixed_wi * np.ones_like(np.array(sw_indices[0])))

        counts = np.ones((n_states, n_states, 2))

        for l in range(l_episodes):

            print(f'Episode {l + 1} of {l_episodes} / Iteration {n} / last iteration was {np.round(duration, 1)} seconds')
            totalrewards = np.zeros((2, n_episodes))
            objectives = np.zeros((2, n_episodes))

            for k in range(n_episodes):

                states = initial_states.copy()
                _lifted = np.zeros(2, dtype=np.int32)
                for t in range(n_steps):
                    _states = np.copy(states)
                    _lifted[0] = max(0, min(SafeW.n_augment[0] - 1, _lifted[0] + _states[0]))
                    _lifted[1] = 0
                    actions = SafeW.Whittle_policy(sw_indices, n_choices, _states, _lifted, t)
                    for a in range(2):
                        if len(tru_rew.shape) == 3:
                            totalrewards[a, k] += tru_rew[_states[a], actions[a], a]
                        else:
                            totalrewards[a, k] += tru_rew[_states[a], a]
                        states[a] = np.random.choice(n_states, p=tru_dyn[_states[a], :, actions[a], a])
                        counts[_states[a], states[a], actions[a]] += 1
                for a in range(2):
                    if u_type == 1:
                        objectives[a, k] = 1 - thresholds[a] ** (- 1 / u_order) * (np.maximum(0, thresholds[a] - totalrewards[a, k])) ** (1 / u_order)
                    elif u_type == 2:
                        objectives[a, k] = (1 + np.exp(-u_order * (1 - thresholds[a]))) / (1 + np.exp(-u_order * (totalrewards[a, k] - thresholds[a])))
                    else:
                        objectives[a, k] = 1 if totalrewards[a, k] >= thresholds[a] else 0

            # print('Update...')
            est_transitions = np.zeros((n_states, n_states, 2))
            for s1 in range(n_states):
                for act in range(2):
                    est_transitions[s1, :, act] = dirichlet.rvs(counts[s1, :, act])
            if t_type == 5:
                cnt = [est_transitions[0, -1, 1]]
                for s1 in range(1, n_states):
                    cnt.append((1 / (s1 + 1)) * est_transitions[s1, -1, 1])
                    for s2 in range(1, s1):
                        cnt.append(est_transitions[s1, s2, 0])
                all_probs[n, l] = np.round(np.minimum(np.maximum(0.1 / n_states, np.mean(cnt)), 1 / n_states), 2)
            if t_type == 3:
                cnt = [(1 / (n_states - 1)) * est_transitions[0, 0, 1]]
                for s1 in range(1, n_states - 1):
                    cnt.append((1 / (n_states - s1 - 1)) * est_transitions[s1, s1, 1])
                    cnt.append((1 / (n_states - s1)) * est_transitions[s1, s1, 0])
                    for s2 in range(1, s1):
                        cnt.append(est_transitions[s1, s2, 0])
                for s2 in range(1, n_states):
                    cnt.append(est_transitions[n_states - 1, s2, 0])
                all_probs[n, l] = np.round(np.minimum(np.maximum(0.1 / n_states, np.mean(cnt)), 1 / n_states), 2)

            Mest = MarkovDynamics(1, n_states, [all_probs[n, l]], t_type, t_increasing)
            SafeW = SafeWhittle(n_states, 1, tru_rew, Mest.transitions, n_steps, u_type, u_order, thresholds)
            SafeW.get_whittle_indices(computation_type=method, params=[0, max_wi], n_trials=n_trials_safety)
            sw_indices = SafeW.w_indices
            all_sumwis[n, l] = np.sum(sw_indices)

            sw_indices.append(fixed_wi * np.ones_like(np.array(sw_indices[0])))

            for a in range(2):
                all_rewards[n, l, a] = np.mean(totalrewards[a, :])
                all_objectives[n, l, a] = np.mean(objectives[a, :])

            print(f'pr = {all_probs[n, l]}')
            print(f'sw = {all_sumwis[n, l]}')
            print(f're = {np.mean(all_rewards[n, l, :])}')
            print(f'ob = {np.mean(all_objectives[n, l, :])}')

        end_time = time.time()
        duration = end_time - start_time

    if save_data:
        joblib.dump([all_probs, all_sumwis, all_rewards, all_objectives], f'./output/singlesafetsrb_{n_steps}{n_states}{t_type}{u_type}{n_choices}{thresholds[0]}.joblib')

    return all_probs, all_sumwis, all_rewards, all_objectives



def Process_SingleSafeSoftTSRB(n_iterations, l_episodes, n_episodes, n_steps, n_states, fixed_wi, n_choices, thresholds, t_type, t_increasing,
                               method, tru_rew, tru_dyn, initial_states, u_type, u_order, save_data, max_wi):
    n_trials_safety = n_states * n_steps

    ##################################################### Process
    all_plan_rewards = np.zeros((n_iterations, l_episodes, 2))
    all_plan_objectives = np.zeros((n_iterations, l_episodes, 2))
    all_learn_rewards = np.zeros((n_iterations, l_episodes, 2))
    all_learn_objectives = np.zeros((n_iterations, l_episodes, 2))
    all_learn_sumwis = np.zeros((n_iterations, l_episodes))
    all_learn_probs = np.round((0.1 / n_states), 2) * np.ones((n_iterations, l_episodes))
    duration = 0

    PlanW = SafeWhittle(n_states, 1, tru_rew, tru_dyn, n_steps, u_type, u_order, thresholds)
    PlanW.get_whittle_indices(computation_type=method, params=[0, max_wi], n_trials=n_trials_safety)
    plan_indices = PlanW.w_indices
    plan_sumwis = np.round(np.sum(plan_indices), 2)
    plan_indices.append(fixed_wi * np.ones_like(np.array(plan_indices[0])))

    for n in range(n_iterations):

        # print(f'Learning iteration {n + 1} out of {n_iterations}')
        start_time = time.time()

        Mest = MarkovDynamics(1, n_states, [all_learn_probs[n, 0]], t_type, t_increasing)
        LearnW = SafeWhittle(n_states, 1, tru_rew, Mest.transitions, n_steps, u_type, u_order, thresholds)
        LearnW.get_whittle_indices(computation_type=method, params=[0, max_wi], n_trials=n_trials_safety)
        learn_indices = LearnW.w_indices
        # print(learn_indices)
        learn_indices.append(fixed_wi * np.ones_like(np.array(learn_indices[0])))

        counts = np.ones((n_states, n_states, 2))

        for l in range(l_episodes):

            print(f'Episode {l + 1} of {l_episodes} / Iteration {n} / last iteration was {np.round(duration, 1)} seconds')
            # totalrewards, objectives, cnts = Process_SingleSoftSafeRB(SafeW, n_episodes, n_steps, n_states, 2, n_choices, thresholds, tru_rew,
            #                                                           tru_dyn, sw_indices, initial_states, u_type, u_order)
            plan_totalrewards, plan_objectives, learn_totalrewards, learn_objectives, cnts = Process_LSSSRB(PlanW, plan_indices, LearnW, learn_indices, n_episodes, n_steps, n_states, 2,
                                                                                                            n_choices, thresholds, tru_rew, tru_dyn, initial_states, u_type, u_order)
            counts = counts + cnts[:, :, :, 0]

            # print('Update...')
            est_transitions = np.zeros((n_states, n_states, 2))
            for s1 in range(n_states):
                for act in range(2):
                    est_transitions[s1, :, act] = dirichlet.rvs(counts[s1, :, act])
            if t_type == 5:
                cnt = [est_transitions[0, -1, 1]]
                for s1 in range(1, n_states):
                    cnt.append((1 / (s1 + 1)) * est_transitions[s1, -1, 1])
                    for s2 in range(1, s1):
                        cnt.append(est_transitions[s1, s2, 0])
                all_learn_probs[n, l] = np.round(np.minimum(np.maximum(0.1 / n_states, np.mean(cnt)), 1 / n_states), 2)
            if t_type == 3:
                cnt = [(1 / (n_states - 1)) * est_transitions[0, 0, 1]]
                for s1 in range(1, n_states - 1):
                    cnt.append((1 / (n_states - s1 - 1)) * est_transitions[s1, s1, 1])
                    cnt.append((1 / (n_states - s1)) * est_transitions[s1, s1, 0])
                    for s2 in range(1, s1):
                        cnt.append(est_transitions[s1, s2, 0])
                for s2 in range(1, n_states):
                    cnt.append(est_transitions[n_states - 1, s2, 0])
                all_learn_probs[n, l] = np.round(np.minimum(np.maximum(0.1 / n_states, np.mean(cnt)), 1 / n_states), 2)

            Mest = MarkovDynamics(1, n_states, [all_learn_probs[n, l]], t_type, t_increasing)
            LearnW = SafeWhittle(n_states, 1, tru_rew, Mest.transitions, n_steps, u_type, u_order, thresholds)
            LearnW.get_whittle_indices(computation_type=method, params=[0, max_wi], n_trials=n_trials_safety)
            learn_indices = LearnW.w_indices
            # print(learn_indices)
            all_learn_sumwis[n, l] = np.round(np.sum(learn_indices), 2)
            learn_indices.append(fixed_wi * np.ones_like(np.array(learn_indices[0])))

            for a in range(2):
                all_plan_rewards[n, l, a] = np.round(np.mean(plan_totalrewards[a, :]), 2)
                all_plan_objectives[n, l, a] = np.round(np.mean(plan_objectives[a, :]), 2)
                all_learn_rewards[n, l, a] = np.round(np.mean(learn_totalrewards[a, :]), 2)
                all_learn_objectives[n, l, a] = np.round(np.mean(learn_objectives[a, :]), 2)

            print(f'pr = {all_learn_probs[n, l]}')
            print(f'sw = {all_learn_sumwis[n, l]}')
            print(f'true-re = {np.round(np.mean(all_plan_rewards[n, l, :]), 2)}')
            print(f'true-ob = {np.round(np.mean(all_plan_objectives[n, l, :]), 2)}')
            print(f're = {np.round(np.mean(all_learn_rewards[n, l, :]), 2)}')
            print(f'ob = {np.round(np.mean(all_learn_objectives[n, l, :]), 2)}')

        end_time = time.time()
        duration = end_time - start_time

    if save_data:
        joblib.dump([all_learn_probs, all_learn_sumwis, all_learn_rewards, all_learn_objectives, all_plan_rewards, all_plan_objectives],
                    f'./output/learnsinglesafesofttsrb_{n_steps}{n_states}{t_type}{u_type}{n_choices}{thresholds[0]}.joblib')

    return all_learn_probs, all_learn_sumwis, all_learn_rewards, all_learn_objectives, plan_sumwis, all_plan_rewards, all_plan_objectives
