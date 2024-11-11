import random
from scipy.stats import dirichlet
import joblib
from Markov import *
from whittle import *
from processes import *
import time


def Process_LearnSafeTSRB(n_iterations, l_episodes, n_episodes, n_steps, n_states, n_arms, n_choices, thresholds,
                          t_type, t_increasing, method, tru_rew, tru_dyn, initial_states, u_type, u_order, save_data,
                          max_wi):
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
    plan_sumwis = [np.sum(plan_indices[a]) for a in range(n_arms)]

    for n in range(n_iterations):

        print(f'Learning iteration {n + 1} out of {n_iterations}')
        start_time = time.time()

        if t_type < 10:
            all_learn_probs[n, 0, :] = np.array([np.round(random.uniform(0.1 / n_states, 1 / n_states), 2)
                                                 for _ in range(n_arms)])
            Mest = MarkovDynamics(n_arms, n_states, all_learn_probs[n, 0, :], t_type, t_increasing)
            LearnW = SafeWhittle(n_states, n_arms, tru_rew, Mest.transitions, n_steps, u_type, u_order, thresholds)
        else:
            est_transitions = np.zeros((n_states, n_states, 2, n_arms))
            for a in range(n_arms):
                for s1 in range(n_states):
                    for act in range(2):
                        est_transitions[s1, :, act, a] = dirichlet.rvs(np.ones(n_states))
            LearnW = SafeWhittle(n_states, n_arms, tru_rew, est_transitions, n_steps, u_type, u_order, thresholds)

        LearnW.get_whittle_indices(computation_type=method, params=[0, max_wi], n_trials=n_trials_safety)
        learn_indices = LearnW.w_indices

        counts = np.ones((n_states, n_states, 2, n_arms))

        for l in range(l_episodes):
            plan_totalrewards, plan_objectives, learn_totalrewards, learn_objectives, cnts = \
                Process_LearnSafeRB(PlanW, plan_indices, LearnW, learn_indices, n_episodes, n_steps, n_states, n_arms,
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
                    all_learn_probs[n, l, a] = np.round(
                        np.minimum(np.maximum(0.1 / n_states, np.mean(cnt)), 1 / n_states)
                        , 2)
                if t_type == 3:
                    cnt = [(1 / (n_states - 1)) * est_transitions[0, 0, 1, a]]
                    for s1 in range(1, n_states - 1):
                        cnt.append((1 / (n_states - s1 - 1)) * est_transitions[s1, s1, 1, a])
                        cnt.append((1 / (n_states - s1)) * est_transitions[s1, s1, 0, a])
                        for s2 in range(1, s1):
                            cnt.append(est_transitions[s1, s2, 0, a])
                    for s2 in range(1, n_states):
                        cnt.append(est_transitions[n_states - 1, s2, 0, a])
                    all_learn_probs[n, l, a] = np.minimum(np.maximum(0.1 / n_states, np.mean(cnt)), 1 / n_states)

            if t_type < 10:
                Mest = MarkovDynamics(n_arms, n_states, np.round(all_learn_probs[n, l, :], 2), t_type, t_increasing)
                SafeW = SafeWhittle(n_states, n_arms, tru_rew, Mest.transitions, n_steps, u_type, u_order, thresholds)
                SafeW.get_whittle_indices(computation_type=method, params=[0, max_wi], n_trials=n_trials_safety)
                sw_indices = SafeW.w_indices
            else:
                SafeW = SafeWhittle(n_states, n_arms, tru_rew, est_transitions, n_steps, u_type, u_order, thresholds)
                SafeW.get_whittle_indices(computation_type=method, params=[0, max_wi], n_trials=n_trials_safety)
                sw_indices = SafeW.w_indices

            print(f"==================================== Episode {l}")
            for a in range(n_arms):
                if l > l_episodes - 3:
                    print(f'------------------ Arm {a}')
                    print(counts[:, :, 0, a])
                    print(counts[:, :, 1, a])
                all_learn_sumwis[n, l, a] = np.sum(sw_indices[a])
                all_plan_rewards[n, l, a] = np.round(np.mean(plan_totalrewards[a, :]), 2)
                all_plan_objectives[n, l, a] = np.round(np.mean(plan_objectives[a, :]), 2)
                all_learn_rewards[n, l, a] = np.round(np.mean(learn_totalrewards[a, :]), 2)
                all_learn_objectives[n, l, a] = np.round(np.mean(learn_objectives[a, :]), 2)

        end_time = time.time()
        duration = end_time - start_time

    if save_data:
        joblib.dump([all_learn_sumwis, all_learn_rewards, all_learn_objectives, plan_sumwis, all_plan_rewards,
                     all_plan_objectives],
                    f'./output-learn-finite/safetsrb_{n_episodes}{n_steps}{n_states}{t_type}{u_type}{n_choices}{thresholds[0]}{u_type}{u_order}.joblib')

    return all_learn_sumwis, all_learn_rewards, all_learn_objectives, plan_sumwis, all_plan_rewards, all_plan_objectives


def ProcessNS_LearnSafeTSRB(n_iterations, l_episodes, n_episodes, n_steps, n_states, n_arms, n_choices, thresholds,
                             t_type, t_increasing, method, tru_rew, tru_dyn, initial_states, u_type, u_order, save_data,
                             max_wi):
    n_trials_safety = n_states * n_steps

    ##################################################### Process
    all_plan_rewards = np.zeros((n_iterations, l_episodes, n_arms))
    all_plan_objectives = np.zeros((n_iterations, l_episodes, n_arms))
    all_learn_rewards = np.zeros((n_iterations, l_episodes, n_arms))
    all_learn_objectives = np.zeros((n_iterations, l_episodes, n_arms))
    all_learn_sumwis = np.zeros((n_iterations, l_episodes, n_arms))
    all_learn_probs = np.ones((n_iterations, l_episodes, n_arms, n_steps))
    duration = 0

    PlanW = SafeWhittleNS(n_states, n_arms, tru_rew, tru_dyn, n_steps, u_type, u_order, thresholds)
    PlanW.get_whittle_indices(computation_type=method, params=[0, max_wi], n_trials=n_trials_safety)
    plan_indices = PlanW.w_indices
    plan_sumwis = [np.sum(plan_indices[a]) for a in range(n_arms)]

    for n in range(n_iterations):

        print(f'Learning iteration {n + 1} out of {n_iterations}')
        start_time = time.time()

        if t_type < 10:
            all_learn_probs[n, 0, :, :] = np.array([[np.round(random.uniform(0.1 / n_states, 1 / n_states), 2) for _ in range(n_steps)] for _ in range(n_arms)])
            Mest = MarkovDynamicsNS(n_steps, n_arms, n_states, all_learn_probs[n, 0, :, :], t_type, t_increasing)
            LearnW = SafeWhittleNS(n_states, n_arms, tru_rew, Mest.transitions, n_steps, u_type, u_order, thresholds)
        else:
            est_transitions = np.zeros((n_states, n_states, 2, n_arms, n_steps))
            for a in range(n_arms):
                for s1 in range(n_states):
                    for act in range(2):
                        for t in range(n_steps):
                            est_transitions[s1, :, act, a, t] = dirichlet.rvs(np.ones(n_states))
            LearnW = SafeWhittleNS(n_states, n_arms, tru_rew, est_transitions, n_steps, u_type, u_order, thresholds)

        LearnW.get_whittle_indices(computation_type=method, params=[0, max_wi], n_trials=n_trials_safety)
        learn_indices = LearnW.w_indices

        counts = np.ones((n_states, n_states, 2, n_arms, n_steps))

        for l in range(l_episodes):
            plan_totalrewards, plan_objectives, learn_totalrewards, learn_objectives, cnts = \
                Process_LearnSafeRB(PlanW, plan_indices, LearnW, learn_indices, n_episodes, n_steps, n_states, n_arms,
                                    n_choices, thresholds, tru_rew, tru_dyn, initial_states, u_type, u_order)
            counts = counts + cnts

            # print('Update...')
            est_transitions = np.zeros((n_states, n_states, 2, n_arms, n_steps))
            for t in range(n_steps):
                for a in range(n_arms):
                    for s1 in range(n_states):
                        for act in range(2):
                            for t in range(n_steps):
                                est_transitions[s1, :, act, a, t] = dirichlet.rvs(counts[s1, :, act, a, t])
                    if t_type == 5:
                        cnt = [est_transitions[0, -1, 1, a, t]]
                        for s1 in range(1, n_states):
                            cnt.append((1 / (s1 + 1)) * est_transitions[s1, -1, 1, a, t])
                            for s2 in range(1, s1):
                                cnt.append(est_transitions[s1, s2, 0, a, t])
                        all_learn_probs[n, l, a, t] = np.round(np.minimum(np.maximum(0.1 / n_states, np.mean(cnt)), 1 / n_states), 2)
                    if t_type == 3:
                        cnt = [(1 / (n_states - 1)) * est_transitions[0, 0, 1, a, t]]
                        for s1 in range(1, n_states - 1):
                            cnt.append((1 / (n_states - s1 - 1)) * est_transitions[s1, s1, 1, a, t])
                            cnt.append((1 / (n_states - s1)) * est_transitions[s1, s1, 0, a, t])
                            for s2 in range(1, s1):
                                cnt.append(est_transitions[s1, s2, 0, a, t])
                        for s2 in range(1, n_states):
                            cnt.append(est_transitions[n_states - 1, s2, 0, a, t])
                        all_learn_probs[n, l, a, t] = np.minimum(np.maximum(0.1 / n_states, np.mean(cnt)), 1 / n_states)

            if t_type < 10:
                Mest = MarkovDynamicsNS(n_steps, n_arms, n_states, np.round(all_learn_probs[n, l, :, :], 2), t_type, t_increasing)
                SafeW = SafeWhittleNS(n_states, n_arms, tru_rew, Mest.transitions, n_steps, u_type, u_order, thresholds)
                SafeW.get_whittle_indices(computation_type=method, params=[0, max_wi], n_trials=n_trials_safety)
                sw_indices = SafeW.w_indices
            else:
                SafeW = SafeWhittleNS(n_states, n_arms, tru_rew, est_transitions, n_steps, u_type, u_order, thresholds)
                SafeW.get_whittle_indices(computation_type=method, params=[0, max_wi], n_trials=n_trials_safety)
                sw_indices = SafeW.w_indices

            print(f"==================================== Episode {l}")
            for a in range(n_arms):
                if l > l_episodes - 3:
                    print(f'------------------ Arm {a}')
                    for t in range(n_steps):
                        print(f'------------------ Time {t}')
                        print(counts[:, :, 0, a, t])
                        print(counts[:, :, 1, a, t])
                all_learn_sumwis[n, l, a] = np.sum(sw_indices[a])
                all_plan_rewards[n, l, a] = np.round(np.mean(plan_totalrewards[a, :]), 2)
                all_plan_objectives[n, l, a] = np.round(np.mean(plan_objectives[a, :]), 2)
                all_learn_rewards[n, l, a] = np.round(np.mean(learn_totalrewards[a, :]), 2)
                all_learn_objectives[n, l, a] = np.round(np.mean(learn_objectives[a, :]), 2)

        end_time = time.time()
        duration = end_time - start_time

    if save_data:
        joblib.dump([all_learn_sumwis, all_learn_rewards, all_learn_objectives, plan_sumwis, all_plan_rewards,
                     all_plan_objectives],
                    f'./output-learn-finite/safetsrb_{n_episodes}{n_steps}{n_states}{t_type}{u_type}{n_choices}{thresholds[0]}{u_type}{u_order}.joblib')

    return all_learn_sumwis, all_learn_rewards, all_learn_objectives, plan_sumwis, all_plan_rewards, all_plan_objectives


def ProcessDNS_LearnSafeTSRB(n_iterations, l_episodes, n_episodes, n_steps, n_states, n_parts, n_arms, n_choices, thresholds,
                            t_type, t_increasing, method, tru_rew, tru_dyn, initial_states, u_type, u_order, save_data,
                            max_wi):
    n_trials_safety = n_states * n_steps

    ##################################################### Process
    all_plan_rewards = np.zeros((n_iterations, l_episodes, n_arms))
    all_plan_objectives = np.zeros((n_iterations, l_episodes, n_arms))
    all_learn_rewards = np.zeros((n_iterations, l_episodes, n_arms))
    all_learn_objectives = np.zeros((n_iterations, l_episodes, n_arms))
    all_learn_sumwis = np.zeros((n_iterations, l_episodes, n_arms))
    all_learn_probs = np.ones((n_iterations, l_episodes, n_arms, n_steps))
    duration = 0

    PlanW = SafeWhittleDNS([n_states, n_parts], n_arms, tru_rew, tru_dyn, n_steps, u_type, u_order, thresholds)
    PlanW.get_whittle_indices(computation_type=method, params=[0, max_wi], n_trials=n_trials_safety)
    plan_indices = PlanW.w_indices
    plan_sumwis = [np.sum(plan_indices[a]) for a in range(n_arms)]

    for n in range(n_iterations):

        print(f'Learning iteration {n + 1} out of {n_iterations}')
        start_time = time.time()

        if t_type < 10:
            all_learn_probs[n, 0, :, :] = np.array([[np.round(random.uniform(0.1 / n_states, 1 / n_states), 2) for _ in range(n_steps)] for _ in range(n_arms)])
            Mest = MarkovDynamicsNS(n_steps, n_arms, n_states, all_learn_probs[n, 0, :, :], t_type, t_increasing)
            LearnW = SafeWhittleDNS([n_states, n_parts], n_arms, tru_rew, Mest.transitions, n_steps, u_type, u_order, thresholds)
        else:
            est_transitions = np.zeros((n_states, n_states, 2, n_arms, n_steps))
            for a in range(n_arms):
                for s1 in range(n_states):
                    for act in range(2):
                        for t in range(n_steps):
                            est_transitions[s1, :, act, a, t] = dirichlet.rvs(np.ones(n_states))
            LearnW = SafeWhittleDNS([n_states, n_parts], n_arms, tru_rew, est_transitions, n_steps, u_type, u_order, thresholds)

        LearnW.get_whittle_indices(computation_type=method, params=[0, max_wi], n_trials=n_trials_safety)
        learn_indices = LearnW.w_indices

        counts = np.ones((n_states, n_states, 2, n_arms, n_steps))

        for l in range(l_episodes):
            plan_totalrewards, plan_objectives, learn_totalrewards, learn_objectives, cnts = \
                Process_LearnSafeRB(PlanW, plan_indices, LearnW, learn_indices, n_episodes, n_steps, n_states, n_arms,
                                    n_choices, thresholds, tru_rew, tru_dyn, initial_states, u_type, u_order)
            counts = counts + cnts

            # print('Update...')
            est_transitions = np.zeros((n_states, n_states, 2, n_arms, n_steps))
            for t in range(n_steps):
                for a in range(n_arms):
                    for s1 in range(n_states):
                        for act in range(2):
                            for t in range(n_steps):
                                est_transitions[s1, :, act, a, t] = dirichlet.rvs(counts[s1, :, act, a, t])
                    if t_type == 5:
                        cnt = [est_transitions[0, -1, 1, a, t]]
                        for s1 in range(1, n_states):
                            cnt.append((1 / (s1 + 1)) * est_transitions[s1, -1, 1, a, t])
                            for s2 in range(1, s1):
                                cnt.append(est_transitions[s1, s2, 0, a, t])
                        all_learn_probs[n, l, a, t] = np.round(np.minimum(np.maximum(0.1 / n_states, np.mean(cnt)), 1 / n_states), 2)
                    if t_type == 3:
                        cnt = [(1 / (n_states - 1)) * est_transitions[0, 0, 1, a, t]]
                        for s1 in range(1, n_states - 1):
                            cnt.append((1 / (n_states - s1 - 1)) * est_transitions[s1, s1, 1, a, t])
                            cnt.append((1 / (n_states - s1)) * est_transitions[s1, s1, 0, a, t])
                            for s2 in range(1, s1):
                                cnt.append(est_transitions[s1, s2, 0, a, t])
                        for s2 in range(1, n_states):
                            cnt.append(est_transitions[n_states - 1, s2, 0, a, t])
                        all_learn_probs[n, l, a, t] = np.minimum(np.maximum(0.1 / n_states, np.mean(cnt)), 1 / n_states)

            if t_type < 10:
                Mest = MarkovDynamicsNS(n_steps, n_arms, n_states, np.round(all_learn_probs[n, l, :, :], 2), t_type, t_increasing)
                SafeW = SafeWhittleDNS([n_states, n_parts], n_arms, tru_rew, Mest.transitions, n_steps, u_type, u_order, thresholds)
                SafeW.get_whittle_indices(computation_type=method, params=[0, max_wi], n_trials=n_trials_safety)
                sw_indices = SafeW.w_indices
            else:
                SafeW = SafeWhittleDNS([n_states, n_parts], n_arms, tru_rew, est_transitions, n_steps, u_type, u_order, thresholds)
                SafeW.get_whittle_indices(computation_type=method, params=[0, max_wi], n_trials=n_trials_safety)
                sw_indices = SafeW.w_indices

            print(f"==================================== Episode {l}")
            for a in range(n_arms):
                if l > l_episodes - 3:
                    print(f'------------------ Arm {a}')
                    for t in range(n_steps):
                        print(f'------------------ Time {t}')
                        print(counts[:, :, 0, a, t])
                        print(counts[:, :, 1, a, t])
                all_learn_sumwis[n, l, a] = np.sum(sw_indices[a])
                all_plan_rewards[n, l, a] = np.round(np.mean(plan_totalrewards[a, :]), 2)
                all_plan_objectives[n, l, a] = np.round(np.mean(plan_objectives[a, :]), 2)
                all_learn_rewards[n, l, a] = np.round(np.mean(learn_totalrewards[a, :]), 2)
                all_learn_objectives[n, l, a] = np.round(np.mean(learn_objectives[a, :]), 2)

        end_time = time.time()
        duration = end_time - start_time

    if save_data:
        joblib.dump([all_learn_sumwis, all_learn_rewards, all_learn_objectives, plan_sumwis, all_plan_rewards,
                     all_plan_objectives],
                    f'./output-learn-finite/safetsrb_{n_episodes}{n_steps}{n_states}{t_type}{u_type}{n_choices}{thresholds[0]}{u_type}{u_order}.joblib')

    return all_learn_sumwis, all_learn_rewards, all_learn_objectives, plan_sumwis, all_plan_rewards, all_plan_objectives


def ProcessDisT_LearnSafeTSRB(beta, n_iterations, l_episodes, n_episodes, n_steps, n_states, n_parts, n_arms, n_choices,
                             thresholds, t_type, t_increasing, method, tru_rew, tru_dyn, initial_states, u_type,
                             u_order, save_data):

    duration = 0

    ##################################################### Process
    all_plan_rewards = np.zeros((n_iterations, l_episodes, n_arms))
    all_plan_objectives = np.zeros((n_iterations, l_episodes, n_arms))
    all_learn_rewards = np.zeros((n_iterations, l_episodes, n_arms))
    all_learn_objectives = np.zeros((n_iterations, l_episodes, n_arms))
    all_learn_sumwis = np.zeros((n_iterations, l_episodes, n_arms))
    all_learn_probs = np.ones((n_iterations, l_episodes, n_arms))
    all_learn_transitionerrors = np.zeros((n_iterations, l_episodes, n_arms))

    PlanW = SafeWhittleDisT(beta, [n_states, n_parts], n_arms, tru_rew, tru_dyn, n_steps, u_type, u_order, thresholds)
    PlanW.get_whittle_indices(computation_type=method, params=[0, 10], n_trials=100)
    plan_indices = PlanW.w_indices
    plan_sumwis = [np.sum(plan_indices[a]) for a in range(n_arms)]

    for n in range(n_iterations):

        print(f'Learning iteration {n + 1} out of {n_iterations}')
        start_time = time.time()

        if t_type < 10:
            all_learn_probs[n, 0, :] = np.array([np.round(random.uniform(0.1 / n_states, 1 / n_states), 2)
                                                 for _ in range(n_arms)])
            Mest = MarkovDynamics(n_arms, n_states, all_learn_probs[n, 0, :], t_type, t_increasing)
            LearnW = SafeWhittleDisT(beta, [n_states, n_parts], n_arms, tru_rew, Mest.transitions, n_steps, u_type, u_order,
                                     thresholds)
        else:
            est_transitions = np.zeros((n_states, n_states, 2, n_arms))
            for a in range(n_arms):
                for s1 in range(n_states):
                    for act in range(2):
                        est_transitions[s1, :, act, a] = dirichlet.rvs(np.ones(n_states))
            LearnW = SafeWhittleDisT(beta, [n_states, n_parts], n_arms, tru_rew, est_transitions, n_steps, u_type, u_order,
                                     thresholds)

        LearnW.get_whittle_indices(computation_type=method, params=[0, 10], n_trials=100)
        learn_indices = LearnW.w_indices

        counts = np.ones((n_states, n_states, 2, n_arms))

        for l in range(l_episodes):

            plan_totalrewards, plan_objectives, learn_totalrewards, learn_objectives, cnts = \
                ProcessDisT_LearnSafeRB(PlanW, plan_indices, LearnW, learn_indices, beta, n_episodes, n_steps, n_states,
                                        n_arms, n_choices, thresholds, tru_rew, tru_dyn, initial_states, u_type, u_order)
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
                    all_learn_probs[n, l, a] = np.round(
                        np.minimum(np.maximum(0.1 / n_states, np.mean(cnt)), 1 / n_states)
                        , 2)
                if t_type == 3:
                    cnt = [(1 / (n_states - 1)) * est_transitions[0, 0, 1, a]]
                    for s1 in range(1, n_states - 1):
                        cnt.append((1 / (n_states - s1 - 1)) * est_transitions[s1, s1, 1, a])
                        cnt.append((1 / (n_states - s1)) * est_transitions[s1, s1, 0, a])
                        for s2 in range(1, s1):
                            cnt.append(est_transitions[s1, s2, 0, a])
                    for s2 in range(1, n_states):
                        cnt.append(est_transitions[n_states - 1, s2, 0, a])
                    all_learn_probs[n, l, a] = np.minimum(np.maximum(0.1 / n_states, np.mean(cnt)), 1 / n_states)

            if t_type < 10:
                Mest = MarkovDynamics(n_arms, n_states, np.round(all_learn_probs[n, l, :], 2), t_type, t_increasing)
                SafeW = SafeWhittleDisT(beta, [n_states, n_parts], n_arms, tru_rew, Mest.transitions, n_steps, u_type, u_order,
                                        thresholds)
                SafeW.get_whittle_indices(computation_type=method, params=[0, 10], n_trials=100)
                sw_indices = SafeW.w_indices
            else:
                SafeW = SafeWhittleDisT(beta, [n_states, n_parts], n_arms, tru_rew, est_transitions, n_steps, u_type, u_order,
                                        thresholds)
                SafeW.get_whittle_indices(computation_type=method, params=[0, 10], n_trials=100)
                sw_indices = SafeW.w_indices

            for a in range(n_arms):
                all_learn_transitionerrors[n, l, a] = np.max(np.abs(est_transitions[:, :, :, a] - tru_dyn[:, :, :, a]))
                all_learn_sumwis[n, l, a] = np.sum(sw_indices[a])
                all_plan_rewards[n, l, a] = np.round(np.mean(plan_totalrewards[a, :]), 2)
                all_plan_objectives[n, l, a] = np.round(np.mean(plan_objectives[a, :]), 2)
                all_learn_rewards[n, l, a] = np.round(np.mean(learn_totalrewards[a, :]), 2)
                all_learn_objectives[n, l, a] = np.round(np.mean(learn_objectives[a, :]), 2)

        end_time = time.time()
        duration = end_time - start_time

    if save_data:
        joblib.dump([all_learn_sumwis, all_learn_rewards, all_learn_objectives, plan_sumwis, all_plan_rewards,
                     all_plan_objectives],
                    f'./output-learn-dis-finite/safetsrb_{beta}{n_episodes}{n_steps}{n_states}{t_type}{u_type}{n_choices}{thresholds[0]}{u_type}{u_order}.joblib')

    return all_learn_transitionerrors, all_learn_sumwis, all_learn_rewards, all_learn_objectives, plan_sumwis, all_plan_rewards, all_plan_objectives


def ProcessDis_LearnSafeTSRB(beta, n_iterations, l_episodes, n_episodes, n_steps, n_states, n_parts, n_arms, n_choices,
                             thresholds, t_type, t_increasing, method, tru_rew, tru_dyn, initial_states, u_type,
                             u_order, save_data):

    duration = 0

    ##################################################### Process
    all_plan_rewards = np.zeros((n_iterations, l_episodes, n_arms))
    all_plan_objectives = np.zeros((n_iterations, l_episodes, n_arms))
    all_learn_rewards = np.zeros((n_iterations, l_episodes, n_arms))
    all_learn_objectives = np.zeros((n_iterations, l_episodes, n_arms))
    all_learn_sumwis = np.zeros((n_iterations, l_episodes, n_arms))
    all_learn_probs = np.ones((n_iterations, l_episodes, n_arms))
    all_learn_transitionerrors = np.zeros((n_iterations, l_episodes, n_arms))

    PlanW = SafeWhittleDis(beta, [n_states, n_parts], n_arms, tru_rew, tru_dyn, n_steps, u_type, u_order, thresholds)
    PlanW.get_whittle_indices(computation_type=method, params=[0, 10], n_trials=100)
    plan_indices = PlanW.w_indices
    plan_sumwis = [np.sum(plan_indices[a]) for a in range(n_arms)]

    for n in range(n_iterations):

        print(f'Learning iteration {n + 1} out of {n_iterations}')
        start_time = time.time()

        if t_type < 10:
            all_learn_probs[n, 0, :] = np.array([np.round(random.uniform(0.1 / n_states, 1 / n_states), 2)
                                                 for _ in range(n_arms)])
            Mest = MarkovDynamics(n_arms, n_states, all_learn_probs[n, 0, :], t_type, t_increasing)
            LearnW = SafeWhittleDis(beta, [n_states, n_parts], n_arms, tru_rew, Mest.transitions, n_steps, u_type, u_order,
                                    thresholds)
        else:
            est_transitions = np.zeros((n_states, n_states, 2, n_arms))
            for a in range(n_arms):
                for s1 in range(n_states):
                    for act in range(2):
                        est_transitions[s1, :, act, a] = dirichlet.rvs(np.ones(n_states))
            LearnW = SafeWhittleDis(beta, [n_states, n_parts], n_arms, tru_rew, est_transitions, n_steps, u_type, u_order,
                                    thresholds)

        LearnW.get_whittle_indices(computation_type=method, params=[0, 10], n_trials=100)
        learn_indices = LearnW.w_indices

        counts = np.ones((n_states, n_states, 2, n_arms))

        for l in range(l_episodes):

            plan_totalrewards, plan_objectives, learn_totalrewards, learn_objectives, cnts = \
                ProcessDis_LearnSafeRB(PlanW, plan_indices, LearnW, learn_indices, beta, n_episodes, n_steps, n_states,
                                       n_arms, n_choices, thresholds, tru_rew, tru_dyn, initial_states, u_type, u_order)
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
                    all_learn_probs[n, l, a] = np.round(
                        np.minimum(np.maximum(0.1 / n_states, np.mean(cnt)), 1 / n_states)
                        , 2)
                if t_type == 3:
                    cnt = [(1 / (n_states - 1)) * est_transitions[0, 0, 1, a]]
                    for s1 in range(1, n_states - 1):
                        cnt.append((1 / (n_states - s1 - 1)) * est_transitions[s1, s1, 1, a])
                        cnt.append((1 / (n_states - s1)) * est_transitions[s1, s1, 0, a])
                        for s2 in range(1, s1):
                            cnt.append(est_transitions[s1, s2, 0, a])
                    for s2 in range(1, n_states):
                        cnt.append(est_transitions[n_states - 1, s2, 0, a])
                    all_learn_probs[n, l, a] = np.minimum(np.maximum(0.1 / n_states, np.mean(cnt)), 1 / n_states)

            if t_type < 10:
                Mest = MarkovDynamics(n_arms, n_states, np.round(all_learn_probs[n, l, :], 2), t_type, t_increasing)
                SafeW = SafeWhittleDis(beta, [n_states, n_parts], n_arms, tru_rew, Mest.transitions, n_steps, u_type, u_order,
                                       thresholds)
                SafeW.get_whittle_indices(computation_type=method, params=[0, 10], n_trials=100)
                sw_indices = SafeW.w_indices
            else:
                SafeW = SafeWhittleDis(beta, [n_states, n_parts], n_arms, tru_rew, est_transitions, n_steps, u_type, u_order,
                                       thresholds)
                SafeW.get_whittle_indices(computation_type=method, params=[0, 10], n_trials=100)
                sw_indices = SafeW.w_indices

            for a in range(n_arms):
                all_learn_transitionerrors[n, l, a] = np.max(np.abs(est_transitions[:, :, :, a] - tru_dyn[:, :, :, a]))
                all_learn_sumwis[n, l, a] = np.sum(sw_indices[a])
                all_plan_rewards[n, l, a] = np.round(np.mean(plan_totalrewards[a, :]), 2)
                all_plan_objectives[n, l, a] = np.round(np.mean(plan_objectives[a, :]), 2)
                all_learn_rewards[n, l, a] = np.round(np.mean(learn_totalrewards[a, :]), 2)
                all_learn_objectives[n, l, a] = np.round(np.mean(learn_objectives[a, :]), 2)

        end_time = time.time()
        duration = end_time - start_time

    if save_data:
        joblib.dump([all_learn_sumwis, all_learn_rewards, all_learn_objectives, plan_sumwis, all_plan_rewards,
                     all_plan_objectives],
                    f'./output-learn-dis/safetsrb_{beta}{n_episodes}{n_steps}{n_states}{t_type}{u_type}{n_choices}{thresholds[0]}{u_type}{u_order}.joblib')

    return all_learn_transitionerrors, all_learn_sumwis, all_learn_rewards, all_learn_objectives, plan_sumwis, all_plan_rewards, all_plan_objectives
