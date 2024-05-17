from itertools import combinations
import random
import numpy as np


def Process_Random(n_episodes, n_steps, n_states, n_bandits, n_choices, thresholds, rewards, transitions, initial_states, u_type, u_order):

    ##################################################### Process
    totalrewards = np.zeros((n_bandits, n_episodes))
    objectives = np.zeros((n_bandits, n_episodes))
    counts = np.zeros((n_states, 2, n_bandits))
    for k in range(n_episodes):
        states = initial_states.copy()
        for t in range(n_steps):
            _states = np.copy(states)
            selected_indices = random.sample(range(n_bandits), n_choices)
            actions = [1 if i in selected_indices else 0 for i in range(n_bandits)]
            for a in range(n_bandits):
                if len(rewards.shape) == 3:
                    totalrewards[a, k] += rewards[_states[a], actions[a], a]
                else:
                    totalrewards[a, k] += rewards[_states[a], a]
                if u_type == 1:
                    objectives[a, k] = 1 - thresholds[a]**(1 - 1/u_order) * (np.maximum(0, thresholds[a] - totalrewards[a, k]))**(1/u_order)
                else:
                    objectives[a, k] = (1 + np.exp(-u_order * (1-thresholds[a]))) / (1 + np.exp(-u_order * (totalrewards[a, k]-thresholds[a])))
                counts[_states[a], actions[a], a] += 1
                # print(transitions[_states[a], :, actions[a], a])
                # print(sum(transitions[_states[a], :, actions[a], a]))
                states[a] = np.random.choice(n_states, p=transitions[_states[a], :, actions[a], a])

    return totalrewards, objectives, counts


def Process_Greedy(n_episodes, n_steps, n_states, n_bandits, n_choices, thresholds, rewards, transitions, initial_states, u_type, u_order):

    ##################################################### Process
    totalrewards = np.zeros((n_bandits, n_episodes))
    objectives = np.zeros((n_bandits, n_episodes))
    counts = np.zeros((n_states, 2, n_bandits))
    for k in range(n_episodes):
        states = initial_states.copy()
        for t in range(n_steps):
            rew_vec = np.zeros(n_bandits)
            for a in range(n_bandits):
                rew_vec[a] = rewards[states[a], a]
            _states = np.copy(states)
            top_indices = np.argsort(rew_vec)[-n_choices:]
            actions = np.zeros_like(rew_vec, dtype=np.int32)
            actions[top_indices] = 1
            for a in range(n_bandits):
                if len(rewards.shape) == 3:
                    totalrewards[a, k] += rewards[_states[a], actions[a], a]
                else:
                    totalrewards[a, k] += rewards[_states[a], a]
                if u_type == 1:
                    objectives[a, k] = 1 - thresholds[a]**(1 - 1/u_order) * (np.maximum(0, thresholds[a] - totalrewards[a, k]))**(1/u_order)
                else:
                    objectives[a, k] = (1 + np.exp(-u_order * (1-thresholds[a]))) / (1 + np.exp(-u_order * (totalrewards[a, k]-thresholds[a])))
                counts[_states[a], actions[a], a] += 1
                states[a] = np.random.choice(n_states, p=transitions[_states[a], :, actions[a], a])

    return totalrewards, objectives, counts


def Process_Myopic(n_episodes, n_steps, n_states, n_bandits, n_choices, thresholds, rewards, transitions, initial_states, u_type, u_order):

    ##################################################### Process
    totalrewards = np.zeros((n_bandits, n_episodes))
    objectives = np.zeros((n_bandits, n_episodes))
    counts = np.zeros((n_states, 2, n_bandits))
    for k in range(n_episodes):
        states = initial_states.copy()
        for t in range(n_steps):

            combs = list(combinations(range(n_bandits), n_choices))

            imm_rews = np.zeros(len(combs))
            for i, comb in enumerate(combs):
                rew = 0
                for a in range(n_bandits):
                    if a in comb:
                        rew += rewards[states[a], 1, a]
                    else:
                        rew += rewards[states[a], 0, a]
                imm_rews[i] = rew

            max_index = np.argmax(imm_rews)

            best_comb = combs[max_index]
            actions = np.array([(1 if i in best_comb else 0) for i in range(n_bandits)])
            _states = np.copy(states)
            for a in range(n_bandits):
                totalrewards[a, k] += rewards[_states[a], actions[a], a]
                if u_type == 1:
                    objectives[a, k] = 1 - thresholds[a]**(1 - 1/u_order) * (np.maximum(0, thresholds[a] - totalrewards[a, k]))**(1/u_order)
                else:
                    objectives[a, k] = (1 + np.exp(-u_order * (1-thresholds[a]))) / (1 + np.exp(-u_order * (totalrewards[a, k]-thresholds[a])))
                counts[_states[a], actions[a], a] += 1
                states[a] = np.random.choice(n_states, p=transitions[_states[a], :, actions[a], a])

    return totalrewards, objectives, counts


def Process_WhtlRB(W, n_episodes, n_steps, n_states, n_bandits, n_choices, thresholds, rewards, transitions, whittle_indices, initial_states, u_type, u_order):

    ##################################################### Process
    totalrewards = np.zeros((n_bandits, n_episodes))
    objectives = np.zeros((n_bandits, n_episodes))
    counts = np.zeros((n_states, 2, n_bandits))
    for k in range(n_episodes):
        states = initial_states.copy()
        for t in range(n_steps):
            _states = np.copy(states)
            actions = W.Whittle_policy(whittle_indices, n_choices, _states, t)
            for a in range(n_bandits):
                if len(rewards.shape) == 3:
                    totalrewards[a, k] += rewards[_states[a], actions[a], a]
                else:
                    totalrewards[a, k] += rewards[_states[a], a]
                if u_type == 1:
                    objectives[a, k] = 1 - thresholds[a]**(1 - 1/u_order) * (np.maximum(0, thresholds[a] - totalrewards[a, k]))**(1/u_order)
                else:
                    objectives[a, k] = (1 + np.exp(-u_order * (1-thresholds[a]))) / (1 + np.exp(-u_order * (totalrewards[a, k]-thresholds[a])))
                counts[_states[a], actions[a], a] += 1
                states[a] = np.random.choice(n_states, p=transitions[_states[a], :, actions[a], a])

    return totalrewards, objectives, counts


def Process_SafeRB(SafeW, n_episodes, n_steps, n_states, n_bandits, n_choices, thresholds, rewards, transitions, whittle_indices, initial_states, u_type, u_order):

    ##################################################### Process
    totalrewards = np.zeros((n_bandits, n_episodes))
    objectives = np.zeros((n_bandits, n_episodes))
    counts = np.zeros((n_states, 2, n_bandits))
    for k in range(n_episodes):
        states = initial_states.copy()
        _lifted = np.zeros(n_bandits, dtype=np.int32)
        for t in range(n_steps):
            _states = np.copy(states)
            for a in range(n_bandits):
                _lifted[a] = max(0, min(SafeW.n_augment[a]-1, _lifted[a] + _states[a]))
            actions = SafeW.Whittle_policy(whittle_indices, n_choices, _states, _lifted, t)
            for a in range(n_bandits):
                if len(rewards.shape) == 3:
                    totalrewards[a, k] += rewards[_states[a], actions[a], a]
                else:
                    totalrewards[a, k] += rewards[_states[a], a]
                if u_type == 1:
                    objectives[a, k] = 1 - thresholds[a]**(1 - 1/u_order) * (np.maximum(0, thresholds[a] - totalrewards[a, k]))**(1/u_order)
                else:
                    objectives[a, k] = (1 + np.exp(-u_order * (1-thresholds[a]))) / (1 + np.exp(-u_order * (totalrewards[a, k]-thresholds[a])))
                counts[_states[a], actions[a], a] += 1
                states[a] = np.random.choice(n_states, p=transitions[_states[a], :, actions[a], a])

    return totalrewards, objectives, counts


def Process_NeutRB(NeutW, n_episodes, n_steps, n_states, n_bandits, n_choices, thresholds, rewards, transitions, whittle_indices, initial_states, u_type, u_order):

    ##################################################### Process
    totalrewards = np.zeros((n_bandits, n_episodes))
    objectives = np.zeros((n_bandits, n_episodes))
    counts = np.zeros((n_states, 2, n_bandits))
    for k in range(n_episodes):
        states = initial_states.copy()
        lifted = np.zeros(n_bandits, dtype=np.int32)
        for t in range(n_steps):
            _states = np.copy(states)
            actions = NeutW.Whittle_policy(whittle_indices, n_choices, _states, lifted, t)
            for a in range(n_bandits):
                totalrewards[a, k] += rewards[_states[a], a]
                if u_type == 1:
                    objectives[a, k] = 1 - thresholds[a]**(1 - 1/u_order) * (np.maximum(0, thresholds[a] - totalrewards[a, k]))**(1/u_order)
                else:
                    objectives[a, k] = (1 + np.exp(-u_order * (1-thresholds[a]))) / (1 + np.exp(-u_order * (totalrewards[a, k]-thresholds[a])))
                counts[_states[a], actions[a], a] += 1
                states[a] = np.random.choice(n_states, p=transitions[_states[a], :, actions[a], a])

    return totalrewards, objectives, counts
