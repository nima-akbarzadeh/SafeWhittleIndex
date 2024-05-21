from itertools import combinations
import random
import numpy as np


def Process_Random(n_episodes, n_steps, n_states, n_bandits, n_choices, thresholds, rewards, transitions, initial_states, u_type, u_order):

    ##################################################### Process
    totalrewards = np.zeros((n_bandits, n_episodes))
    objectives = np.zeros((n_bandits, n_episodes))
    counts = np.zeros((n_states, n_states, 2, n_bandits))
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
                states[a] = np.random.choice(n_states, p=transitions[_states[a], :, actions[a], a])
                counts[_states[a], states[a], actions[a], a] += 1
        for a in range(n_bandits):
            if u_type == 1:
                objectives[a, k] = 1 - thresholds[a] ** (- 1 / u_order) * (np.maximum(0, thresholds[a] - totalrewards[a, k])) ** (1 / u_order)
            elif u_type == 2:
                objectives[a, k] = (1 + np.exp(-u_order * (1 - thresholds[a]))) / (1 + np.exp(-u_order * (totalrewards[a, k] - thresholds[a])))
            else:
                objectives[a, k] = 1 if totalrewards[a, k] >= thresholds[a] else 0

    return np.around(totalrewards, 2), np.around(objectives, 2), counts


def Process_Greedy(n_episodes, n_steps, n_states, n_bandits, n_choices, thresholds, rewards, transitions, initial_states, u_type, u_order):

    ##################################################### Process
    totalrewards = np.zeros((n_bandits, n_episodes))
    objectives = np.zeros((n_bandits, n_episodes))
    counts = np.zeros((n_states, n_states, 2, n_bandits))
    for k in range(n_episodes):
        states = initial_states.copy()
        for t in range(n_steps):
            rew_vec = np.zeros(n_bandits)
            for a2 in range(n_bandits):
                rew_vec[a2] = rewards[states[a2], a2]
            _states = np.copy(states)
            top_indices = np.argsort(rew_vec)[-n_choices:]
            actions = np.zeros_like(rew_vec, dtype=np.int32)
            actions[top_indices] = 1
            for a in range(n_bandits):
                if len(rewards.shape) == 3:
                    totalrewards[a, k] += rewards[_states[a], actions[a], a]
                else:
                    totalrewards[a, k] += rewards[_states[a], a]
                states[a] = np.random.choice(n_states, p=transitions[_states[a], :, actions[a], a])
                counts[_states[a], states[a], actions[a], a] += 1
        for a in range(n_bandits):
            if u_type == 1:
                objectives[a, k] = 1 - thresholds[a]**(- 1/u_order) * (np.maximum(0, thresholds[a] - totalrewards[a, k]))**(1/u_order)
            elif u_type == 2:
                objectives[a, k] = (1 + np.exp(-u_order * (1-thresholds[a]))) / (1 + np.exp(-u_order * (totalrewards[a, k]-thresholds[a])))
            else:
                objectives[a, k] = 1 if totalrewards[a, k] >= thresholds[a] else 0

    return np.around(totalrewards, 2), np.around(objectives, 2), counts


def Process_Myopic(n_episodes, n_steps, n_states, n_bandits, n_choices, thresholds, rewards, transitions, initial_states, u_type, u_order):

    ##################################################### Process
    totalrewards = np.zeros((n_bandits, n_episodes))
    objectives = np.zeros((n_bandits, n_episodes))
    counts = np.zeros((n_states, n_states, 2, n_bandits))
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
                states[a] = np.random.choice(n_states, p=transitions[_states[a], :, actions[a], a])
                counts[_states[a], states[a], actions[a], a] += 1
        for a in range(n_bandits):
            if u_type == 1:
                objectives[a, k] = 1 - thresholds[a]**(- 1/u_order) * (np.maximum(0, thresholds[a] - totalrewards[a, k]))**(1/u_order)
            elif u_type == 2:
                objectives[a, k] = (1 + np.exp(-u_order * (1-thresholds[a]))) / (1 + np.exp(-u_order * (totalrewards[a, k]-thresholds[a])))
            else:
                objectives[a, k] = 1 if totalrewards[a, k] >= thresholds[a] else 0

    return np.around(totalrewards, 2), np.around(objectives, 2), counts


def Process_WhtlRB(W, whittle_indices, n_episodes, n_steps, n_states, n_bandits, n_choices, thresholds, rewards, transitions, initial_states, u_type, u_order):

    ##################################################### Process
    totalrewards = np.zeros((n_bandits, n_episodes))
    objectives = np.zeros((n_bandits, n_episodes))
    counts = np.zeros((n_states, n_states, 2, n_bandits))
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
                states[a] = np.random.choice(n_states, p=transitions[_states[a], :, actions[a], a])
                counts[_states[a], states[a], actions[a], a] += 1
        for a in range(n_bandits):
            if u_type == 1:
                objectives[a, k] = 1 - thresholds[a]**(- 1/u_order) * (np.maximum(0, thresholds[a] - totalrewards[a, k]))**(1/u_order)
            elif u_type == 2:
                objectives[a, k] = (1 + np.exp(-u_order * (1-thresholds[a]))) / (1 + np.exp(-u_order * (totalrewards[a, k]-thresholds[a])))
            else:
                objectives[a, k] = 1 if totalrewards[a, k] >= thresholds[a] else 0

    return np.around(totalrewards, 2), np.around(objectives, 2), counts


def Process_SafeRB(SafeW, whittle_indices, n_episodes, n_steps, n_states, n_bandits, n_choices, thresholds, rewards, transitions, initial_states, u_type, u_order):

    ##################################################### Process
    totalrewards = np.zeros((n_bandits, n_episodes))
    objectives = np.zeros((n_bandits, n_episodes))
    counts = np.zeros((n_states, n_states, 2, n_bandits))
    for k in range(n_episodes):
        states = initial_states.copy()
        _lifted = np.zeros(n_bandits, dtype=np.int32)
        for t in range(n_steps):
            _states = np.copy(states)
            for a2 in range(n_bandits):
                _lifted[a2] = max(0, min(SafeW.n_augment[a2]-1, _lifted[a2] + _states[a2]))
            actions = SafeW.Whittle_policy(whittle_indices, n_choices, _states, _lifted, t)
            for a in range(n_bandits):
                if len(rewards.shape) == 3:
                    totalrewards[a, k] += rewards[_states[a], actions[a], a]
                else:
                    totalrewards[a, k] += rewards[_states[a], a]
                states[a] = np.random.choice(n_states, p=transitions[_states[a], :, actions[a], a])
                counts[_states[a], states[a], actions[a], a] += 1
        for a in range(n_bandits):
            if u_type == 1:
                objectives[a, k] = 1 - thresholds[a]**(- 1/u_order) * (np.maximum(0, thresholds[a] - totalrewards[a, k]))**(1/u_order)
            elif u_type == 2:
                objectives[a, k] = (1 + np.exp(-u_order * (1-thresholds[a]))) / (1 + np.exp(-u_order * (totalrewards[a, k]-thresholds[a])))
            else:
                objectives[a, k] = 1 if totalrewards[a, k] >= thresholds[a] else 0

    return np.around(totalrewards, 2), np.around(objectives, 2), counts


def Process_SoftSafeRB(SafeW, whittle_indices, n_episodes, n_steps, n_states, n_bandits, n_choices, thresholds, rewards, transitions, initial_states, u_type, u_order):

    ##################################################### Process
    totalrewards = np.zeros((n_bandits, n_episodes))
    objectives = np.zeros((n_bandits, n_episodes))
    counts = np.zeros((n_states, n_states, 2, n_bandits))
    for k in range(n_episodes):
        states = initial_states.copy()
        _lifted = np.zeros(n_bandits, dtype=np.int32)
        for t in range(n_steps):
            _states = np.copy(states)
            for a in range(n_bandits):
                _lifted[a] = max(0, min(SafeW.n_augment[a]-1, _lifted[a] + _states[a]))
            actions = SafeW.Whittle_softpolicy(whittle_indices, n_choices, _states, _lifted, t)
            for a in range(n_bandits):
                if len(rewards.shape) == 3:
                    totalrewards[a, k] += rewards[_states[a], actions[a], a]
                else:
                    totalrewards[a, k] += rewards[_states[a], a]
                states[a] = np.random.choice(n_states, p=transitions[_states[a], :, actions[a], a])
                counts[_states[a], states[a], actions[a], a] += 1
        for a in range(n_bandits):
            if u_type == 1:
                objectives[a, k] = 1 - thresholds[a]**(- 1/u_order) * (np.maximum(0, thresholds[a] - totalrewards[a, k]))**(1/u_order)
            elif u_type == 2:
                objectives[a, k] = (1 + np.exp(-u_order * (1-thresholds[a]))) / (1 + np.exp(-u_order * (totalrewards[a, k]-thresholds[a])))
            else:
                objectives[a, k] = 1 if totalrewards[a, k] >= thresholds[a] else 0

    return totalrewards, objectives, counts


def Process_LearnSafeRB(SafeW, whittle_indices, LearnW, learn_indices, n_episodes, n_steps, n_states, n_bandits, n_choices, thresholds, rewards, transitions, initial_states, u_type, u_order):

    ##################################################### Process
    totalrewards = np.zeros((n_bandits, n_episodes))
    objectives = np.zeros((n_bandits, n_episodes))
    learn_totalrewards = np.zeros((n_bandits, n_episodes))
    learn_objectives = np.zeros((n_bandits, n_episodes))
    counts = np.zeros((n_states, n_states, 2, n_bandits))
    for k in range(n_episodes):
        states = initial_states.copy()
        _lifted = np.zeros(n_bandits, dtype=np.int32)
        learn_states = initial_states.copy()
        _learn_lifted = np.zeros(n_bandits, dtype=np.int32)
        for t in range(n_steps):
            _states = np.copy(states)
            _learn_states = np.copy(learn_states)
            for a in range(n_bandits):
                _lifted[a] = max(0, min(SafeW.n_augment[a]-1, _lifted[a] + _states[a]))
                _learn_lifted[a] = max(0, min(LearnW.n_augment[a]-1, _learn_lifted[a] + _learn_states[a]))
            actions = SafeW.Whittle_policy(whittle_indices, n_choices, _states, _lifted, t)
            learn_actions = LearnW.Whittle_policy(learn_indices, n_choices, _learn_states, _learn_lifted, t)
            for a in range(n_bandits):
                if len(rewards.shape) == 3:
                    totalrewards[a, k] += rewards[_states[a], actions[a], a]
                    learn_totalrewards[a, k] += rewards[_learn_states[a], learn_actions[a], a]
                else:
                    totalrewards[a, k] += rewards[_states[a], a]
                    learn_totalrewards[a, k] += rewards[_learn_states[a], a]
                states[a] = np.random.choice(n_states, p=transitions[_states[a], :, actions[a], a])
                if actions[a] == learn_actions[a] and _states[a] == _learn_states[a]:
                    learn_states[a] = np.copy(states[a])
                else:
                    learn_states[a] = np.random.choice(n_states, p=transitions[_learn_states[a], :, learn_actions[a], a])
                counts[_learn_states[a], learn_states[a], learn_actions[a], a] += 1
                counts[_states[a], states[a], actions[a], a] += 1
        for a in range(n_bandits):
            if u_type == 1:
                objectives[a, k] = 1 - thresholds[a] ** (- 1 / u_order) * (np.maximum(0, thresholds[a] - totalrewards[a, k])) ** (1 / u_order)
                learn_objectives[a, k] = 1 - thresholds[a] ** (- 1 / u_order) * (np.maximum(0, thresholds[a] - learn_totalrewards[a, k])) ** (1 / u_order)
            elif u_type == 2:
                objectives[a, k] = (1 + np.exp(-u_order * (1 - thresholds[a]))) / (1 + np.exp(-u_order * (totalrewards[a, k] - thresholds[a])))
                learn_objectives[a, k] = (1 + np.exp(-u_order * (1 - thresholds[a]))) / (1 + np.exp(-u_order * (learn_totalrewards[a, k] - thresholds[a])))
            else:
                objectives[a, k] = 1 if totalrewards[a, k] >= thresholds[a] else 0
                learn_objectives[a, k] = 1 if learn_totalrewards[a, k] >= thresholds[a] else 0

    return totalrewards, objectives, learn_totalrewards, learn_objectives, counts


def Process_LearnSoftSafeRB(SafeW, whittle_indices, LearnW, learn_indices, n_episodes, n_steps, n_states, n_bandits, n_choices, thresholds, rewards, transitions, initial_states, u_type, u_order):

    ##################################################### Process
    totalrewards = np.zeros((n_bandits, n_episodes))
    objectives = np.zeros((n_bandits, n_episodes))
    learn_totalrewards = np.zeros((n_bandits, n_episodes))
    learn_objectives = np.zeros((n_bandits, n_episodes))
    counts = np.zeros((n_states, n_states, 2, n_bandits))
    for k in range(n_episodes):
        states = initial_states.copy()
        _lifted = np.zeros(n_bandits, dtype=np.int32)
        learn_states = initial_states.copy()
        _learn_lifted = np.zeros(n_bandits, dtype=np.int32)
        for t in range(n_steps):
            _states = np.copy(states)
            _learn_states = np.copy(learn_states)
            for a in range(n_bandits):
                _lifted[a] = max(0, min(SafeW.n_augment[a]-1, _lifted[a] + _states[a]))
                _learn_lifted[a] = max(0, min(LearnW.n_augment[a]-1, _learn_lifted[a] + _learn_states[a]))
            actions = SafeW.Whittle_softpolicy(whittle_indices, n_choices, _states, _lifted, t)
            learn_actions = LearnW.Whittle_softpolicy(learn_indices, n_choices, _learn_states, _learn_lifted, t)
            for a in range(n_bandits):
                if len(rewards.shape) == 3:
                    totalrewards[a, k] += rewards[_states[a], actions[a], a]
                    learn_totalrewards[a, k] += rewards[_learn_states[a], learn_actions[a], a]
                else:
                    totalrewards[a, k] += rewards[_states[a], a]
                    learn_totalrewards[a, k] += rewards[_learn_states[a], a]
                states[a] = np.random.choice(n_states, p=transitions[_states[a], :, actions[a], a])
                if actions[a] == learn_actions[a] and _states[a] == _learn_states[a]:
                    learn_states[a] = np.copy(states[a])
                else:
                    learn_states[a] = np.random.choice(n_states, p=transitions[_learn_states[a], :, learn_actions[a], a])
                counts[_learn_states[a], learn_states[a], learn_actions[a], a] += 1
                counts[_states[a], states[a], actions[a], a] += 1
        for a in range(n_bandits):
            if u_type == 1:
                objectives[a, k] = 1 - thresholds[a] ** (- 1 / u_order) * (np.maximum(0, thresholds[a] - totalrewards[a, k])) ** (1 / u_order)
                learn_objectives[a, k] = 1 - thresholds[a] ** (- 1 / u_order) * (np.maximum(0, thresholds[a] - learn_totalrewards[a, k])) ** (1 / u_order)
            elif u_type == 2:
                objectives[a, k] = (1 + np.exp(-u_order * (1 - thresholds[a]))) / (1 + np.exp(-u_order * (totalrewards[a, k] - thresholds[a])))
                learn_objectives[a, k] = (1 + np.exp(-u_order * (1 - thresholds[a]))) / (1 + np.exp(-u_order * (learn_totalrewards[a, k] - thresholds[a])))
            else:
                objectives[a, k] = 1 if totalrewards[a, k] >= thresholds[a] else 0
                learn_objectives[a, k] = 1 if learn_totalrewards[a, k] >= thresholds[a] else 0

    return totalrewards, objectives, learn_totalrewards, learn_objectives, counts

# def Process_LearnSoftSafeRB(SafeW, whittle_indices, n_episodes, n_steps, n_states, n_bandits, n_choices, thresholds, rewards, transitions, initial_states, u_type, u_order):
#     # LearnW, learn_indices,
#     ##################################################### Process
#     totalrewards = np.zeros((n_bandits, n_episodes))
#     objectives = np.zeros((n_bandits, n_episodes))
#     # learn_totalrewards = np.zeros((n_bandits, n_episodes))
#     # learn_objectives = np.zeros((n_bandits, n_episodes))
#     counts = np.zeros((n_states, n_states, 2, n_bandits))
#     for k in range(n_episodes):
#         states = initial_states.copy()
#         _lifted = np.zeros(n_bandits, dtype=np.int32)
#         # learn_states = initial_states.copy()
#         # _learn_lifted = np.zeros(n_bandits, dtype=np.int32)
#         for t in range(n_steps):
#             _states = np.copy(states)
#             # _learn_states = np.copy(learn_states)
#             for a in range(n_bandits):
#                 _lifted[a] = max(0, min(SafeW.n_augment[a] - 1, _lifted[a] + _states[a]))
#                 # _learn_lifted[a] = max(0, min(LearnW.n_augment[a]-1, _learn_lifted[a] + _learn_states[a]))
#             actions = SafeW.Whittle_softpolicy(whittle_indices, n_choices, _states, _lifted, t)
#             # learn_actions = LearnW.Whittle_softpolicy(learn_indices, n_choices, _learn_states, _learn_lifted, t)
#             for a in range(n_bandits):
#                 if len(rewards.shape) == 3:
#                     totalrewards[a, k] += rewards[_states[a], actions[a], a]
#                     # learn_totalrewards[a, k] += rewards[_learn_states[a], learn_actions[a], a]
#                 else:
#                     totalrewards[a, k] += rewards[_states[a], a]
#                     # learn_totalrewards[a, k] += rewards[_learn_states[a], a]
#                 states[a] = np.random.choice(n_states, p=transitions[_states[a], :, actions[a], a])
#                 # if plan_actions[a] == learn_actions[a] and _plan_states[a] == _learn_states[a]:
#                 #     learn_states[a] = np.copy(plan_states[a])
#                 # else:
#                 #     learn_states[a] = np.random.choice(n_states, p=transitions[_learn_states[a], :, learn_actions[a], a])
#                 # counts[_learn_states[a], learn_states[a], learn_actions[a], a] += 1
#         for a in range(n_bandits):
#             if u_type == 1:
#                 objectives[a, k] = 1 - thresholds[a] ** (- 1 / u_order) * (np.maximum(0, thresholds[a] - totalrewards[a, k])) ** (1 / u_order)
#                 learn_objectives[a, k] = 1 - thresholds[a] ** (- 1 / u_order) * (np.maximum(0, thresholds[a] - learn_totalrewards[a, k])) ** (1 / u_order)
#             elif u_type == 2:
#                 objectives[a, k] = (1 + np.exp(-u_order * (1-thresholds[a]))) / (1 + np.exp(-u_order * (totalrewards[a, k]-thresholds[a])))
#                 learn_objectives[a, k] = (1 + np.exp(-u_order * (1 - thresholds[a]))) / (1 + np.exp(-u_order * (learn_totalrewards[a, k] - thresholds[a])))
#             else:
#                 objectives[a, k] = 1 if totalrewards[a, k] >= thresholds[a] else 0
#                 learn_objectives[a, k] = 1 if learn_totalrewards[a, k] >= thresholds[a] else 0
#         # learn_totalrewards, learn_objectives,
#         return totalrewards, objectives, counts


def Process_SingleSafeRB(SafeW, whittle_indices, n_episodes, n_steps, n_states, n_bandits, n_choices, thresholds, rewards, transitions, initial_states, u_type, u_order):

    ##################################################### Process
    totalrewards = np.zeros((n_bandits, n_episodes))
    objectives = np.zeros((n_bandits, n_episodes))
    counts = np.zeros((n_states, n_states, 2, n_bandits))
    for k in range(n_episodes):
        states = initial_states.copy()
        _lifted = np.zeros(n_bandits, dtype=np.int32)
        for t in range(n_steps):
            _states = np.copy(states)
            _lifted[0] = max(0, min(SafeW.n_augment[0]-1, _lifted[0] + _states[0]))
            _lifted[1] = 0
            actions = SafeW.Whittle_policy(whittle_indices, n_choices, _states, _lifted, t)
            for a in range(n_bandits):
                if len(rewards.shape) == 3:
                    totalrewards[a, k] += rewards[_states[a], actions[a], a]
                else:
                    totalrewards[a, k] += rewards[_states[a], a]
                states[a] = np.random.choice(n_states, p=transitions[_states[a], :, actions[a], a])
                counts[_states[a], states[a], actions[a], a] += 1
        for a in range(n_bandits):
            if u_type == 1:
                objectives[a, k] = 1 - thresholds[a]**(- 1/u_order) * (np.maximum(0, thresholds[a] - totalrewards[a, k]))**(1/u_order)
            elif u_type == 2:
                objectives[a, k] = (1 + np.exp(-u_order * (1-thresholds[a]))) / (1 + np.exp(-u_order * (totalrewards[a, k]-thresholds[a])))
            else:
                objectives[a, k] = 1 if totalrewards[a, k] >= thresholds[a] else 0

    return totalrewards, objectives, counts


def Process_SingleSoftSafeRB(SafeW, whittle_indices, n_episodes, n_steps, n_states, n_bandits, n_choices, thresholds, rewards, transitions, initial_states, u_type, u_order):

    ##################################################### Process
    totalrewards = np.zeros((n_bandits, n_episodes))
    objectives = np.zeros((n_bandits, n_episodes))
    counts = np.zeros((n_states, n_states, 2, n_bandits))
    for k in range(n_episodes):
        states = initial_states.copy()
        _lifted = np.zeros(n_bandits, dtype=np.int32)
        for t in range(n_steps):
            _states = np.copy(states)
            _lifted[0] = max(0, min(SafeW.n_augment[0]-1, _lifted[0] + _states[0]))
            _lifted[1] = 0
            actions = SafeW.Whittle_softpolicy(whittle_indices, n_choices, _states, _lifted, t)
            for a in range(n_bandits):
                if len(rewards.shape) == 3:
                    totalrewards[a, k] += rewards[_states[a], actions[a], a]
                else:
                    totalrewards[a, k] += rewards[_states[a], a]
                states[a] = np.random.choice(n_states, p=transitions[_states[a], :, actions[a], a])
                counts[_states[a], states[a], actions[a], a] += 1
                
        for a in range(n_bandits):
            if u_type == 1:
                objectives[a, k] = 1 - thresholds[a]**(- 1/u_order) * (np.maximum(0, thresholds[a] - totalrewards[a, k]))**(1/u_order)
            elif u_type == 2:
                objectives[a, k] = (1 + np.exp(-u_order * (1-thresholds[a]))) / (1 + np.exp(-u_order * (totalrewards[a, k]-thresholds[a])))
            else:
                objectives[a, k] = 1 if totalrewards[a, k] >= thresholds[a] else 0

    return totalrewards, objectives, counts


def Process_LSSSRB(PlanW, plan_indices, LearnW, learn_indices, n_episodes, n_steps, n_states, n_bandits, n_choices, thresholds, rewards, transitions, initial_states, u_type, u_order):
    ##################################################### Process
    plan_totalrewards = np.zeros((n_bandits, n_episodes))
    plan_objectives = np.zeros((n_bandits, n_episodes))
    learn_totalrewards = np.zeros((n_bandits, n_episodes))
    learn_objectives = np.zeros((n_bandits, n_episodes))
    counts = np.zeros((n_states, n_states, 2, n_bandits))
    for k in range(n_episodes):
        plan_states = initial_states.copy()
        learn_states = initial_states.copy()
        _plan_lifted = np.zeros(n_bandits, dtype=np.int32)
        _learn_lifted = np.zeros(n_bandits, dtype=np.int32)
        for t in range(n_steps):
            _plan_states = np.copy(plan_states)
            _learn_states = np.copy(learn_states)
            _plan_lifted[0] = max(0, min(PlanW.n_augment[0] - 1, _plan_lifted[0] + _plan_states[0]))
            _plan_lifted[1] = 0
            _learn_lifted[0] = max(0, min(LearnW.n_augment[0] - 1, _learn_lifted[0] + _learn_states[0]))
            _learn_lifted[1] = 0
            plan_actions = PlanW.Whittle_softpolicy(plan_indices, n_choices, _plan_states, _plan_lifted, t)
            learn_actions = LearnW.Whittle_softpolicy(learn_indices, n_choices, _learn_states, _learn_lifted, t)
            for a in range(n_bandits):
                if len(rewards.shape) == 3:
                    plan_totalrewards[a, k] += rewards[_plan_states[a], plan_actions[a], a]
                    learn_totalrewards[a, k] += rewards[_learn_states[a], learn_actions[a], a]
                else:
                    plan_totalrewards[a, k] += rewards[_plan_states[a], a]
                    learn_totalrewards[a, k] += rewards[_learn_states[a], a]
                plan_states[a] = np.random.choice(n_states, p=transitions[_plan_states[a], :, plan_actions[a], a])
                if plan_actions[a] == learn_actions[a] and _plan_states[a] == _learn_states[a]:
                    learn_states[a] = np.copy(plan_states[a])
                else:
                    learn_states[a] = np.random.choice(n_states, p=transitions[_learn_states[a], :, learn_actions[a], a])
                counts[_learn_states[a], learn_states[a], learn_actions[a], a] += 1

        for a in range(n_bandits):
            if u_type == 1:
                plan_objectives[a, k] = 1 - thresholds[a] ** (- 1 / u_order) * (np.maximum(0, thresholds[a] - plan_totalrewards[a, k])) ** (1 / u_order)
                learn_objectives[a, k] = 1 - thresholds[a] ** (- 1 / u_order) * (np.maximum(0, thresholds[a] - learn_totalrewards[a, k])) ** (1 / u_order)
            elif u_type == 2:
                plan_objectives[a, k] = (1 + np.exp(-u_order * (1 - thresholds[a]))) / (1 + np.exp(-u_order * (plan_totalrewards[a, k] - thresholds[a])))
                learn_objectives[a, k] = (1 + np.exp(-u_order * (1 - thresholds[a]))) / (1 + np.exp(-u_order * (learn_totalrewards[a, k] - thresholds[a])))
            else:
                plan_objectives[a, k] = 1 if plan_totalrewards[a, k] >= thresholds[a] else 0
                learn_objectives[a, k] = 1 if learn_totalrewards[a, k] >= thresholds[a] else 0

    return plan_totalrewards, plan_objectives, learn_totalrewards, learn_objectives, counts
