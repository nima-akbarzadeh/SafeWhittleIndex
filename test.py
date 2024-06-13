from whittle import *
from processes import *
from learning import *
import matplotlib.pyplot as plt
import joblib


if __name__ == '__main__':

    # Basic Parameters
    n_steps = 5
    n_states = 2
    n_arms = 2
    n_coeff = 1
    u_type = 3
    u_order = 16
    thresholds = 0.5 * np.ones(n_arms)
    choice_fraction = 0.3

    transition_type = 3

    function_type = np.ones(n_arms, dtype=np.int32)
    # function_type = 1 + np.arange(n_arms)
    # np.random.shuffle(function_type)

    n_episodes = 500
    # np.random.seed(42)

    na = n_arms
    ns = n_states
    tt = transition_type
    prob_remain = np.round(np.linspace(0.1 / ns, 1 / ns, na), 2)
    if tt == 11:
        n_states = 4
        pr_ss_0 = np.round(np.linspace(0.596, 0.690, na), 3)
        np.random.shuffle(pr_ss_0)
        pr_sr_0 = np.round(np.linspace(0.045, 0.061, na), 3)
        np.random.shuffle(pr_sr_0)
        pr_sp_0 = np.round(np.linspace(0.201, 0.287, na), 3)
        np.random.shuffle(pr_sp_0)
        pr_rr_0 = np.round(np.linspace(0.759, 0.822, na), 3)
        np.random.shuffle(pr_rr_0)
        pr_rp_0 = np.round(np.linspace(0.130, 0.169, na), 3)
        np.random.shuffle(pr_rp_0)
        pr_pp_0 = np.round(np.linspace(0.882, 0.922, na), 3)
        np.random.shuffle(pr_pp_0)
        pr_ss_1 = np.round(np.linspace(0.733, 0.801, na), 3)
        np.random.shuffle(pr_ss_1)
        pr_sr_1 = np.round(np.linspace(0.047, 0.078, na), 3)
        np.random.shuffle(pr_sr_1)
        pr_sp_1 = np.round(np.linspace(0.115, 0.171, na), 3)
        np.random.shuffle(pr_sp_1)
        pr_rr_1 = np.round(np.linspace(0.758, 0.847, na), 3)
        np.random.shuffle(pr_rr_1)
        pr_rp_1 = np.round(np.linspace(0.121, 0.193, na), 3)
        np.random.shuffle(pr_rp_1)
        pr_pp_1 = np.round(np.linspace(0.879, 0.921, na), 3)
        np.random.shuffle(pr_pp_1)
        prob_remain = [pr_ss_0, pr_sr_0, pr_sp_0, pr_rr_0, pr_rp_0, pr_pp_0, pr_ss_1, pr_sr_1, pr_sp_1, pr_rr_1, pr_rp_1, pr_pp_1]
    elif tt == 12:
        n_states = 4
        pr_ss_0 = np.round(np.linspace(0.668, 0.738, na), 3)
        np.random.shuffle(pr_ss_0)
        pr_sr_0 = np.round(np.linspace(0.045, 0.061, na), 3)
        np.random.shuffle(pr_sr_0)
        pr_rr_0 = np.round(np.linspace(0.831, 0.870, na), 3)
        np.random.shuffle(pr_rr_0)
        pr_pp_0 = np.round(np.linspace(0.882, 0.922, na), 3)
        np.random.shuffle(pr_pp_0)
        pr_ss_1 = np.round(np.linspace(0.782, 0.833, na), 3)
        np.random.shuffle(pr_ss_1)
        pr_sr_1 = np.round(np.linspace(0.047, 0.078, na), 3)
        np.random.shuffle(pr_sr_1)
        pr_rr_1 = np.round(np.linspace(0.807, 0.879, na), 3)
        np.random.shuffle(pr_rr_1)
        pr_pp_1 = np.round(np.linspace(0.879, 0.921, na), 3)
        np.random.shuffle(pr_pp_1)
        prob_remain = [pr_ss_0, pr_sr_0, pr_rr_0, pr_pp_0, pr_ss_1, pr_sr_1, pr_rr_1, pr_pp_1]
    elif tt == 13:
        n_states = 3
        pr_ss_0 = np.round(np.linspace(0.657, 0.762, na), 3)
        np.random.shuffle(pr_ss_0)
        pr_sp_0 = np.round(np.linspace(0.201, 0.287, na), 3)
        np.random.shuffle(pr_sp_0)
        pr_pp_0 = np.round(np.linspace(0.882, 0.922, na), 3)
        np.random.shuffle(pr_pp_0)
        pr_ss_1 = np.round(np.linspace(0.806, 0.869, na), 3)
        np.random.shuffle(pr_ss_1)
        pr_sp_1 = np.round(np.linspace(0.115, 0.171, na), 3)
        np.random.shuffle(pr_sp_1)
        pr_pp_1 = np.round(np.linspace(0.879, 0.921, na), 3)
        np.random.shuffle(pr_pp_1)
        prob_remain = [pr_ss_0, pr_sp_0, pr_pp_0, pr_ss_1, pr_sp_1, pr_pp_1]
    elif tt == 14:
        n_states = 3
        pr_ss_0 = np.round(np.linspace(0.713, 0.799, na), 3)
        np.random.shuffle(pr_ss_0)
        pr_pp_0 = np.round(np.linspace(0.882, 0.922, na), 3)
        np.random.shuffle(pr_pp_0)
        pr_ss_1 = np.round(np.linspace(0.829, 0.885, na), 3)
        np.random.shuffle(pr_ss_1)
        pr_pp_1 = np.round(np.linspace(0.879, 0.921, na), 3)
        np.random.shuffle(pr_pp_1)
        prob_remain = [pr_ss_0, pr_pp_0, pr_ss_1, pr_pp_1]
    print(prob_remain)

    reward_increasing = True
    transition_increasing = True
    max_wi = 1

    # Simulation Parameters
    n_choices = np.maximum(1, int(choice_fraction * n_arms))
    initial_states = (n_states - 1) * np.ones(n_arms, dtype=np.int32)

    # Basic Parameters
    R = Values(n_steps, n_arms, n_states, function_type, reward_increasing)
    M = MarkovDynamics(n_arms, n_states, prob_remain, transition_type, transition_increasing)
    reward_bandits = R.vals
    transition_bandits = M.transitions

    n_trials_neutrl = n_arms * n_states * n_steps
    n_trials_safety = n_arms * n_states * n_steps
    method = 3

    W = Whittle(n_states, n_arms, reward_bandits, transition_bandits, n_steps)
    W.get_whittle_indices(computation_type=method, params=[0, max_wi], n_trials=n_trials_neutrl)
    w_bandits = W.w_indices

    SafeW = SafeWhittle(n_states, n_arms, reward_bandits, transition_bandits, n_steps, u_type, u_order, thresholds)
    SafeW.get_whittle_indices(computation_type=method, params=[0, max_wi], n_trials=n_trials_safety)
    sw_bandits = SafeW.w_indices

    print('Process Begins ...')
    rew_r, obj_r, _ = Process_Random(n_episodes, n_steps, n_states, n_arms, n_choices, thresholds, reward_bandits, transition_bandits, initial_states, u_type, u_order)
    rew_m, obj_m, _ = Process_Greedy(n_episodes, n_steps, n_states, n_arms, n_choices, thresholds, reward_bandits, transition_bandits, initial_states, u_type, u_order)
    rew_w, obj_w, _ = Process_WhtlRB(W, w_bandits, n_episodes, n_steps, n_states, n_arms, n_choices, thresholds, reward_bandits, transition_bandits, initial_states, u_type, u_order)
    rew_ss, obj_ss, _, _, _ = Process_LearnSoftSafeRB(SafeW, sw_bandits, SafeW, sw_bandits.copy(), n_episodes, n_steps, n_states, n_arms, n_choices, thresholds, reward_bandits, transition_bandits, initial_states, u_type, u_order)
    rew_s, obj_s, _ = Process_SafeRB(SafeW, sw_bandits, n_episodes, n_steps, n_states, n_arms, n_choices, thresholds, reward_bandits, transition_bandits, initial_states, u_type, u_order)
    print('Process Ends ...')

    print("=============== REWARD/OBJECTIVE PER-ARM =====================")
    for a in range(n_arms):
        print(f'------ Arm {a}')
        print(f'REW - Whittl: {np.mean(rew_w[a, :])}')
        print(f'REW - Safety: {np.mean(rew_s[a, :])}')
        print(f'REW - Safety-Whittl: {100 * (np.mean(rew_s[a, :]) - np.mean(rew_w[a, :])) / np.mean(rew_w[a, :])}')
        print(f'OBJ - Whittl: {np.mean(obj_w[a, :])}')
        print(f'OBJ - Safety: {np.mean(obj_s[a, :])}')
        print(f'OBJ - Safety-Whittl: {100 * (np.mean(obj_s[a, :]) - np.mean(obj_w[a, :])) / np.mean(obj_w[a, :])}')

    print("====================== REWARDS =========================")
    print(f'Random: {np.mean(rew_r)}')
    print(f'Myopic: {np.mean(rew_m)}')
    print(f'Whittl: {np.mean(rew_w)}')
    print(f'SoSafe: {np.mean(rew_ss)}')
    print(f'Safety: {np.mean(rew_s)}')

    print("====================== LOSS ========================")
    print(f'Safety-Whittl: {100 * (np.mean(rew_s) - np.mean(rew_w)) / np.mean(rew_w)}')
    print(f'Safety-Myopic: {100 * (np.mean(rew_s) - np.mean(rew_m)) / np.mean(rew_m)}')
    print(f'Safety-Random: {100 * (np.mean(rew_s) - np.mean(rew_r)) / np.mean(rew_r)}')

    print("===================== OBJECTIVE ========================")
    print(f'Random: {np.mean(obj_r)}')
    print(f'Myopic: {np.mean(obj_m)}')
    print(f'Whittl: {np.mean(obj_w)}')
    print(f'SoSafe: {np.mean(obj_ss)}')
    print(f'Safety: {np.mean(obj_s)}')

    print("===================== IMPROVEMENT ========================")
    print(f'Safety-SoSafe: {100 * (np.mean(obj_s) - np.mean(obj_ss)) / np.mean(obj_ss)}')
    print(f'Safety-Whittl: {100 * (np.mean(obj_s) - np.mean(obj_w)) / np.mean(obj_w)}')
    print(f'Safety-Myopic: {100 * (np.mean(obj_s) - np.mean(obj_m)) / np.mean(obj_m)}')
    print(f'Safety-Random: {100 * (np.mean(obj_s) - np.mean(obj_r)) / np.mean(obj_r)}')

    # arm_index = int(input('Which arm: '))
    # bins = np.linspace(0.2, 0.8, 400)
    # plt.hist(rew_w[arm_index, :], bins=bins, alpha=0.5, label='Risk-Neutral', width=0.05, align='left', color='gray', edgecolor='black', hatch='/')
    # plt.hist(rew_s[arm_index, :], bins=bins, alpha=0.5, label='Risk-Aware', width=0.05, align='mid', color='white', edgecolor='black', hatch='\\')
    # plt.xticks(fontsize=12, fontweight='bold')
    # plt.yticks(fontsize=12, fontweight='bold')
    # plt.axvline(x=thresholds[-1], color='r', linestyle='-')
    # plt.xlim(0, 1)
    # plt.xlabel('Total Rewards', fontsize=14, fontweight='bold')
    # plt.ylabel('Frequency', fontsize=14, fontweight='bold')
    # plt.title('Distribution of Rewards', fontsize=14, fontweight='bold')
    # plt.legend()
    # plt.grid()
    # plt.show()

    # bins = np.linspace(0, 1, 20)
    # plt.hist(np.mean(rew_w, axis=0), bins=bins, alpha=0.5, label='Risk-Neutral', width=0.05, align='left')
    # plt.hist(np.mean(rew_s, axis=0), bins=bins, alpha=0.5, label='Risk-Aware', width=0.05, align='mid')
    # plt.xticks(fontsize=12, fontweight='bold')
    # plt.yticks(fontsize=12, fontweight='bold')
    # plt.axvline(x=thresholds[-1], color='r', linestyle='-')
    # plt.xlim(0.2, 0.8)
    # plt.xlabel('Total Rewards', fontsize=14, fontweight='bold')
    # plt.ylabel('Frequency', fontsize=14, fontweight='bold')
    # plt.title('Distribution of Rewards', fontsize=14, fontweight='bold')
    # plt.legend()
    # plt.grid()
    # plt.show()

    rb_type = 'hard'  # 'hard' or 'soft'
    exp_type = 'det'  # 'det' or 'rand'
    n_episodes = 100
    n_iterations = 1
    n_priors = 10
    l_episodes = 500

    ##################################################### Process
    n_bandits = n_arms
    LearnW = SafeW
    learn_indices = sw_bandits
    for a in range(n_arms):
        print(f'============== {a}')
        print(sw_bandits[a])
        print(learn_indices[a])
        # learn_indices[a] = sw_bandits[a] + np.random.normal(10, 100, sw_bandits[a].shape)

    totalrewards = np.zeros((n_bandits, n_episodes))
    objectives = np.zeros((n_bandits, n_episodes))
    learn_totalrewards = np.zeros((n_bandits, n_episodes))
    learn_objectives = np.zeros((n_bandits, n_episodes))
    counts = np.zeros((n_states, n_states, 2, n_bandits))

    for k in range(n_episodes):
        print(k)
        states = initial_states.copy()
        _lifted = np.zeros(n_bandits, dtype=np.int32)
        learn_states = initial_states.copy()
        _learn_lifted = np.zeros(n_bandits, dtype=np.int32)
        for t in range(n_steps):
            _states = np.copy(states)
            _learn_states = np.copy(learn_states)
            for a in range(n_bandits):
                _lifted[a] = max(0, min(SafeW.n_augment[a] - 1, _lifted[a] + _states[a]))
                _learn_lifted[a] = max(0, min(LearnW.n_augment[a] - 1, _learn_lifted[a] + _learn_states[a]))
            actions = SafeW.Whittle_policy(sw_bandits, n_choices, _states, _lifted, t)
            learn_actions = LearnW.Whittle_policy(learn_indices, n_choices, _learn_states, _learn_lifted, t)
            for a in range(n_bandits):
                if len(reward_bandits.shape) == 3:
                    totalrewards[a, k] += reward_bandits[_states[a], actions[a], a]
                    learn_totalrewards[a, k] += reward_bandits[_learn_states[a], learn_actions[a], a]
                else:
                    totalrewards[a, k] += reward_bandits[_states[a], a]
                    learn_totalrewards[a, k] += reward_bandits[_learn_states[a], a]
                states[a] = np.random.choice(n_states, p=transition_bandits[_states[a], :, actions[a], a])
                if actions[a] == learn_actions[a] and _states[a] == _learn_states[a]:
                    learn_states[a] = np.copy(states[a])
                else:
                    learn_states[a] = np.random.choice(n_states, p=transition_bandits[_learn_states[a], :, learn_actions[a], a])
                counts[_learn_states[a], learn_states[a], learn_actions[a], a] += 1
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

    print('3')
    print(np.mean(objectives))
    print(np.mean(learn_objectives))
