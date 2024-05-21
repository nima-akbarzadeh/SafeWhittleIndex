import numpy as np

from whittle import *
from processes import *
from learning import *
import matplotlib.pyplot as plt
import joblib


if __name__ == '__main__':

    # Basic Parameters
    n_steps = 10
    n_states = 3
    u_type = 1
    u_order = 8
    n_arms = 1
    thresholds = 0.5 * np.ones(n_arms)

    transition_type = 3
    function_type = np.ones(n_arms, dtype=np.int32)

    n_episodes = 100
    np.random.seed(42)

    prob_remain = [np.round(0.5 / n_states, 2)]
    reward_increasing = True
    transition_increasing = True
    max_wi = 1

    # Simulation Parameters
    n_choices = 1

    # Basic Parameters
    R = Values(n_steps, n_arms, n_states, function_type, reward_increasing)
    M = MarkovDynamics(n_arms, n_states, prob_remain, transition_type, transition_increasing)
    reward_bandits = R.vals
    transition_bandits = M.transitions

    n_trials_neutrl = n_arms * n_states * n_steps
    n_trials_safety = n_arms * n_states * n_steps
    method = 3

    # fixed_wi = 0.01
    # fixed_wi = 0.5
    fixed_wi = 1

    W = Whittle(n_states, n_arms, reward_bandits, transition_bandits, n_steps)
    W.get_whittle_indices(computation_type=method, params=[0, max_wi], n_trials=n_trials_neutrl)
    w_bandits = W.w_indices
    w_bandits.append(fixed_wi * np.ones_like(np.array(w_bandits[0])))

    SafeW = SafeWhittle(n_states, n_arms, reward_bandits, transition_bandits, n_steps, u_type, u_order, thresholds)
    SafeW.get_whittle_indices(computation_type=method, params=[0, max_wi], n_trials=n_trials_safety)
    sw_bandits = SafeW.w_indices
    sw_bandits.append(fixed_wi * np.ones_like(np.array(sw_bandits[0])))

    n_arms = 2
    reward_bandits = np.concatenate((R.vals, 0.5 * np.ones_like(R.vals)), axis=1)
    transition_bandits = np.concatenate((M.transitions, M.transitions), axis=3)
    initial_states = (n_states - 1) * np.ones(n_arms, dtype=np.int32)
    thresholds = 0.5 * np.ones(n_arms)

    print('Process Begins ...')
    rew_r, obj_r, _ = Process_Random(n_episodes, n_steps, n_states, n_arms, n_choices, thresholds, reward_bandits, transition_bandits, initial_states, u_type, u_order)
    rew_m, obj_m, _ = Process_Greedy(n_episodes, n_steps, n_states, n_arms, n_choices, thresholds, reward_bandits, transition_bandits, initial_states, u_type, u_order)
    rew_w, obj_w, _ = Process_WhtlRB(W, w_bandits, n_episodes, n_steps, n_states, n_arms, n_choices, thresholds, reward_bandits, transition_bandits, initial_states, u_type, u_order)
    rew_ss, obj_ss, _ = Process_SingleSoftSafeRB(SafeW, sw_bandits, n_episodes, n_steps, n_states, n_arms, n_choices, thresholds, reward_bandits, transition_bandits, initial_states, u_type, u_order)
    rew_s, obj_s, _ = Process_SingleSafeRB(SafeW, sw_bandits, n_episodes, n_steps, n_states, n_arms, n_choices, thresholds, reward_bandits, transition_bandits, initial_states, u_type, u_order)
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

    print('=============================================================================')
    print(f'true_pr = {prob_remain[0]}')
    print(f'true_sw = {np.round(np.sum(sw_bandits[0]), 2)}')
    print(f'true_re = {np.round(np.mean(rew_ss), 2)}')
    print(f'true_ob = {np.round(np.mean(obj_ss), 2)}')

    rb_type = 'soft'  # 'hard' or 'soft'
    # initial_states = np.random.randint(0, n_states, n_arms)
    n_iterations = 1
    l_episodes = 50
    if rb_type == 'hard':
        n_episodes = 100
        probs_l, sumwis_l, rew_l, obj_l = Process_SingleSafeTSRB(n_iterations, l_episodes, n_episodes, n_steps, n_states, fixed_wi, n_choices, thresholds,
                                                                 transition_type, transition_increasing, method, reward_bandits, transition_bandits,
                                                                 initial_states, u_type, u_order, True, max_wi)
        # rew_ss, obj_ss, _ = Process_SingleSafeRB(SafeW, n_episodes, n_steps, n_states, n_arms, n_choices, thresholds, reward_bandits, transition_bandits,
        #                                          sw_bandits, initial_states, u_type, u_order)
        # n_episodes = 1000
        # probs_l, sumwis_l, rew_l, obj_l = Process_SingleSafeTSRB(n_iterations, l_episodes, n_episodes, n_steps, n_states, fixed_wi, n_choices, thresholds,
        #                                                          transition_type, transition_increasing, method, reward_bandits, transition_bandits,
        #                                                          initial_states, u_type, u_order, True, max_wi)
        # # learn_list = joblib.load(f'./output/safetsrb_{n_steps}{n_states}{n_arms}{tt}{u_type}{n_choices}{thresholds[0]}.joblib')
        # # probs_l = learn_list[0]
        # # sumwis_l = learn_list[1]
        # # rew_l = learn_list[2]
        # # obj_l = learn_list[3]
    else:
        probs_l, sumwis_l, rew_l, obj_l, swi_ss, rew_ss, obj_ss = Process_SingleSafeSoftTSRB(n_iterations, l_episodes, n_episodes, n_steps, n_states, fixed_wi, n_choices, thresholds,
                                                                                             transition_type, transition_increasing, method, reward_bandits, transition_bandits,
                                                                                             initial_states, u_type, u_order, True, max_wi)
        # rew_ss, obj_ss, _ = Process_SingleSoftSafeRB(SafeW, n_episodes, n_steps, n_states, n_arms, n_choices, thresholds, reward_bandits, transition_bandits,
        #                                              sw_bandits, initial_states, u_type, u_order)
        # n_episodes = 1000
        # probs_l, sumwis_l, rew_l, obj_l = Process_SingleSafeSoftTSRB(n_iterations, l_episodes, n_episodes, n_steps, n_states, fixed_wi, n_choices, thresholds,
        #                                                              transition_type, transition_increasing, method, reward_bandits, transition_bandits,
        #                                                              initial_states, u_type, u_order, True, max_wi)
        # # learn_list = joblib.load(f'./output/safesofttsrb_{n_steps}{n_states}{n_arms}{tt}{u_type}{n_choices}{thresholds[0]}.joblib')
        # # probs_l = learn_list[0]
        # # sumwis_l = learn_list[1]
        # # rew_l = learn_list[2]
        # # obj_l = learn_list[3]

    ma_coef = 0.1
    def moving_average(x, w):
        return np.convolve(x, np.ones(w), 'same') / w

    prb_err = prob_remain[0]*np.ones(l_episodes) - np.round(np.mean(probs_l, axis=0), 2)
    # for a in range(n_arms):
    #     prb_err[:, a] = moving_average(prb_err[:, a], int(ma_coef*l_episodes))
    plt.figure(figsize=(8, 6))
    plt.plot(prb_err, label='Mean')
    plt.xlabel('Episodes')
    plt.ylabel('Regret')
    plt.title('Mean and Bounds over regret')
    plt.legend()
    plt.grid(True)
    plt.show()

    swi_err = swi_ss*np.ones(l_episodes) - np.round(np.mean(sumwis_l, axis=0), 2)
    # for a in range(n_arms):
    #     swi_err[:, a] = moving_average(swi_err[:, a], int(ma_coef*l_episodes))
    plt.figure(figsize=(8, 6))
    plt.plot(swi_err, label='Mean')
    plt.xlabel('Episodes')
    plt.ylabel('Regret')
    plt.title('Mean and Bounds over regret')
    plt.legend()
    plt.grid(True)
    plt.show()

    reg = np.cumsum(np.round(np.mean(obj_ss)*np.ones(l_episodes), 2) - np.round(np.mean(obj_l, axis=(0, 2)), 2))
    # reg = moving_average(reg, int(ma_coef*l_episodes))
    plt.figure(figsize=(8, 6))
    plt.plot(reg, label='Mean')
    plt.xlabel('Episodes')
    plt.ylabel('Regret')
    plt.title('Mean and Bounds over regret')
    plt.legend()
    plt.grid(True)
    plt.show()

    # mean_reg = np.mean(reg, axis=0)
    #
    # # Calculate upper and lower bounds
    # upper_bound = np.max(reg, axis=0)
    # lower_bound = np.min(reg, axis=0)

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot([reg[t]/(t+1) for t in range(len(reg))], label='Mean')

    # # Fill between lower bound and upper bound
    # plt.fill_between(range(len(mean_reg)), lower_bound, upper_bound, color='skyblue', alpha=0.4, label='Bounds')

    plt.xlabel('Episodes')
    plt.ylabel('Regret')
    plt.title('Mean and Bounds over regret')
    plt.legend()
    plt.grid(True)
    plt.show()
    # plt.savefig(f'./output/regret_{n_steps}{n_states}{n_arms}{tt}{u_type}{u_order}{n_choices}{thresholds[0]}.png')
    # plt.savefig(f'./output/regret_{n_steps}{n_states}{n_arms}{tt}{u_type}{u_order}{n_choices}{thresholds[0]}.jpg')
