from whittle import *
from processes import *
from learning import *
import matplotlib.pyplot as plt
import joblib


if __name__ == '__main__':

    # Basic Parameters
    n_steps = 5
    n_coeff = 1
    n_states = 2
    u_type = 1
    u_order = 8
    n_arms = n_coeff * n_states
    thresholds = 0.5 * np.ones(n_arms)
    choice_fraction = 0.2

    transition_type = 3
    function_type = np.ones(n_arms, dtype=np.int32)
    # function_type = 1 + np.arange(n_arms)

    n_episodes = 100
    np.random.seed(42)

    na = n_arms
    ns = n_states
    tt = transition_type
    if tt == 0:
        prob_remain = np.round(np.linspace(0.1, 0.9, na), 2)
        np.random.shuffle(prob_remain)
    elif tt == 1:
        prob_remain = np.round(np.linspace(0.05, 0.45, na), 2)
        np.random.shuffle(prob_remain)
    elif tt == 2:
        prob_remain = np.round(np.linspace(0.05, 0.45, na), 2)
        np.random.shuffle(prob_remain)
    elif tt == 3:
        # prob_remain = np.round(np.linspace(0.1 / ns, 1 / ns, na), 2)
        prob_remain = np.round(np.linspace(0.5 / ns, 0.5 / ns, na), 2)
        np.random.shuffle(prob_remain)
    elif tt == 4:
        prob_remain = np.round(np.linspace(0.1 / ns, 1 / ns, na), 2)
        np.random.shuffle(prob_remain)
    elif tt == 5:
        prob_remain = np.round(np.linspace(0.1 / ns, 1 / ns, na), 2)
        np.random.shuffle(prob_remain)
    elif tt == 6:
        prob_remain = np.round(np.linspace(0.2, 0.8, na), 2)
        np.random.shuffle(prob_remain)
    elif tt == 11:
        pr_ss_0 = np.round(np.linspace(0.596, 0.690, na), 3)
        np.random.shuffle(pr_ss_0)
        print(pr_ss_0)
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
        pr_ss_0 = np.round(np.linspace(0.713, 0.799, na), 3)
        np.random.shuffle(pr_ss_0)
        pr_pp_0 = np.round(np.linspace(0.882, 0.922, na), 3)
        np.random.shuffle(pr_pp_0)
        pr_ss_1 = np.round(np.linspace(0.829, 0.885, na), 3)
        np.random.shuffle(pr_ss_1)
        pr_pp_1 = np.round(np.linspace(0.879, 0.921, na), 3)
        np.random.shuffle(pr_pp_1)
        prob_remain = [pr_ss_0, pr_pp_0, pr_ss_1, pr_pp_1]
    else:
        prob_remain = np.round(np.linspace(0.1, 0.9, na), 2)

    reward_increasing = True
    transition_increasing = True
    max_wi = 1

    np.random.shuffle(function_type)

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
    # bins = np.linspace(0.2, 1, 400)
    # plt.hist(rew_w[arm_index, :], bins=bins, alpha=0.5, label='Risk-Neutral', width=0.05, align='left')
    # plt.hist(rew_s[arm_index, :], bins=bins, alpha=0.5, label='Risk-Aware', width=0.05, align='mid')
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
    #
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

    rb_type = 'soft'  # 'hard' or 'soft'
    n_iterations = 5
    l_episodes = 50
    if rb_type == 'hard':
        probs_l, sumwis_l, rew_l, obj_l, swi_ss, rew_ss, obj_ss = Process_LearnSafeTSRB(n_iterations, l_episodes, n_episodes, n_steps, n_states, n_arms, n_choices, thresholds,
                                                                                        transition_type, transition_increasing, method, reward_bandits, transition_bandits,
                                                                                        initial_states, u_type, u_order, True, max_wi)
        # learn_list = joblib.load(f'./output/safetsrb_{n_steps}{n_states}{n_arms}{tt}{u_type}{n_choices}{thresholds[0]}.joblib')
        # probs_l = learn_list[0]
        # sumwis_l = learn_list[1]
        # rew_l = learn_list[2]
        # obj_l = learn_list[3]
    else:
        probs_l, sumwis_l, rew_l, obj_l, swi_ss, rew_ss, obj_ss = Process_LearnSoftSafeTSRB(n_iterations, l_episodes, n_episodes, n_steps, n_states, n_arms, n_choices, thresholds,
                                                                                            transition_type, transition_increasing, method, reward_bandits, transition_bandits,
                                                                                            initial_states, u_type, u_order, True, max_wi)
        # rew_ss, obj_ss, _ = Process_SoftSafeRB(SafeW, n_episodes, n_steps, n_states, n_arms, n_choices, thresholds, reward_bandits, transition_bandits,
        #                                        sw_bandits, initial_states, u_type, u_order)
        # probs_l, sumwis_l, rew_l, obj_l = Process_SafeSoftTSRB(n_iterations, l_episodes, n_episodes, n_steps, n_states, n_arms, n_choices, thresholds,
        #                                                        transition_type, transition_increasing, method, reward_bandits, transition_bandits,
        #                                                        initial_states, u_type, u_order, True, max_wi)
        # learn_list = joblib.load(f'./output/safesofttsrb_{n_steps}{n_states}{n_arms}{tt}{u_type}{n_choices}{thresholds[0]}.joblib')
        # probs_l = learn_list[0]
        # sumwis_l = learn_list[1]
        # rew_l = learn_list[2]
        # obj_l = learn_list[3]

    ma_coef = 0.1
    def moving_average(x, w):
        return np.convolve(x, np.ones(w), 'same') / w

    prb_err = np.abs(np.transpose(np.array([prob_remain[a] - np.mean(probs_l[:, :, a], axis=0) for a in range(n_arms)])))
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

    swi_err = np.abs(np.transpose(np.array([swi_ss[a] - np.mean(sumwis_l[:, :, a], axis=0) for a in range(n_arms)])))
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

    reg = np.cumsum(np.mean(obj_ss) - np.mean(obj_l, axis=(0, 2)))
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
