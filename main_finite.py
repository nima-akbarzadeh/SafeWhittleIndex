from whittle import *
from processes import *
from learning import *
import matplotlib.pyplot as plt
import joblib


if __name__ == '__main__':

    # Basic Parameters
    n_steps = 100
    n_states = 2
    n_arms = 3
    n_coeff = 1
    u_type = 3
    u_order = 1
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
    rew_r, obj_r, _ = Process_Random(n_episodes, n_steps, n_states, n_arms, n_choices, thresholds, reward_bandits,
                                     transition_bandits, initial_states, u_type, u_order)
    rew_m, obj_m, _ = Process_Greedy(n_episodes, n_steps, n_states, n_arms, n_choices, thresholds, reward_bandits,
                                     transition_bandits, initial_states, u_type, u_order)
    rew_w, obj_w, _ = Process_WhtlRB(W, w_bandits, n_episodes, n_steps, n_states, n_arms, n_choices, thresholds,
                                     reward_bandits, transition_bandits, initial_states, u_type, u_order)
    rew_s, obj_s, _ = Process_SafeRB(SafeW, sw_bandits, n_episodes, n_steps, n_states, n_arms, n_choices, thresholds,
                                     reward_bandits, transition_bandits, initial_states, u_type, u_order)
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
    print(f'Safety: {np.mean(rew_s)}')

    print("====================== LOSS ========================")
    print(f'Safety-Whittl: {100 * np.mean((obj_s - obj_w) / obj_w)}')
    print(f'Safety-Myopic: {100 * (np.mean(rew_s) - np.mean(rew_m)) / np.mean(rew_m)}')
    print(f'Safety-Random: {100 * (np.mean(rew_s) - np.mean(rew_r)) / np.mean(rew_r)}')

    print("===================== OBJECTIVE ========================")
    print(f'Random: {np.mean(obj_r)}')
    print(f'Myopic: {np.mean(obj_m)}')
    print(f'Whittl: {np.mean(obj_w)}')
    print(f'SoSafe: {np.mean(obj_ss)}')
    print(f'Safety: {np.mean(obj_s)}')

    print("===================== IMPROVEMENT ========================")
    print(f'Safety-Whittl: {100 * np.mean((obj_s - obj_w) / obj_w)}')
    print(f'Safety-Myopic: {100 * (np.mean(obj_s) - np.mean(obj_m)) / np.mean(obj_m)}')
    print(f'Safety-Random: {100 * (np.mean(obj_s) - np.mean(obj_r)) / np.mean(obj_r)}')

    arm_index = 2

    # Define the bins
    bins = np.linspace(0.05, 0.95, 10)
    print(bins)
    # bin_centers = (bins[:-1] + bins[1:]) / 2
    bar_width = 2 * (bins[1] - bins[0]) / 4
    plt.hist(rew_w[arm_index, :], bins=bins, alpha=0.75, label='Risk-Neutral', width=bar_width, align='left', color='gray', edgecolor='black', hatch='/')
    plt.hist(rew_s[arm_index, :], bins=bins, alpha=0.75, label='Risk-Aware', width=bar_width, align='mid', color='white', edgecolor='black', hatch='\\')
    # plt.hist(rew_w[arm_index, :], bins=bins, alpha=0.75, label='Risk-Neutral', width=bar_width, align='left', color='gray', edgecolor='black', hatch='/')
    # plt.hist(rew_s[arm_index, :], bins=bins, alpha=0.75, label='Risk-Aware', width=bar_width, align='left', color='white', edgecolor='black', hatch='\\')
    # for i in range(len(bin_centers)):
    #     plt.bar(bin_centers[i] - bar_width / 2, np.histogram(rew_w[arm_index, :], bins=bins)[0][i], width=bar_width, alpha=0.75, color='gray', edgecolor='black', hatch='/')
    #     plt.bar(bin_centers[i] + bar_width / 2, np.histogram(rew_s[arm_index, :], bins=bins)[0][i], width=bar_width, alpha=0.75, color='white', edgecolor='black', hatch='\\')
    plt.xticks(bins, rotation=90, fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    threshold = 0.5
    plt.axvline(x=threshold, color='r', linestyle='-')
    plt.xlim(0, 1)
    plt.xlabel('Total Rewards', fontsize=14, fontweight='bold')
    plt.ylabel('Frequency', fontsize=14, fontweight='bold')
    plt.title('Distribution of Rewards', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

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

    # rb_type = 'hard'  # 'hard' or 'soft'
    # exp_type = 'det'  # 'det' or 'rand'
    # n_episodes = 100
    # n_iterations = 10
    # n_priors = 10
    # l_episodes = 100
    # if rb_type == 'hard':
    #     if exp_type == 'det':
    #         transerror_l, wierrors_l, rew_l, obj_l, rew_ss, obj_ss = MultiProcess_LearnSafeTSRB(n_iterations, l_episodes, n_episodes, n_steps, n_states, n_arms, n_choices, thresholds,
    #                                                                                             transition_type, transition_increasing, method, reward_bandits, transition_bandits, initial_states,
    #                                                                                             u_type, u_order, True, max_wi)
    #     else:
    #         sumwis_l, rew_l, obj_l, swi_ss, rew_ss, obj_ss = MultiProcess_LearnSafeRandomTSRB(n_priors, n_iterations, l_episodes, n_episodes, n_steps, n_states, n_arms, n_choices, thresholds,
    #                                                                                           transition_type, transition_increasing, method, reward_bandits, initial_states, u_type, u_order, True,
    #                                                                                           max_wi)
    # else:
    #     if exp_type == 'det':
    #         sumwis_l, rew_l, obj_l, swi_ss, rew_ss, obj_ss = Process_LearnSoftSafeTSRB(n_iterations, l_episodes, n_episodes, n_steps, n_states, n_arms, n_choices, thresholds,
    #                                                                                    transition_type, transition_increasing, method, reward_bandits, transition_bandits,
    #                                                                                    initial_states, u_type, u_order, True, max_wi)
    #     else:
    #         sumwis_l, rew_l, obj_l, swi_ss, rew_ss, obj_ss = Process_LearnSoftSafeTSRBRandom(n_iterations, l_episodes, n_episodes, n_steps, n_states, n_arms, n_choices, thresholds,
    #                                                                                          transition_type, transition_increasing, method, reward_bandits,
    #                                                                                          initial_states, u_type, u_order, True, max_wi)
    #     # rew_ss, obj_ss, _ = Process_SoftSafeRB(SafeW, n_episodes, n_steps, n_states, n_arms, n_choices, thresholds, reward_bandits, transition_bandits,
    #     #                                        sw_bandits, initial_states, u_type, u_order)
    #     # probs_l, sumwis_l, rew_l, obj_l = Process_SafeSoftTSRB(n_iterations, l_episodes, n_episodes, n_steps, n_states, n_arms, n_choices, thresholds,
    #     #                                                        transition_type, transition_increasing, method, reward_bandits, transition_bandits,
    #     #                                                        initial_states, u_type, u_order, True, max_wi)
    #     # learn_list = joblib.load(f'./output/safesofttsrb_{n_steps}{n_states}{n_arms}{tt}{u_type}{n_choices}{thresholds[0]}.joblib')
    #     # probs_l = learn_list[0]
    #     # sumwis_l = learn_list[1]
    #     # rew_l = learn_list[2]
    #     # obj_l = learn_list[3]
    #
    # trn_err = np.mean(transerror_l, axis=(0, 2))
    # plt.figure(figsize=(8, 6))
    # plt.plot(trn_err, linewidth=4)
    # plt.xlabel('Episodes', fontsize=14, fontweight='bold')
    # plt.ylabel('Max Transition Error', fontsize=14, fontweight='bold')
    # plt.xticks(fontsize=12, fontweight='bold')
    # plt.yticks(fontsize=12, fontweight='bold')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    #
    # wis_err = np.mean(wierrors_l, axis=(0, 2))
    # plt.figure(figsize=(8, 6))
    # plt.plot(wis_err, linewidth=4)
    # plt.xlabel('Episodes', fontsize=14, fontweight='bold')
    # plt.ylabel('Max WI Error', fontsize=14, fontweight='bold')
    # plt.xticks(fontsize=12, fontweight='bold')
    # plt.yticks(fontsize=12, fontweight='bold')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    #
    # reg_obj = obj_ss - obj_l
    # reg = np.cumsum(np.mean(reg_obj, axis=(0, 2)))
    # plt.figure(figsize=(8, 6))
    # plt.plot(reg, linewidth=8)
    # plt.xlabel('Episodes', fontsize=14, fontweight='bold')
    # plt.ylabel('Regret', fontsize=14, fontweight='bold')
    # plt.xticks(fontsize=12, fontweight='bold')
    # plt.yticks(fontsize=12, fontweight='bold')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    #
    # # mean_reg = np.mean(reg, axis=0)
    # #
    # # # Calculate upper and lower bounds
    # # upper_bound = np.max(reg, axis=0)
    # # lower_bound = np.min(reg, axis=0)
    #
    # # Plotting
    # plt.figure(figsize=(8, 6))
    # plt.plot([reg[t]/(t+1) for t in range(len(reg))], linewidth=8)
    # plt.xlabel('Episodes', fontsize=14, fontweight='bold')
    # plt.ylabel('Regret/T', fontsize=14, fontweight='bold')
    # plt.xticks(fontsize=12, fontweight='bold')
    # plt.yticks(fontsize=12, fontweight='bold')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    # # plt.savefig(f'./output/regret_{n_steps}{n_states}{n_arms}{tt}{u_type}{u_order}{n_choices}{thresholds[0]}.png')
    # # plt.savefig(f'./output/regret_{n_steps}{n_states}{n_arms}{tt}{u_type}{u_order}{n_choices}{thresholds[0]}.jpg')
    #
    # wip_obj = np.mean(np.sum(obj_ss, axis=2), axis=0)
    # lrp_obj = np.mean(np.sum(obj_l, axis=2), axis=0)
    # wip_out = [sum(wip_obj[:t]) / t for t in range(1, 1 + len(wip_obj))]
    # lrp_out = [sum(lrp_obj[:t]) / t for t in range(1, 1 + len(lrp_obj))]
    #
    # # plt.figure(figsize=(8, 6))
    # # plt.plot(lrp_out, label='Learner', color='blue', linewidth=4)
    # # plt.plot(wip_out, label='Oracle', color='black', linestyle='--', linewidth=4)
    # # # plt.axhline(y=wip_obj, label='Oracle', color='black', linestyle='--', linewidth=4)
    # # plt.xlabel('Episodes', fontsize=14, fontweight='bold')
    # # plt.ylabel('Objective', fontsize=14, fontweight='bold')
    # # plt.xticks(fontsize=12, fontweight='bold')
    # # plt.yticks(fontsize=12, fontweight='bold')
    # # plt.legend(prop={'weight':'bold', 'size':12})
    # # plt.grid(True)
    # # plt.show()
    #
    # # plt.figure(figsize=(8, 6))
    # # plt.plot(lrp_out, label='Learner', color='blue', linewidth=4)
    # # plt.plot(np.mean(wip_obj)*np.ones(len(lrp_out)), label='Oracle', color='black', linestyle='--', linewidth=4)
    # # # plt.axhline(y=wip_obj, label='Oracle', color='black', linestyle='--', linewidth=4)
    # # plt.xlabel('Episodes', fontsize=14, fontweight='bold')
    # # plt.ylabel('Objective', fontsize=14, fontweight='bold')
    # # plt.xticks(fontsize=12, fontweight='bold')
    # # plt.yticks(fontsize=12, fontweight='bold')
    # # plt.legend(prop={'weight': 'bold', 'size': 12})
    # # plt.grid(True)
    # # plt.show()
