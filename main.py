from whittle import *
from processes import *
from learning import *
import matplotlib.pyplot as plt
import joblib


if __name__ == '__main__':

    # Basic Parameters
    n_steps = 5
    n_states = 3
    n_arms = 2
    n_coeff = 1
    u_type = 1
    u_order = 16
    thresholds = 0.5 * np.ones(n_arms)
    choice_fraction = 0.5

    transition_type = 3
    function_type = np.ones(n_arms, dtype=np.int32)
    # function_type = 1 + np.arange(n_arms)
    # np.random.shuffle(function_type)

    n_episodes = 100
    # np.random.seed(42)

    na = n_arms
    ns = n_states
    tt = transition_type
    prob_remain = np.round(np.linspace(0.1 / ns, 0.1 / ns, na), 2)
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

    rb_type = 'hard'  # 'hard' or 'soft'
    n_iterations = 1
    l_episodes = 100
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

    wip_obj = np.mean(np.sum(obj_ss, axis=2))
    lrp_obj = np.mean(np.sum(obj_l, axis=2), axis=0)
    lrp_out = [sum(lrp_obj[:t]) / t for t in range(1, 1 + len(lrp_obj))]
    plt.figure(figsize=(8, 6))
    plt.plot(lrp_out, label='Learning Policy', color='blue')
    plt.axhline(y=wip_obj, label='Risk Aware Whittle Index Policy', color='black', linestyle='--')
    plt.xlabel('Learning Episodes')
    plt.ylabel('Average Performance')
    plt.legend()
    plt.grid(True)
    plt.show()
