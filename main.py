from whittle import *
from processes import *
from learning import *
import matplotlib.pyplot as plt


if __name__ == '__main__':

    # Basic Parameters
    n_steps = 3
    n_coeff = 3
    n_states = 4
    u_type = 2
    u_order = 8
    n_arms = n_coeff * n_states
    thresholds = 0.3 * np.ones(n_arms)
    choice_fraction = 0.5
    n_episodes = 100
    n_iterations = 10

    function_type = np.ones(n_arms, dtype=np.int32)
    # function_type = 1 + np.arange(n_arms)

    transition_type = 11
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
        prob_remain = np.round(np.linspace(0.1 / ns, 1 / ns, na), 2)
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
    rew_m, obj_m, _ = Process_Greedy(n_episodes, n_steps, n_states, n_arms, n_choices, thresholds, reward_bandits, transition_bandits, initial_states, u_type, u_order)
    rew_w, obj_w, _ = Process_WhtlRB(W, n_episodes, n_steps, n_states, n_arms, n_choices, thresholds, reward_bandits, transition_bandits, w_bandits, initial_states, u_type, u_order)
    rew_s, obj_s, _ = Process_SafeRB(SafeW, n_episodes, n_steps, n_states, n_arms, n_choices, thresholds, reward_bandits, transition_bandits, sw_bandits, initial_states, u_type, u_order)
    print('Process Ends ...')

    print("====================== REWARDS =========================")
    print(f'Myopic: {np.mean(rew_m)}')
    print(f'Whittl: {np.mean(rew_w)}')
    print(f'Safety: {np.mean(rew_s)}')

    print("===================== OBJECTIVE ========================")
    print(f'Myopic: {np.mean(obj_m)}')
    print(f'Whittl: {np.mean(obj_w)}')
    print(f'Safety: {np.mean(obj_s)}')

    print("===================== IMPROVEMENT ========================")
    print(f'Safety-Whittl: {100 * (np.mean(obj_s) - np.mean(obj_w)) / np.mean(obj_w)}')
    print(f'Safety-Myopic: {100 * (np.mean(obj_s) - np.mean(obj_m)) / np.mean(obj_m)}')

    rew_l, obj_l, est_probs, counts, sum_wi = Process_SafeTSRB(n_iterations, n_episodes, n_steps, n_states, n_arms, n_choices, thresholds,
                                                               transition_type, transition_increasing, method, reward_bandits, transition_bandits,
                                                               initial_states, u_type, u_order, True)

    # prb_err = np.mean(prob_remain.mean()*np.ones(n_episodes) - est_probs.mean(axis=[0, 1]))
    # plt.figure(figsize=(8, 6))
    # plt.plot(prb_err, label='Mean')
    # plt.xlabel('Episodes')
    # plt.ylabel('Regret')
    # plt.title('Mean and Bounds over regret')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    # obj_l = joblib.load(f"obj_safetsrb_{n_steps}{n_states}{n_arms}{transition_type}{u_type}{u_order}{n_choices}{thresholds[0]}.joblib")

    reg = np.cumsum(np.mean(obj_s, axis=0) - np.mean(obj_l, axis=1), axis=1)

    mean_reg = np.mean(reg, axis=0)

    # Calculate upper and lower bounds
    upper_bound = np.max(reg, axis=0)
    lower_bound = np.min(reg, axis=0)

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(mean_reg, label='Mean')

    # Fill between lower bound and upper bound
    plt.fill_between(range(len(mean_reg)), lower_bound, upper_bound, color='skyblue', alpha=0.4, label='Bounds')

    plt.xlabel('Episodes')
    plt.ylabel('Regret')
    plt.title('Mean and Bounds over regret')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig(f'./output/regret_{n_steps}{n_states}{n_arms}{tt}{u_type}{u_order}{n_choices}{thresholds[0]}.png')
    plt.savefig(f'./output/regret_{n_steps}{n_states}{n_arms}{tt}{u_type}{u_order}{n_choices}{thresholds[0]}.jpg')
