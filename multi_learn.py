from processes import *
from learning import *
from Markov import *
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt


if __name__ == '__main__':

    # Basic Parameters
    n_steps_set = [4]
    n_states_set = [3, 2]
    n_arms_set = [4, 3]
    f_type_set = ['hom']
    t_type_set = [3]
    u_type_set = [1, 2]
    u_order_set = [4, 8]
    threshold_set = [0.4, 0.5]
    fraction_set = [0.3, 0.4]

    PATH1 = f'./output/TestRes_t{n_steps_set[-1]}.xlsx'
    PATH2 = f'./output/TestRes_t{n_steps_set[-1]}_avg.xlsx'
    PATH3 = f'./output/'

    method = 3
    n_episodes = 100
    n_iterations = 10
    np.random.seed(42)

    count = 0
    total = len(n_steps_set) * len(n_arms_set) * len(n_states_set) * len(f_type_set) * len(t_type_set) * len(u_type_set) * len(u_type_set) * len(fraction_set) * len(threshold_set)
    for nt in n_steps_set:
        for ns in n_states_set:
            for nc in n_arms_set:
                na = nc * ns
                for ft_type in f_type_set:
                    if ft_type == 'hom':
                        ft = np.ones(na, dtype=np.int32)
                    else:
                        ft = 1 + np.arange(na)
                    for tt in t_type_set:
                        if tt == 0:
                            prob_remain = np.round(np.linspace(0.1, 0.9, na), 2)
                        elif tt == 1:
                            prob_remain = np.round(np.linspace(0.05, 0.45, na), 2)
                        elif tt == 2:
                            prob_remain = np.round(np.linspace(0.05, 0.45, na), 2)
                        elif tt == 3:
                            prob_remain = np.round(np.linspace(0.1 / ns, 1 / ns, na), 2)
                        elif tt == 4:
                            prob_remain = np.round(np.linspace(0.1 / ns, 1 / ns, na), 2)
                        elif tt == 5:
                            prob_remain = np.round(np.linspace(0.1 / ns, 1 / ns, na), 2)
                        elif tt == 6:
                            prob_remain = np.round(np.linspace(0.2, 0.8, na), 2)
                        else:
                            prob_remain = np.round(np.linspace(0.1, 0.9, na), 2)

                        np.random.shuffle(ft)
                        np.random.shuffle(prob_remain)
                        R = Values(nt, na, ns, ft, True)
                        M = MarkovDynamics(na, ns, prob_remain, tt, True)
                        max_wi = 1

                        for ut in u_type_set:
                            for uo in u_order_set:
                                for th in threshold_set:

                                    thresh = th * np.ones(na)
                                    SafeW = SafeWhittle(ns, na, R.vals, M.transitions, nt, ut, uo, thresh)
                                    SafeW.get_whittle_indices(computation_type=method, params=[0, max_wi], n_trials=nt * na * ns)
                                    sw_indices = SafeW.w_indices

                                    for fr in fraction_set:
                                        nch = np.maximum(1, int(np.around(fr * na)))
                                        initial_states = (ns - 1) * np.ones(na, dtype=np.int32)

                                        # rew_n, obj_n, _ = Process_Myopic(n_episodes, nt, ns, na, nch, thresh, R.vals, M.transitions, initial_states, ut)
                                        # rew_w, obj_w, _ = Process_WhtlRB(WhtlW, n_episodes, nt, ns, na, nch, thresh, R.vals, M.transitions, ww_indices, initial_states, ut)
                                        # rew_n, obj_n, _ = Process_NeutRB(NeutW, n_episodes, nt, ns, na, nch, thresh, R.vals, M.transitions, nw_indices, initial_states, ut)
                                        rew_s, obj_s, _ = Process_SafeRB(SafeW, n_episodes, nt, ns, na, nch, thresh, R.vals, M.transitions, sw_indices, initial_states, ut, uo)
                                        rew_l, obj_l, _, _, _ = Process_SafeTSRB(n_iterations, n_episodes, nt, ns, na, nch, thresh, tt, True, method, R.vals, M.transitions,
                                                                                 initial_states, ut, uo, False)

                                        key_value = f'nt{nt}_nc{nc}_ns{ns}_{ft_type}_tt{tt}_ut{ut}_uo{uo}_th{th}_fr{fr}'
                                        joblib.dump([rew_l, obj_l], PATH3 + key_value + "_Learning.joblib")
                                        joblib.dump([rew_s, obj_s], PATH3 + key_value + "_SafatyLr.joblib")

                                        count += 1
                                        print(f"{count} / {total}: {key_value}")

                                        reg = np.cumsum(np.mean(obj_s, axis=0) - np.mean(obj_l, axis=1), axis=1)
                                        mean_reg = np.mean(reg, axis=0)
                                        # Calculate upper and lower bounds
                                        upper_bound = np.max(reg, axis=0)
                                        lower_bound = np.min(reg, axis=0)
                                        # Plotting
                                        plt.figure(figsize=(4, 3))
                                        plt.plot(mean_reg, label='Mean')
                                        # Fill between lower bound and upper bound
                                        plt.fill_between(range(len(mean_reg)), lower_bound, upper_bound, color='skyblue', alpha=0.4, label='Bounds')

                                        plt.xlabel('Episodes')
                                        plt.ylabel('Regret')
                                        plt.title(f'Mean and Bounds over regret {key_value}')
                                        plt.legend()
                                        plt.grid(True)
                                        plt.show()
                                        plt.savefig(f'./output/regret_{nt}{ns}{na}{tt}{ut}{uo}{int(10 * nch)}{int(10 * th)}.png')
                                        plt.savefig(f'./output/regret_{nt}{ns}{na}{tt}{ut}{uo}{int(10 * nch)}{int(10 * th)}.jpg')
