from processes import *
from learning import *
from Markov import *
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt


if __name__ == '__main__':

    # Basic Parameters
    nt = 5
    n_states_set = [2, 3]
    n_armscoef_set = [1, 2]
    f_type_set = ['hom']
    t_type_set = [3]
    u_type_set = [1, 2]
    u_order_set = [1, 16]
    threshold_set = [0.5]
    fraction_set = [0.1, 0.3, 0.5]

    method = 3
    l_episodes = 25
    n_episodes = 100
    n_iterations = 20
    # np.random.seed(42)

    count = 0
    total = len(n_armscoef_set) * len(n_states_set) * len(f_type_set) * len(t_type_set) * len(u_type_set) \
            * len(u_order_set) * len(fraction_set) * len(threshold_set)
    for ns in n_states_set:
        for nc in n_armscoef_set:
            na = nc * ns
            for ft_type in f_type_set:
                if ft_type == 'hom':
                    ft = np.ones(na, dtype=np.int32)
                else:
                    ft = 1 + np.arange(na)
                    # np.random.shuffle(ft)
                for tt in t_type_set:
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
                        prob_remain = np.round(np.linspace(0.1 / ns, 0.1 / ns, na), 2)
                        # np.random.shuffle(prob_remain)
                    elif tt == 4:
                        prob_remain = np.round(np.linspace(0.1 / ns, 1 / ns, na), 2)
                        np.random.shuffle(prob_remain)
                    elif tt == 5:
                        prob_remain = np.round(np.linspace(0.1 / ns, 1 / ns, na), 2)
                        np.random.shuffle(prob_remain)
                    elif tt == 6:
                        prob_remain = np.round(np.linspace(0.2, 0.8, na), 2)
                        np.random.shuffle(prob_remain)
                    else:
                        prob_remain = np.round(np.linspace(0.1, 0.9, na), 2)
                        np.random.shuffle(prob_remain)

                    R = Values(nt, na, ns, ft, True)
                    M = MarkovDynamics(na, ns, prob_remain, tt, True)
                    max_wi = 1

                    for ut in u_type_set:
                        for uo in u_order_set:
                            for th in threshold_set:

                                thresh = th * np.ones(na)
                                SafeW = SafeWhittleAvg(ns, na, R.vals, M.transitions, ut, uo, thresh)
                                SafeW.get_whittle_indices(
                                    computation_type=method, params=[0, max_wi], n_trials=nt*na*ns
                                )
                                sw_indices = SafeW.w_indices

                                for fr in fraction_set:
                                    nch = np.maximum(1, int(np.around(fr * na)))
                                    initial_states = (ns - 1) * np.ones(na, dtype=np.int32)

                                    probs_l, sumwis_l, rew_l, obj_l, swi_s, rew_s, obj_s = \
                                        ProcessAvg_LearnSafeTSRB(n_iterations, l_episodes, n_episodes, nt, ns, na, nch,
                                                                 thresh, tt, True, method, R.vals, M.transitions,
                                                                 initial_states, ut, uo, False, max_wi)

                                    key_value = f'nt{nt}_ns{ns}_nc{nc}_{ft_type}_tt{tt}_ut{ut}_uo{uo}_th{th}_fr{fr}'

                                    count += 1
                                    print(f"{count} / {total}: {key_value}")

                                    wip_obj = np.mean(np.sum(obj_s, axis=2))
                                    lrp_obj = np.mean(np.sum(obj_l, axis=2), axis=0)
                                    lrp_out = [sum(lrp_obj[:t]) / t for t in range(1, 1 + len(lrp_obj))]
                                    plt.figure(figsize=(8, 6))
                                    plt.plot(lrp_out, label='Learning Policy', color='blue')
                                    plt.axhline(y=wip_obj, label='Risk Aware Whittle Index Policy', color='black',
                                                linestyle='--')
                                    plt.xlabel('Learning Episodes')
                                    plt.ylabel('Average Performance')
                                    plt.title(f'{key_value}')
                                    plt.legend()
                                    plt.grid(True)
                                    plt.savefig(f'./output/performance/{nt}{ns}{na}{tt}{ut}{uo}{nch}{int(10 * th)}_'
                                                f'performance.png')

                                    prb_err = np.abs(
                                        np.transpose(np.array([prob_remain[a] - np.mean(probs_l[:, :, a], axis=0)
                                                               for a in range(na)]))
                                    )
                                    plt.figure(figsize=(8, 6))
                                    plt.plot(prb_err)
                                    plt.xlabel('Learning Episodes')
                                    plt.ylabel('Parameter Error')
                                    plt.title(f'{key_value}')
                                    plt.legend()
                                    plt.grid(True)
                                    plt.savefig(f'./output/proberrors/{nt}{ns}{na}{tt}{ut}{uo}{nch}{int(10 * th)}_'
                                                f'proberrors.png')

                                    wip_arms = np.mean(obj_s, axis=(0, 1))
                                    lrp_arms = np.mean(obj_l, axis=0)
                                    lrp_arm1 = [sum(lrp_arms[:t, 0]) / t for t in range(1, 1 + lrp_obj.shape[0])]
                                    lrp_arm2 = [sum(lrp_arms[:t, 1]) / t for t in range(1, 1 + lrp_obj.shape[0])]
                                    plt.figure(figsize=(8, 6))
                                    plt.plot(lrp_arm1, label='Learning Policy of Arm 1', color='blue')
                                    plt.plot(lrp_arm2, label='Learning Policy of Arm 2', color='black')
                                    plt.axhline(y=wip_arms[0], label='Risk Aware Whittle Index Policy of Arm 1',
                                                color='blue', linestyle='--')
                                    plt.axhline(y=wip_arms[1], label='Risk Aware Whittle Index Policy of Arm 2',
                                                color='black', linestyle='--')
                                    plt.xlabel('Learning Episodes')
                                    plt.ylabel('Parameter Error')
                                    plt.title(f'{key_value}')
                                    plt.grid(True)
                                    plt.savefig(f'./output/armperformances/{nt}{ns}{na}{tt}{ut}{uo}{nch}{int(10 * th)}_'
                                                f'armperformances.png')
