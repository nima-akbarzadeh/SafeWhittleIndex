from processes import *
from whittle import *
from safe_whittle import *
from Markov import *
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import time


if __name__ == '__main__':

    # Basic Parameters

    n_steps_set = [4, 5]
    n_states_set = [2, 3]
    n_arms_set = [3, 4, 5]
    f_type_set = ['hom']
    t_type_set = [3]
    u_type_set = [2, 4, 8]
    threshold_set = [0.2, 0.3, 0.4, 0.5]
    fraction_set = [0.2, 0.3, 0.4, 0.5]

    PATH1 = f'TestRes_t{n_steps_set[-1]}.xlsx'
    PATH2 = f'TestRes_t{n_steps_set[-1]}_avg.xlsx'

    method = 3
    n_episodes = 100

    results1 = {}
    results2 = {}
    results3 = {}
    results4 = {}
    results5 = {}
    results6 = {}
    results7 = {}
    results8 = {}
    results9 = {}
    results0 = {}
    res1 = {}
    res2 = {}
    res3 = {}
    res4 = {}
    res5 = {}
    res6 = {}
    res7 = {}
    res8 = {}
    res9 = {}
    res0 = {}
    mean_neut = {}
    mean_safe = {}
    mean_impr = {}
    mean_rela = {}
    cvar_neut = {}
    cvar_safe = {}
    cvar_impr = {}
    cvar_rela = {}
    mean_rel2 = {}
    cvar_rel2 = {}

    for nt in n_steps_set:
        res1[f'n_steps_set_{nt}'] = []
        res2[f'n_steps_set_{nt}'] = []
        res3[f'n_steps_set_{nt}'] = []
        res4[f'n_steps_set_{nt}'] = []
        res5[f'n_steps_set_{nt}'] = []
        res6[f'n_steps_set_{nt}'] = []
        res7[f'n_steps_set_{nt}'] = []
        res8[f'n_steps_set_{nt}'] = []
        res9[f'n_steps_set_{nt}'] = []
        res0[f'n_steps_set_{nt}'] = []
        mean_neut[f'n_steps_set_{nt}'] = 0
        mean_safe[f'n_steps_set_{nt}'] = 0
        mean_impr[f'n_steps_set_{nt}'] = 0
        mean_rela[f'n_steps_set_{nt}'] = 0
        cvar_neut[f'n_steps_set_{nt}'] = 0
        cvar_safe[f'n_steps_set_{nt}'] = 0
        cvar_impr[f'n_steps_set_{nt}'] = 0
        cvar_rela[f'n_steps_set_{nt}'] = 0
        mean_rel2[f'n_steps_set_{nt}'] = 0
        cvar_rel2[f'n_steps_set_{nt}'] = 0
    for na in n_arms_set:
        res1[f'n_arms_set_{na}'] = []
        res2[f'n_arms_set_{na}'] = []
        res3[f'n_arms_set_{na}'] = []
        res4[f'n_arms_set_{na}'] = []
        res5[f'n_arms_set_{na}'] = []
        res6[f'n_arms_set_{na}'] = []
        res7[f'n_arms_set_{na}'] = []
        res8[f'n_arms_set_{na}'] = []
        res9[f'n_arms_set_{na}'] = []
        res0[f'n_arms_set_{na}'] = []
        mean_neut[f'n_arms_set_{na}'] = 0
        mean_safe[f'n_arms_set_{na}'] = 0
        mean_impr[f'n_arms_set_{na}'] = 0
        mean_rela[f'n_arms_set_{na}'] = 0
        cvar_neut[f'n_arms_set_{na}'] = 0
        cvar_safe[f'n_arms_set_{na}'] = 0
        cvar_impr[f'n_arms_set_{na}'] = 0
        cvar_rela[f'n_arms_set_{na}'] = 0
        mean_rel2[f'n_arms_set_{na}'] = 0
        cvar_rel2[f'n_arms_set_{na}'] = 0
    for ns in n_states_set:
        res1[f'n_states_set_{ns}'] = []
        res2[f'n_states_set_{ns}'] = []
        res3[f'n_states_set_{ns}'] = []
        res4[f'n_states_set_{ns}'] = []
        res5[f'n_states_set_{ns}'] = []
        res6[f'n_states_set_{ns}'] = []
        res7[f'n_states_set_{ns}'] = []
        res8[f'n_states_set_{ns}'] = []
        res9[f'n_states_set_{ns}'] = []
        res0[f'n_states_set_{ns}'] = []
        mean_neut[f'n_states_set_{ns}'] = 0
        mean_safe[f'n_states_set_{ns}'] = 0
        mean_impr[f'n_states_set_{ns}'] = 0
        mean_rela[f'n_states_set_{ns}'] = 0
        cvar_neut[f'n_states_set_{ns}'] = 0
        cvar_safe[f'n_states_set_{ns}'] = 0
        cvar_impr[f'n_states_set_{ns}'] = 0
        cvar_rela[f'n_states_set_{ns}'] = 0
        mean_rel2[f'n_states_set_{ns}'] = 0
        cvar_rel2[f'n_states_set_{ns}'] = 0
    for ft in f_type_set:
        res1[f'f_type_set_{ft}'] = []
        res2[f'f_type_set_{ft}'] = []
        res3[f'f_type_set_{ft}'] = []
        res4[f'f_type_set_{ft}'] = []
        res5[f'f_type_set_{ft}'] = []
        res6[f'f_type_set_{ft}'] = []
        res7[f'f_type_set_{ft}'] = []
        res8[f'f_type_set_{ft}'] = []
        res9[f'f_type_set_{ft}'] = []
        res0[f'f_type_set_{ft}'] = []
        mean_neut[f'f_type_set_{ft}'] = 0
        mean_safe[f'f_type_set_{ft}'] = 0
        mean_impr[f'f_type_set_{ft}'] = 0
        mean_rela[f'f_type_set_{ft}'] = 0
        cvar_neut[f'f_type_set_{ft}'] = 0
        cvar_safe[f'f_type_set_{ft}'] = 0
        cvar_impr[f'f_type_set_{ft}'] = 0
        cvar_rela[f'f_type_set_{ft}'] = 0
        mean_rel2[f'f_type_set_{ft}'] = 0
        cvar_rel2[f'f_type_set_{ft}'] = 0
    for tt in t_type_set:
        res1[f't_type_set_{tt}'] = []
        res2[f't_type_set_{tt}'] = []
        res3[f't_type_set_{tt}'] = []
        res4[f't_type_set_{tt}'] = []
        res5[f't_type_set_{tt}'] = []
        res6[f't_type_set_{tt}'] = []
        res7[f't_type_set_{tt}'] = []
        res8[f't_type_set_{tt}'] = []
        res9[f't_type_set_{tt}'] = []
        res0[f't_type_set_{tt}'] = []
        mean_neut[f't_type_set_{tt}'] = 0
        mean_safe[f't_type_set_{tt}'] = 0
        mean_impr[f't_type_set_{tt}'] = 0
        mean_rela[f't_type_set_{tt}'] = 0
        cvar_neut[f't_type_set_{tt}'] = 0
        cvar_safe[f't_type_set_{tt}'] = 0
        cvar_impr[f't_type_set_{tt}'] = 0
        cvar_rela[f't_type_set_{tt}'] = 0
        mean_rel2[f't_type_set_{tt}'] = 0
        cvar_rel2[f't_type_set_{tt}'] = 0
    for ut in u_type_set:
        res1[f'u_type_set_{ut}'] = []
        res2[f'u_type_set_{ut}'] = []
        res3[f'u_type_set_{ut}'] = []
        res4[f'u_type_set_{ut}'] = []
        res5[f'u_type_set_{ut}'] = []
        res6[f'u_type_set_{ut}'] = []
        res7[f'u_type_set_{ut}'] = []
        res8[f'u_type_set_{ut}'] = []
        res9[f'u_type_set_{ut}'] = []
        res0[f'u_type_set_{ut}'] = []
        mean_neut[f'u_type_set_{ut}'] = 0
        mean_safe[f'u_type_set_{ut}'] = 0
        mean_impr[f'u_type_set_{ut}'] = 0
        mean_rela[f'u_type_set_{ut}'] = 0
        cvar_neut[f'u_type_set_{ut}'] = 0
        cvar_safe[f'u_type_set_{ut}'] = 0
        cvar_impr[f'u_type_set_{ut}'] = 0
        cvar_rela[f'u_type_set_{ut}'] = 0
        mean_rel2[f'u_type_set_{ut}'] = 0
        cvar_rel2[f'u_type_set_{ut}'] = 0
    for fr in fraction_set:
        res1[f'fraction_set_{fr}'] = []
        res2[f'fraction_set_{fr}'] = []
        res3[f'fraction_set_{fr}'] = []
        res4[f'fraction_set_{fr}'] = []
        res5[f'fraction_set_{fr}'] = []
        res6[f'fraction_set_{fr}'] = []
        res7[f'fraction_set_{fr}'] = []
        res8[f'fraction_set_{fr}'] = []
        res9[f'fraction_set_{fr}'] = []
        res0[f'fraction_set_{fr}'] = []
        mean_neut[f'fraction_set_{fr}'] = 0
        mean_safe[f'fraction_set_{fr}'] = 0
        mean_impr[f'fraction_set_{fr}'] = 0
        mean_rela[f'fraction_set_{fr}'] = 0
        cvar_neut[f'fraction_set_{fr}'] = 0
        cvar_safe[f'fraction_set_{fr}'] = 0
        cvar_impr[f'fraction_set_{fr}'] = 0
        cvar_rela[f'fraction_set_{fr}'] = 0
        mean_rel2[f'fraction_set_{fr}'] = 0
        cvar_rel2[f'fraction_set_{fr}'] = 0
    for th in threshold_set:
        res1[f'threshold_set_{th}'] = []
        res2[f'threshold_set_{th}'] = []
        res3[f'threshold_set_{th}'] = []
        res4[f'threshold_set_{th}'] = []
        res5[f'threshold_set_{th}'] = []
        res6[f'threshold_set_{th}'] = []
        res7[f'threshold_set_{th}'] = []
        res8[f'threshold_set_{th}'] = []
        res9[f'threshold_set_{th}'] = []
        res0[f'threshold_set_{th}'] = []
        mean_neut[f'threshold_set_{th}'] = 0
        mean_safe[f'threshold_set_{th}'] = 0
        mean_impr[f'threshold_set_{th}'] = 0
        mean_rela[f'threshold_set_{th}'] = 0
        cvar_neut[f'threshold_set_{th}'] = 0
        cvar_safe[f'threshold_set_{th}'] = 0
        cvar_impr[f'threshold_set_{th}'] = 0
        cvar_rela[f'threshold_set_{th}'] = 0
        mean_rel2[f'threshold_set_{th}'] = 0
        cvar_rel2[f'threshold_set_{th}'] = 0

    count = 0
    total = len(n_steps_set) * len(n_arms_set) * len(n_states_set) * len(f_type_set) * len(t_type_set) * len(u_type_set) * len(fraction_set) * len(threshold_set)
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

                        WhtlW = Whittle(ns, na, R.vals, M.transitions, nt)
                        WhtlW.get_whittle_indices(computation_type=method, params=[0, max_wi], n_trials=nt * na * ns)
                        ww_indices = WhtlW.w_indices

                        for ut in u_type_set:

                            # NeutW = SafeWhittleV3(ns, na, R.vals, M.transitions, nt, ut, np.ones(na))
                            # NeutW.get_whittle_indices(computation_type=method, params=[0, max_wi], n_trials=nt*na*ns)
                            # nw_indices = NeutW.w_indices

                            for th in threshold_set:

                                thresh = th * np.ones(na)
                                SafeW = SafeWhittle(ns, na, R.vals, M.transitions, nt, ut, thresh)
                                SafeW.get_whittle_indices(computation_type=method, params=[0, max_wi], n_trials=nt * na * ns)
                                sw_indices = SafeW.w_indices

                                for fr in fraction_set:
                                    nch = np.maximum(1, int(np.around(fr * na)))

                                    initial_states = (ns - 1) * np.ones(na, dtype=np.int32)

                                    rew_n, obj_n, _ = Process_Greedy(n_episodes, nt, ns, na, nch, thresh, R.vals, M.transitions, initial_states, ut)
                                    rew_w, obj_w, _ = Process_WhtlRB(WhtlW, n_episodes, nt, ns, na, nch, thresh, R.vals, M.transitions, ww_indices, initial_states, ut)
                                    # rew_n, obj_n, _ = Process_NeutRB(NeutW, n_episodes, nt, ns, na, nch, thresh, R.vals, M.transitions, nw_indices, initial_states, ut)
                                    rew_s, obj_s, _ = Process_SafeRB(SafeW, n_episodes, nt, ns, na, nch, thresh, R.vals, M.transitions, sw_indices, initial_states, ut)

                                    navg_vec = obj_n.mean(axis=0)
                                    wavg_vec = obj_w.mean(axis=0)
                                    savg_vec = obj_s.mean(axis=0)
                                    navg = np.round(np.mean(obj_n), 3)
                                    wavg = np.round(np.mean(obj_w), 3)
                                    savg = np.round(np.mean(obj_s), 3)
                                    impr_val = np.round(savg - wavg, 2)
                                    impr_prc = np.round(100 * (savg - wavg) / wavg, 2)
                                    impr_prc2 = np.round(100 * (savg - navg) / navg, 2)
                                    cvar_n = np.round(np.mean(np.sort(navg_vec)[:int(th * len(navg_vec))]), 3)
                                    cvar_w = np.round(np.mean(np.sort(wavg_vec)[:int(th * len(wavg_vec))]), 3)
                                    cvar_s = np.round(np.mean(np.sort(savg_vec)[:int(th * len(savg_vec))]), 3)
                                    cvar_imp = np.round(cvar_s - cvar_w, 2)
                                    cvar_prc = np.round(100 * (cvar_s - cvar_w) / cvar_w, 2)
                                    cvar_prc2 = np.round(100 * (cvar_s - cvar_n) / cvar_n, 2)
                                    key_value = f'nt{nt}_nc{nc}_ns{ns}_{ft_type}_tt{tt}_ut{ut}_th{th}_fr{fr}'
                                    count += 1
                                    print(f"{count} / {total}: {key_value} ---> MEAN-Relative: {impr_prc}, CVAR-Relative: {cvar_prc}, MEAN-Relative2: {impr_prc2}, CVAR-Relative2: {cvar_prc2}")
                                    results1[key_value] = wavg
                                    results2[key_value] = savg
                                    results3[key_value] = impr_val
                                    results4[key_value] = impr_prc
                                    results5[key_value] = cvar_w
                                    results6[key_value] = cvar_s
                                    results7[key_value] = cvar_imp
                                    results8[key_value] = cvar_prc
                                    results9[key_value] = impr_prc2
                                    results0[key_value] = cvar_prc2
                                    df1_1 = pd.DataFrame(list(results1.items()), columns=['Key', 'MEAN-Neut'])
                                    df1_2 = pd.DataFrame(results2.values(), columns=['MEAN-Safe'])
                                    df1_3 = pd.DataFrame(results3.values(), columns=['MEAN-Impr'])
                                    df1_4 = pd.DataFrame(results4.values(), columns=['MEAN-Rela'])
                                    df1_5 = pd.DataFrame(results5.values(), columns=['CVAR-Neut'])
                                    df1_6 = pd.DataFrame(results6.values(), columns=['CVAR-Safe'])
                                    df1_7 = pd.DataFrame(results7.values(), columns=['CVAR-Impr'])
                                    df1_8 = pd.DataFrame(results8.values(), columns=['CVAR-Rela'])
                                    df1_9 = pd.DataFrame(results9.values(), columns=['MEAN-Rel2'])
                                    df1_0 = pd.DataFrame(results0.values(), columns=['CVAR-Rel2'])
                                    df1 = pd.concat([df1_1, df1_2, df1_3, df1_4, df1_5, df1_6, df1_7, df1_8, df1_9, df1_0], axis=1)
                                    df1.to_excel(PATH1, index=False)

                                    res1[f'n_steps_set_{nt}'].append(results1[key_value])
                                    res1[f'n_arms_set_{nc}'].append(results1[key_value])
                                    res1[f'n_states_set_{ns}'].append(results1[key_value])
                                    res1[f'f_type_set_{ft_type}'].append(results1[key_value])
                                    res1[f't_type_set_{tt}'].append(results1[key_value])
                                    res1[f'u_type_set_{ut}'].append(results1[key_value])
                                    res1[f'fraction_set_{fr}'].append(results1[key_value])
                                    res1[f'threshold_set_{th}'].append(results1[key_value])
                                    res2[f'n_steps_set_{nt}'].append(results2[key_value])
                                    res2[f'n_arms_set_{nc}'].append(results2[key_value])
                                    res2[f'n_states_set_{ns}'].append(results2[key_value])
                                    res2[f'f_type_set_{ft_type}'].append(results2[key_value])
                                    res2[f't_type_set_{tt}'].append(results2[key_value])
                                    res2[f'u_type_set_{ut}'].append(results2[key_value])
                                    res2[f'fraction_set_{fr}'].append(results2[key_value])
                                    res2[f'threshold_set_{th}'].append(results2[key_value])
                                    res3[f'n_steps_set_{nt}'].append(results3[key_value])
                                    res3[f'n_arms_set_{nc}'].append(results3[key_value])
                                    res3[f'n_states_set_{ns}'].append(results3[key_value])
                                    res3[f'f_type_set_{ft_type}'].append(results3[key_value])
                                    res3[f't_type_set_{tt}'].append(results3[key_value])
                                    res3[f'u_type_set_{ut}'].append(results3[key_value])
                                    res3[f'fraction_set_{fr}'].append(results3[key_value])
                                    res3[f'threshold_set_{th}'].append(results3[key_value])
                                    res4[f'n_steps_set_{nt}'].append(results4[key_value])
                                    res4[f'n_arms_set_{nc}'].append(results4[key_value])
                                    res4[f'n_states_set_{ns}'].append(results4[key_value])
                                    res4[f'f_type_set_{ft_type}'].append(results4[key_value])
                                    res4[f't_type_set_{tt}'].append(results4[key_value])
                                    res4[f'u_type_set_{ut}'].append(results4[key_value])
                                    res4[f'fraction_set_{fr}'].append(results4[key_value])
                                    res4[f'threshold_set_{th}'].append(results4[key_value])
                                    res5[f'n_steps_set_{nt}'].append(results5[key_value])
                                    res5[f'n_arms_set_{nc}'].append(results5[key_value])
                                    res5[f'n_states_set_{ns}'].append(results5[key_value])
                                    res5[f'f_type_set_{ft_type}'].append(results5[key_value])
                                    res5[f't_type_set_{tt}'].append(results5[key_value])
                                    res5[f'u_type_set_{ut}'].append(results5[key_value])
                                    res5[f'fraction_set_{fr}'].append(results5[key_value])
                                    res5[f'threshold_set_{th}'].append(results5[key_value])
                                    res6[f'n_steps_set_{nt}'].append(results6[key_value])
                                    res6[f'n_arms_set_{nc}'].append(results6[key_value])
                                    res6[f'n_states_set_{ns}'].append(results6[key_value])
                                    res6[f'f_type_set_{ft_type}'].append(results6[key_value])
                                    res6[f't_type_set_{tt}'].append(results6[key_value])
                                    res6[f'u_type_set_{ut}'].append(results6[key_value])
                                    res6[f'fraction_set_{fr}'].append(results6[key_value])
                                    res6[f'threshold_set_{th}'].append(results6[key_value])
                                    res7[f'n_steps_set_{nt}'].append(results7[key_value])
                                    res7[f'n_arms_set_{nc}'].append(results7[key_value])
                                    res7[f'n_states_set_{ns}'].append(results7[key_value])
                                    res7[f'f_type_set_{ft_type}'].append(results7[key_value])
                                    res7[f't_type_set_{tt}'].append(results7[key_value])
                                    res7[f'u_type_set_{ut}'].append(results7[key_value])
                                    res7[f'fraction_set_{fr}'].append(results7[key_value])
                                    res7[f'threshold_set_{th}'].append(results7[key_value])
                                    res8[f'n_steps_set_{nt}'].append(results8[key_value])
                                    res8[f'n_arms_set_{nc}'].append(results8[key_value])
                                    res8[f'n_states_set_{ns}'].append(results8[key_value])
                                    res8[f'f_type_set_{ft_type}'].append(results8[key_value])
                                    res8[f't_type_set_{tt}'].append(results8[key_value])
                                    res8[f'u_type_set_{ut}'].append(results8[key_value])
                                    res8[f'fraction_set_{fr}'].append(results8[key_value])
                                    res8[f'threshold_set_{th}'].append(results8[key_value])
                                    res9[f'n_steps_set_{nt}'].append(results9[key_value])
                                    res9[f'n_arms_set_{nc}'].append(results9[key_value])
                                    res9[f'n_states_set_{ns}'].append(results9[key_value])
                                    res9[f'f_type_set_{ft_type}'].append(results9[key_value])
                                    res9[f't_type_set_{tt}'].append(results9[key_value])
                                    res9[f'u_type_set_{ut}'].append(results9[key_value])
                                    res9[f'fraction_set_{fr}'].append(results9[key_value])
                                    res9[f'threshold_set_{th}'].append(results9[key_value])
                                    res0[f'n_steps_set_{nt}'].append(results0[key_value])
                                    res0[f'n_arms_set_{nc}'].append(results0[key_value])
                                    res0[f'n_states_set_{ns}'].append(results0[key_value])
                                    res0[f'f_type_set_{ft_type}'].append(results0[key_value])
                                    res0[f't_type_set_{tt}'].append(results0[key_value])
                                    res0[f'u_type_set_{ut}'].append(results0[key_value])
                                    res0[f'fraction_set_{fr}'].append(results0[key_value])
                                    res0[f'threshold_set_{th}'].append(results0[key_value])

                                    for key in list(res1.keys()):
                                        if len(res1[key]) != 0:
                                            mean_neut[key] = sum(res1[key]) / len(res1[key])
                                        if len(res2[key]) != 0:
                                            mean_safe[key] = sum(res2[key]) / len(res2[key])
                                        if len(res3[key]) != 0:
                                            mean_impr[key] = sum(res3[key]) / len(res3[key])
                                        if len(res4[key]) != 0:
                                            mean_rela[key] = sum(res4[key]) / len(res4[key])
                                        if len(res5[key]) != 0:
                                            cvar_neut[key] = sum(res5[key]) / len(res5[key])
                                        if len(res6[key]) != 0:
                                            cvar_safe[key] = sum(res6[key]) / len(res6[key])
                                        if len(res7[key]) != 0:
                                            cvar_impr[key] = sum(res7[key]) / len(res7[key])
                                        if len(res8[key]) != 0:
                                            cvar_rela[key] = sum(res8[key]) / len(res8[key])
                                        if len(res9[key]) != 0:
                                            mean_rel2[key] = sum(res9[key]) / len(res9[key])
                                        if len(res0[key]) != 0:
                                            cvar_rel2[key] = sum(res0[key]) / len(res0[key])

                                    df2_1 = pd.DataFrame(list(mean_neut.items()), columns=['Key', 'MEAN-Neut'])
                                    df2_2 = pd.DataFrame(list(mean_safe.values()), columns=['MEAN-Safe'])
                                    df2_3 = pd.DataFrame(list(mean_impr.values()), columns=['MEAN-Impr'])
                                    df2_4 = pd.DataFrame(list(mean_rela.values()), columns=['MEAN-Rela'])
                                    df2_5 = pd.DataFrame(list(cvar_neut.values()), columns=['CVAR-Neut'])
                                    df2_6 = pd.DataFrame(list(cvar_safe.values()), columns=['CVAR-Safe'])
                                    df2_7 = pd.DataFrame(list(cvar_impr.values()), columns=['CVAR-Impr'])
                                    df2_8 = pd.DataFrame(list(cvar_rela.values()), columns=['CVAR-Rela'])
                                    df2_9 = pd.DataFrame(list(mean_rel2.values()), columns=['MEAN-Rel2'])
                                    df2_0 = pd.DataFrame(list(cvar_rel2.values()), columns=['CVAR-Rel2'])
                                    df2 = pd.concat([df2_1, df2_2, df2_3, df2_4, df2_5, df2_6, df2_7, df2_8, df2_9, df2_0], axis=1)
                                    df2.to_excel(PATH2, index=False)
