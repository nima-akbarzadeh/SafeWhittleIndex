from processes import *
from whittle import *
from safe_whittle import *
from Markov import *
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':

    # Basic Parameters
    n_steps_set = [5, 4, 3]
    n_states_set = [4, 3, 2]
    armcoef_set = [4, 3]
    f_type_set = ['hom']
    t_type_set = [3]
    u_type_set = [1, 2]
    u_order_set = [4, 8, 16]
    threshold_set = [0.3, 0.4, 0.5]
    fraction_set = [0.2, 0.3, 0.4, 0.5]

    PATH1 = f'TestRes_t{n_steps_set}.xlsx'
    PATH2 = f'TestRes_t{n_steps_set}_avg.xlsx'
    PATH3 = f'./output/'

    method = 3
    n_episodes = 100

    results1 = {}
    results2 = {}
    results3 = {}
    results4 = {}
    results5 = {}
    results6 = {}
    res1 = {}
    res2 = {}
    res3 = {}
    res4 = {}
    res5 = {}
    res6 = {}
    mean_neut = {}
    mean_safe = {}
    mean_impr = {}
    mean_relw = {}
    mean_relm = {}
    mean_relr = {}

    for nt in n_steps_set:
        res1[f'n_steps_set_{nt}'] = []
        res2[f'n_steps_set_{nt}'] = []
        res3[f'n_steps_set_{nt}'] = []
        res4[f'n_steps_set_{nt}'] = []
        res5[f'n_steps_set_{nt}'] = []
        res6[f'n_steps_set_{nt}'] = []
        mean_neut[f'n_steps_set_{nt}'] = 0
        mean_safe[f'n_steps_set_{nt}'] = 0
        mean_impr[f'n_steps_set_{nt}'] = 0
        mean_relw[f'n_steps_set_{nt}'] = 0
        mean_relm[f'n_steps_set_{nt}'] = 0
        mean_relr[f'n_steps_set_{nt}'] = 0
    for na in armcoef_set:
        res1[f'armcoef_set_{na}'] = []
        res2[f'armcoef_set_{na}'] = []
        res3[f'armcoef_set_{na}'] = []
        res4[f'armcoef_set_{na}'] = []
        res5[f'armcoef_set_{na}'] = []
        res6[f'armcoef_set_{na}'] = []
        mean_neut[f'armcoef_set_{na}'] = 0
        mean_safe[f'armcoef_set_{na}'] = 0
        mean_impr[f'armcoef_set_{na}'] = 0
        mean_relw[f'armcoef_set_{na}'] = 0
        mean_relm[f'armcoef_set_{na}'] = 0
        mean_relr[f'armcoef_set_{na}'] = 0
    for ns in n_states_set:
        res1[f'n_states_set_{ns}'] = []
        res2[f'n_states_set_{ns}'] = []
        res3[f'n_states_set_{ns}'] = []
        res4[f'n_states_set_{ns}'] = []
        res5[f'n_states_set_{ns}'] = []
        res6[f'n_states_set_{ns}'] = []
        mean_neut[f'n_states_set_{ns}'] = 0
        mean_safe[f'n_states_set_{ns}'] = 0
        mean_impr[f'n_states_set_{ns}'] = 0
        mean_relw[f'n_states_set_{ns}'] = 0
        mean_relm[f'n_states_set_{ns}'] = 0
        mean_relr[f'n_states_set_{ns}'] = 0
    for ft in f_type_set:
        res1[f'f_type_set_{ft}'] = []
        res2[f'f_type_set_{ft}'] = []
        res3[f'f_type_set_{ft}'] = []
        res4[f'f_type_set_{ft}'] = []
        res5[f'f_type_set_{ft}'] = []
        res6[f'f_type_set_{ft}'] = []
        mean_neut[f'f_type_set_{ft}'] = 0
        mean_safe[f'f_type_set_{ft}'] = 0
        mean_impr[f'f_type_set_{ft}'] = 0
        mean_relw[f'f_type_set_{ft}'] = 0
        mean_relm[f'f_type_set_{ft}'] = 0
        mean_relr[f'f_type_set_{ft}'] = 0
    for tt in t_type_set:
        res1[f't_type_set_{tt}'] = []
        res2[f't_type_set_{tt}'] = []
        res3[f't_type_set_{tt}'] = []
        res4[f't_type_set_{tt}'] = []
        res5[f't_type_set_{tt}'] = []
        res6[f't_type_set_{tt}'] = []
        mean_neut[f't_type_set_{tt}'] = 0
        mean_safe[f't_type_set_{tt}'] = 0
        mean_impr[f't_type_set_{tt}'] = 0
        mean_relw[f't_type_set_{tt}'] = 0
        mean_relm[f't_type_set_{tt}'] = 0
        mean_relr[f't_type_set_{tt}'] = 0
    for ut in u_type_set:
        res1[f'u_type_set_{ut}'] = []
        res2[f'u_type_set_{ut}'] = []
        res3[f'u_type_set_{ut}'] = []
        res4[f'u_type_set_{ut}'] = []
        res5[f'u_type_set_{ut}'] = []
        res6[f'u_type_set_{ut}'] = []
        mean_neut[f'u_type_set_{ut}'] = 0
        mean_safe[f'u_type_set_{ut}'] = 0
        mean_impr[f'u_type_set_{ut}'] = 0
        mean_relw[f'u_type_set_{ut}'] = 0
        mean_relm[f'u_type_set_{ut}'] = 0
        mean_relr[f'u_type_set_{ut}'] = 0
    for uo in u_order_set:
        res1[f'u_order_set_{uo}'] = []
        res2[f'u_order_set_{uo}'] = []
        res3[f'u_order_set_{uo}'] = []
        res4[f'u_order_set_{uo}'] = []
        res5[f'u_order_set_{uo}'] = []
        res6[f'u_order_set_{uo}'] = []
        mean_neut[f'u_order_set_{uo}'] = 0
        mean_safe[f'u_order_set_{uo}'] = 0
        mean_impr[f'u_order_set_{uo}'] = 0
        mean_relw[f'u_order_set_{uo}'] = 0
        mean_relm[f'u_order_set_{uo}'] = 0
        mean_relr[f'u_order_set_{uo}'] = 0
    for fr in fraction_set:
        res1[f'fraction_set_{fr}'] = []
        res2[f'fraction_set_{fr}'] = []
        res3[f'fraction_set_{fr}'] = []
        res4[f'fraction_set_{fr}'] = []
        res5[f'fraction_set_{fr}'] = []
        res6[f'fraction_set_{fr}'] = []
        mean_neut[f'fraction_set_{fr}'] = 0
        mean_safe[f'fraction_set_{fr}'] = 0
        mean_impr[f'fraction_set_{fr}'] = 0
        mean_relw[f'fraction_set_{fr}'] = 0
        mean_relm[f'fraction_set_{fr}'] = 0
        mean_relr[f'fraction_set_{fr}'] = 0
    for th in threshold_set:
        res1[f'threshold_set_{th}'] = []
        res2[f'threshold_set_{th}'] = []
        res3[f'threshold_set_{th}'] = []
        res4[f'threshold_set_{th}'] = []
        res5[f'threshold_set_{th}'] = []
        res6[f'threshold_set_{th}'] = []
        mean_neut[f'threshold_set_{th}'] = 0
        mean_safe[f'threshold_set_{th}'] = 0
        mean_impr[f'threshold_set_{th}'] = 0
        mean_relw[f'threshold_set_{th}'] = 0
        mean_relm[f'threshold_set_{th}'] = 0
        mean_relr[f'threshold_set_{th}'] = 0

    count = 0
    total = len(n_steps_set) * len(armcoef_set) * len(n_states_set) * len(f_type_set) * len(t_type_set) * len(u_type_set) * len(u_order_set) * len(fraction_set) * len(threshold_set)
    for nt in n_steps_set:
        for ns in n_states_set:
            for nc in armcoef_set:
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
                            for uo in u_order_set:

                                # NeutW = SafeWhittleV3(ns, na, R.vals, M.transitions, nt, ut, uo, np.ones(na))
                                # NeutW.get_whittle_indices(computation_type=method, params=[0, max_wi], n_trials=nt*na*ns)
                                # nw_indices = NeutW.w_indices

                                for th in threshold_set:

                                    thresh = th * np.ones(na)
                                    SafeW = SafeWhittle(ns, na, R.vals, M.transitions, nt, ut, uo, thresh)
                                    SafeW.get_whittle_indices(computation_type=method, params=[0, max_wi], n_trials=nt * na * ns)
                                    sw_indices = SafeW.w_indices

                                    for fr in fraction_set:
                                        nch = np.maximum(1, int(np.around(fr * na)))
                                        initial_states = (ns - 1) * np.ones(na, dtype=np.int32)

                                        rew_r, obj_r, _ = Process_Random(n_episodes, nt, ns, na, nch, thresh, R.vals, M.transitions, initial_states, ut, uo)
                                        rew_m, obj_m, _ = Process_Greedy(n_episodes, nt, ns, na, nch, thresh, R.vals, M.transitions, initial_states, ut, uo)
                                        rew_w, obj_w, _ = Process_WhtlRB(WhtlW, n_episodes, nt, ns, na, nch, thresh, R.vals, M.transitions, ww_indices, initial_states, ut, uo)
                                        # rew_n, obj_n, _ = Process_NeutRB(NeutW, n_episodes, nt, ns, na, nch, thresh, R.vals, M.transitions, nw_indices, initial_states, ut, uo)
                                        rew_s, obj_s, _ = Process_SafeRB(SafeW, n_episodes, nt, ns, na, nch, thresh, R.vals, M.transitions, sw_indices, initial_states, ut, uo)

                                        key_value = f'nt{nt}_nc{nc}_ns{ns}_{ft_type}_tt{tt}_ut{ut}_uo{uo}_th{th}_fr{fr}'
                                        joblib.dump([rew_r, obj_r], PATH3 + key_value + "_Random.joblib")
                                        joblib.dump([rew_m, obj_m], PATH3 + key_value + "_Myopic.joblib")
                                        joblib.dump([rew_w, obj_w], PATH3 + key_value + "_Whittl.joblib")
                                        joblib.dump([rew_s, obj_s], PATH3 + key_value + "_Safaty.joblib")

                                        ravg_vec = obj_r.mean(axis=0)
                                        mavg_vec = obj_m.mean(axis=0)
                                        wavg_vec = obj_w.mean(axis=0)
                                        savg_vec = obj_s.mean(axis=0)
                                        ravg = np.round(np.mean(obj_r), 3)
                                        mavg = np.round(np.mean(obj_m), 3)
                                        wavg = np.round(np.mean(obj_w), 3)
                                        savg = np.round(np.mean(obj_s), 3)
                                        impr_val = np.round(savg - wavg, 2)
                                        impr_prcw = np.round(100 * (savg - wavg) / wavg, 2)
                                        impr_prcr = np.round(100 * (savg - ravg) / ravg, 2)
                                        impr_prcm = np.round(100 * (savg - mavg) / mavg, 2)
                                        count += 1

                                        print(f"{count} / {total}: {key_value} ---> MEAN-Rel-W: {impr_prcw}, MEAN-Rel-M: {impr_prcm}, MEAN-Rel-R: {impr_prcr}")
                                        results1[key_value] = wavg
                                        results2[key_value] = savg
                                        results3[key_value] = impr_val
                                        results4[key_value] = impr_prcw
                                        results5[key_value] = impr_prcm
                                        results6[key_value] = impr_prcr
                                        df1_1 = pd.DataFrame(list(results1.items()), columns=['Key', 'MEAN-Neut'])
                                        df1_2 = pd.DataFrame(results2.values(), columns=['MEAN-Safe'])
                                        df1_3 = pd.DataFrame(results3.values(), columns=['MEAN-Impr'])
                                        df1_4 = pd.DataFrame(results4.values(), columns=['MEAN-RelW'])
                                        df1_5 = pd.DataFrame(results5.values(), columns=['MEAN-RelM'])
                                        df1_6 = pd.DataFrame(results6.values(), columns=['MEAN-RelR'])
                                        df1 = pd.concat([df1_1, df1_2, df1_3, df1_4, df1_5, df1_6], axis=1)
                                        df1.to_excel(PATH1, index=False)

                                        res1[f'n_steps_set_{nt}'].append(results1[key_value])
                                        res1[f'armcoef_set_{nc}'].append(results1[key_value])
                                        res1[f'n_states_set_{ns}'].append(results1[key_value])
                                        res1[f'f_type_set_{ft_type}'].append(results1[key_value])
                                        res1[f't_type_set_{tt}'].append(results1[key_value])
                                        res1[f'u_type_set_{ut}'].append(results1[key_value])
                                        res1[f'u_order_set_{uo}'].append(results6[key_value])
                                        res1[f'fraction_set_{fr}'].append(results1[key_value])
                                        res1[f'threshold_set_{th}'].append(results1[key_value])
                                        res2[f'n_steps_set_{nt}'].append(results2[key_value])
                                        res2[f'armcoef_set_{nc}'].append(results2[key_value])
                                        res2[f'n_states_set_{ns}'].append(results2[key_value])
                                        res2[f'f_type_set_{ft_type}'].append(results2[key_value])
                                        res2[f't_type_set_{tt}'].append(results2[key_value])
                                        res2[f'u_type_set_{ut}'].append(results2[key_value])
                                        res2[f'u_order_set_{uo}'].append(results6[key_value])
                                        res2[f'fraction_set_{fr}'].append(results2[key_value])
                                        res2[f'threshold_set_{th}'].append(results2[key_value])
                                        res3[f'n_steps_set_{nt}'].append(results3[key_value])
                                        res3[f'armcoef_set_{nc}'].append(results3[key_value])
                                        res3[f'n_states_set_{ns}'].append(results3[key_value])
                                        res3[f'f_type_set_{ft_type}'].append(results3[key_value])
                                        res3[f't_type_set_{tt}'].append(results3[key_value])
                                        res3[f'u_type_set_{ut}'].append(results3[key_value])
                                        res3[f'u_order_set_{uo}'].append(results6[key_value])
                                        res3[f'fraction_set_{fr}'].append(results3[key_value])
                                        res3[f'threshold_set_{th}'].append(results3[key_value])
                                        res4[f'n_steps_set_{nt}'].append(results4[key_value])
                                        res4[f'armcoef_set_{nc}'].append(results4[key_value])
                                        res4[f'n_states_set_{ns}'].append(results4[key_value])
                                        res4[f'f_type_set_{ft_type}'].append(results4[key_value])
                                        res4[f't_type_set_{tt}'].append(results4[key_value])
                                        res4[f'u_type_set_{ut}'].append(results4[key_value])
                                        res4[f'u_order_set_{uo}'].append(results6[key_value])
                                        res4[f'fraction_set_{fr}'].append(results4[key_value])
                                        res4[f'threshold_set_{th}'].append(results4[key_value])
                                        res5[f'n_steps_set_{nt}'].append(results5[key_value])
                                        res5[f'armcoef_set_{nc}'].append(results5[key_value])
                                        res5[f'n_states_set_{ns}'].append(results5[key_value])
                                        res5[f'f_type_set_{ft_type}'].append(results5[key_value])
                                        res5[f't_type_set_{tt}'].append(results5[key_value])
                                        res5[f'u_type_set_{ut}'].append(results5[key_value])
                                        res5[f'u_order_set_{uo}'].append(results6[key_value])
                                        res5[f'fraction_set_{fr}'].append(results5[key_value])
                                        res5[f'threshold_set_{th}'].append(results5[key_value])
                                        res6[f'n_steps_set_{nt}'].append(results6[key_value])
                                        res6[f'armcoef_set_{nc}'].append(results6[key_value])
                                        res6[f'n_states_set_{ns}'].append(results6[key_value])
                                        res6[f'f_type_set_{ft_type}'].append(results6[key_value])
                                        res6[f't_type_set_{tt}'].append(results6[key_value])
                                        res6[f'u_type_set_{ut}'].append(results6[key_value])
                                        res6[f'u_order_set_{uo}'].append(results6[key_value])
                                        res6[f'fraction_set_{fr}'].append(results6[key_value])
                                        res6[f'threshold_set_{th}'].append(results6[key_value])

                                        for key in list(res1.keys()):
                                            if len(res1[key]) != 0:
                                                mean_neut[key] = sum(res1[key]) / len(res1[key])
                                            if len(res2[key]) != 0:
                                                mean_safe[key] = sum(res2[key]) / len(res2[key])
                                            if len(res3[key]) != 0:
                                                mean_impr[key] = sum(res3[key]) / len(res3[key])
                                            if len(res4[key]) != 0:
                                                mean_relw[key] = sum(res4[key]) / len(res4[key])
                                            if len(res5[key]) != 0:
                                                mean_relm[key] = sum(res5[key]) / len(res5[key])
                                            if len(res6[key]) != 0:
                                                mean_relr[key] = sum(res6[key]) / len(res6[key])

                                        df2_1 = pd.DataFrame(list(mean_neut.items()), columns=['Key', 'MEAN-Neut'])
                                        df2_2 = pd.DataFrame(list(mean_safe.values()), columns=['MEAN-Safe'])
                                        df2_3 = pd.DataFrame(list(mean_impr.values()), columns=['MEAN-Impr'])
                                        df2_4 = pd.DataFrame(list(mean_relw.values()), columns=['MEAN-RelW'])
                                        df2_5 = pd.DataFrame(list(mean_relm.values()), columns=['MEAN-RelM'])
                                        df2_6 = pd.DataFrame(list(mean_relr.values()), columns=['MEAN-RelR'])
                                        df2 = pd.concat([df2_1, df2_2, df2_3, df2_4, df2_5, df2_6], axis=1)
                                        df2.to_excel(PATH2, index=False)
