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
    n_steps_set = [3, 4, 5]
    n_states_set = [2, 3, 4, 5]
    armcoef_set = [3, 4, 5]
    f_type_set = ['hom']
    t_type_set = [3]
    u_type_set = [1, 2]
    u_order_set = [4, 8, 16]
    threshold_set = [0.4, 0.5, 0.6]
    fraction_set = [0.3, 0.4, 0.5]

    PATH1 = f'./results/Res_{t_type_set}{n_states_set}{armcoef_set}.xlsx'
    PATH2 = f'./results/ResAvg_{t_type_set}{n_states_set}{armcoef_set}.xlsx'
    PATH3 = f'./results/'

    method = 3
    n_episodes = 100
    # np.random.seed(42)

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

    count = 0
    total = len(n_steps_set) * len(armcoef_set) * len(n_states_set) * len(f_type_set) * len(t_type_set) * len(u_type_set) * len(u_order_set) * len(fraction_set) * len(threshold_set)
    for nt in n_steps_set:
        for ns in n_states_set:
            for nc in armcoef_set:
                na = nc * ns
                for ft_type in f_type_set:

                    for tt in t_type_set:

                        for ut in u_type_set:
                            for uo in u_order_set:

                                for th in threshold_set:

                                    thresh = th * np.ones(na)

                                    for fr in fraction_set:
                                        nch = np.maximum(1, int(np.around(fr * na)))

                                        key_value = f'nt{nt}_nc{nc}_ns{ns}_{ft_type}_tt{tt}_ut{ut}_uo{uo}_th{th}_fr{fr}'

                                        r_load = joblib.load(PATH3 + key_value + "_Random.joblib")
                                        m_load = joblib.load(PATH3 + key_value + "_Myopic.joblib")
                                        w_load = joblib.load(PATH3 + key_value + "_Whittl.joblib")
                                        n_load = joblib.load(PATH3 + key_value + "_SoSafe.joblib")
                                        s_load = joblib.load(PATH3 + key_value + "_Safaty.joblib")
                                        rew_r, obj_r = r_load[0], r_load[1]
                                        rew_m, obj_m = m_load[0], m_load[1]
                                        rew_w, obj_w = w_load[0], w_load[1]
                                        rew_n, obj_n = n_load[0], n_load[1]
                                        rew_s, obj_s = s_load[0], s_load[1]

                                        ravg_vec = obj_r.mean(axis=0)
                                        mavg_vec = obj_m.mean(axis=0)
                                        wavg_vec = obj_w.mean(axis=0)
                                        navg_vec = obj_n.mean(axis=0)
                                        savg_vec = obj_s.mean(axis=0)
                                        ravg = np.round(np.mean(obj_r), 3)
                                        mavg = np.round(np.mean(obj_m), 3)
                                        wavg = np.round(np.mean(obj_w), 3)
                                        navg = np.round(np.mean(obj_n), 3)
                                        savg = np.round(np.mean(obj_s), 3)
                                        impr_val = np.round(savg - wavg, 2)
                                        impr_sw = np.round(100 * (savg - wavg) / wavg, 2)
                                        impr_sr = np.round(100 * (savg - ravg) / ravg, 2)
                                        impr_sm = np.round(100 * (savg - mavg) / mavg, 2)
                                        impr_nw = np.round(100 * (navg - wavg) / wavg, 2)
                                        impr_nr = np.round(100 * (navg - ravg) / ravg, 2)
                                        impr_nm = np.round(100 * (navg - mavg) / mavg, 2)
                                        count += 1

                                        rewwavg = np.round(np.mean(rew_w), 3)
                                        rewsavg = np.round(np.mean(rew_s), 3)
                                        loss_sw = np.round(100 * (rewsavg - rewwavg) / rewwavg, 2)

                                        print(f"{count} / {total}: {key_value} ---> MEAN-Rel-W: {impr_sw}, MEAN-Rel-M: {impr_sm}, MEAN-Rel-R: {impr_sr}, MEAN-Rel-R: {impr_nr}")
                                        results1[key_value] = wavg
                                        results2[key_value] = savg
                                        results3[key_value] = impr_val
                                        results4[key_value] = impr_sw
                                        results5[key_value] = impr_sm
                                        results6[key_value] = impr_sr
                                        results7[key_value] = impr_nw
                                        results8[key_value] = impr_nm
                                        results9[key_value] = impr_nr
                                        results0[key_value] = loss_sw

    df1_1 = pd.DataFrame(list(results1.items()), columns=['Key', 'MEAN-Neut'])
    df1_2 = pd.DataFrame(results2.values(), columns=['MEAN-Safe'])
    df1_3 = pd.DataFrame(results3.values(), columns=['MEAN-Impr'])
    df1_4 = pd.DataFrame(results4.values(), columns=['MEAN-RelW'])
    df1_5 = pd.DataFrame(results5.values(), columns=['MEAN-RelM'])
    df1_6 = pd.DataFrame(results6.values(), columns=['MEAN-RelR'])
    df1_7 = pd.DataFrame(results7.values(), columns=['MEAN-RelSoftW'])
    df1_8 = pd.DataFrame(results8.values(), columns=['MEAN-RelSoftM'])
    df1_9 = pd.DataFrame(results9.values(), columns=['MEAN-RelSoftR'])
    df1_0 = pd.DataFrame(results0.values(), columns=['MEAN-LOSS-SW'])
    df1 = pd.concat([df1_1, df1_2, df1_3, df1_4, df1_5, df1_6, df1_7, df1_8, df1_9, df1_0], axis=1)
    df1.to_excel(PATH1, index=False)
