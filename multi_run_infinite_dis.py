import os
import numpy
import pandas as pd
import joblib
from processes import *
from whittle import *
from Markov import *
import warnings
from multiprocessing import Pool, cpu_count
warnings.filterwarnings("ignore")


def run_combination(params):
    nt, ns, np, nc, ft, tt, ut, uo, th, fr, df, method, n_episodes, PATH3 = params
    na = nc * ns
    ftype = numpy.ones(na, dtype=numpy.int32) if ft == 'hom' else 1 + numpy.arange(na)

    prob_remain = numpy.round(numpy.linspace(0.1 / ns, 1 / ns, na), 2)
    numpy.random.shuffle(prob_remain)

    r_vals = values(nt, na, ns, ftype, True)
    M = MarkovDynamics(na, ns, prob_remain, tt, True)

    WhtlW = WhittleDisInf(df, ns, na, r_vals, M.transitions)
    WhtlW.get_whittle_indices(computation_type=method, params=[0, int(nt/(1-df))], n_trials=10*int(nt/(1-df)))

    SafeW = SafeWhittleDisInf(df, [ns, np, np], na, r_vals, M.transitions, nt, ut, uo, th * numpy.ones(na))
    SafeW.get_whittle_indices(computation_type=method, params=[0, int(nt/(1-df))], n_trials=10*int(nt/(1-df)))

    nch = max(1, int(round(fr * na)))
    initial_states = (ns - 1) * numpy.ones(na, dtype=numpy.int32)

    processes = [
        ("Random", ProcessDis_Random),
        ("Greedy", ProcessDis_Greedy),
        ("Whittl", lambda *args: ProcessDisInf_WhtlRB(WhtlW, *args)),
        ("Safaty", lambda *args: ProcessDisInf_SafeRB(SafeW, *args))
    ]

    results = {}
    for name, process in processes:
        rew, obj, _ = process(df, n_episodes, nt, ns, na, nch, th * numpy.ones(na), r_vals, M.transitions,
                              initial_states, ut, uo)
        joblib.dump([rew, obj],
                    f"{PATH3}nt{nt}_np{np}_nc{nc}_ns{ns}_ft{ft}_tt{tt}_ut{ut}_uo{uo}_th{th}_fr{fr}_df{df}_{name}.joblib")
        results[name] = numpy.round(numpy.mean(obj), 4)

    key_value = f'nt{nt}_np{np}_nc{nc}_ns{ns}_ft{ft}_tt{tt}_ut{ut}_uo{uo}_th{th}_fr{fr}_df{df}'
    impr_vl = numpy.round(results['Safaty'] - results['Whittl'], 3)
    impr_sw = numpy.round(100 * (results['Safaty'] - results['Whittl']) / results['Whittl'], 3)
    impr_sr = numpy.round(100 * (results['Safaty'] - results['Random']) / results['Random'], 3)
    impr_sm = numpy.round(100 * (results['Safaty'] - results['Greedy']) / results['Greedy'], 3)

    return key_value, results['Whittl'], results['Safaty'], impr_vl, impr_sw, impr_sr, impr_sm


def main():

    param_sets = {
        'n_steps_set': [100],
        'n_partitions_s_set': [10, 20],
        'n_states_set': [2, 5],
        'armcoef_set': [3, 5],
        'f_type_set': ['hom'],
        't_type_set': [3],
        'u_type_set': [1, 2],
        'u_order_set': [4, 16],
        'threshold_set': [0.3, 0.5, 0.7],
        'fraction_set': [0.3, 0.5],
        'discount_set': [0.9, 0.95]
    }

    PATH1 = f'./output-infinite-dis/Res_{param_sets["t_type_set"]}{param_sets["n_states_set"]}{param_sets["armcoef_set"]}.xlsx'
    PATH2 = f'./output-infinite-dis/ResAvg_{param_sets["t_type_set"]}{param_sets["n_states_set"]}{param_sets["armcoef_set"]}.xlsx'
    PATH3 = f'./output-infinite-dis/'
    if not os.path.exists(PATH3):
        os.makedirs(PATH3)

    method = 3
    n_episodes = 100

    results = {key: {} for key in ['1', '2', '3', '4', '5', '6']}
    averages = {key: {} for key in ['neut', 'safe', 'impr', 'relw', 'relm', 'relr']}
    for avg_key in averages:
        for param, values in param_sets.items():
            for value in values:
                averages[avg_key][f'{param}_{value}'] = []

    param_list = [
        (nt, ns, np,  nc, ft_type, tt, ut, uo, th, fr, df, method, n_episodes, PATH3)
        for nt in param_sets['n_steps_set']
        for ns in param_sets['n_states_set']
        for np in param_sets['n_partitions_s_set']
        for nc in param_sets['armcoef_set']
        for ft_type in param_sets['f_type_set']
        for tt in param_sets['t_type_set']
        for ut in param_sets['u_type_set']
        for uo in param_sets['u_order_set']
        for th in param_sets['threshold_set']
        for fr in param_sets['fraction_set']
        for df in param_sets['discount_set']
    ]

    total = len(param_list)

    # Determine the number of CPUs to use
    num_cpus = cpu_count()-1
    print(f"Using {num_cpus} CPUs")

    # Create a Pool of workers
    with Pool(num_cpus) as pool:
        # Use imap to get results as they complete
        for count, result in enumerate(pool.imap_unordered(run_combination, param_list), 1):
            key_value, wavg, savg, impr_vl, impr_sw, impr_sm, impr_sr = result
            for i, value in enumerate([wavg, savg, impr_vl, impr_sw, impr_sm, impr_sr]):
                results[str(i + 1)][key_value] = value

            print(f"{count} / {total}: {key_value} ---> MEAN-Rel-W: {impr_sw}, MEAN-Rel-M: {impr_sm}, MEAN-Rel-R: {impr_sr}")

            for param, value in zip(['nt', 'np', 'nc', 'ns', 'ft', 'tt', 'ut', 'uo', 'th', 'fr', 'df'], result[0].split('_')):
                param_key = f'{param}_{value}'
                for i, avg_key in enumerate(['neut', 'safe', 'impr', 'relw', 'relm', 'relr']):
                    if param_key not in averages[avg_key]:
                        averages[avg_key][param_key] = []
                    averages[avg_key][param_key].append(results[str(i + 1)][key_value])

    # Save results to Excel
    df1 = pd.DataFrame({f'MEAN-{key.capitalize()}': value for key, value in results.items()})
    df1.index.name = 'Key'
    df1.to_excel(PATH1)

    df2 = pd.DataFrame({f'MEAN-{key.capitalize()}': {k: numpy.mean(v) if v else 0 for k, v in avg.items()}
                        for key, avg in averages.items()})
    df2.index.name = 'Key'
    df2.to_excel(PATH2)


if __name__ == '__main__':
    main()
