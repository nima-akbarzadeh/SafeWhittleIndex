import os
import numpy
from learning import *
from Markov import *
import warnings
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
import joblib

warnings.filterwarnings("ignore")


def run_combination(params):
    nt, ns, np, nc, ft, tt, ut, uo, th, fr, df, method, l_episodes, n_episodes, n_iterations, PATH3 = params
    na = nc * ns
    ftype = numpy.ones(na, dtype=numpy.int32) if ft == 'hom' else 1 + numpy.arange(na)

    if tt == 0:
        prob_remain_1d = numpy.round(numpy.linspace(0.1, 0.9, na), 2)
    elif tt == 1 or tt == 2:
        prob_remain_1d = numpy.round(numpy.linspace(0.05, 0.45, na), 2)
    elif tt == 3:
        prob_remain_1d = numpy.round(numpy.linspace(0.1 / ns, 0.1 / ns, na), 2)
    elif tt == 4 or tt == 5:
        prob_remain_1d = numpy.round(numpy.linspace(0.1 / ns, 1 / ns, na), 2)
    elif tt == 6:
        prob_remain_1d = numpy.round(numpy.linspace(0.2, 0.8, na), 2)
    else:
        prob_remain_1d = numpy.round(numpy.linspace(0.1, 0.9, na), 2)
    numpy.random.shuffle(prob_remain_1d)
    prob_remain = numpy.tile(prob_remain_1d, (nt, 1)).T

    R = ValuesNS(df, nt, na, ns, ftype, True)
    M = MarkovDynamicsNS(nt, na, ns, prob_remain, tt, True)
    thresh = th * numpy.ones(na)

    nch = max(1, int(round(fr * na)))
    initial_states = (ns - 1) * numpy.ones(na, dtype=numpy.int32)

    probs_l, _, _, obj_l, _, _, obj_s = ProcessNS_LearnSafeTSRB(
        n_iterations, l_episodes, n_episodes, nt, ns, na, nch,
        thresh, tt, True, method, R.vals, M.transitions,
        initial_states, ut, uo, False, nt
    )

    key_value = f'nt{nt}_np{np}_ns{ns}_nc{nc}_ft{ft}_tt{tt}_ut{ut}_uo{uo}_th{th}_fr{fr}_df{df}'

    wip_obj = numpy.mean(numpy.sum(obj_s, axis=2), axis=0)
    joblib.dump(wip_obj, f'{PATH3}wip_obj_{key_value}.joblib')
    lrp_obj = numpy.mean(numpy.sum(obj_l, axis=2), axis=0)
    joblib.dump(lrp_obj, f'{PATH3}lrp_obj_{key_value}.joblib')
    lrp_reg = [sum(numpy.abs(wip_obj[:t] - lrp_obj[:t])) / t for t in range(1, len(lrp_obj))]
    joblib.dump(lrp_reg, f'{PATH3}lrp_reg_{key_value}.joblib')

    plt.figure(figsize=(8, 6))
    plt.plot(lrp_reg, label='Learning Policy Regret', color='blue')
    plt.xlabel('Learning Episodes')
    plt.ylabel('Average Performance')
    plt.title(key_value)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{PATH3}{key_value}_performance.png')
    plt.close()

    prb_err = numpy.abs(
        numpy.transpose(numpy.array([prob_remain[a] - numpy.mean(probs_l[:, :, a], axis=0) for a in range(na)])))
    plt.figure(figsize=(8, 6))
    plt.plot(prb_err)
    plt.xlabel('Learning Episodes')
    plt.ylabel('Parameter Error')
    plt.title(key_value)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{PATH3}{key_value}_proberrors.png')
    plt.close()

    return wip_obj, lrp_obj, lrp_reg


def main():
    param_sets = {
        'n_steps_set': [10],
        'n_partitions_set': [50],
        'n_states_set': [2, 3],
        'armcoef_set': [1, 2],
        'f_type_set': ['hom'],
        't_type_set': [3],
        'u_type_set': [1],
        'u_order_set': [4, 16],
        'threshold_set': [0.5],
        'fraction_set': [0.3],
        'nsrew_discount_set': [0.95],
    }

    PATH3 = './output-learn-finite-ns/'
    if not os.path.exists(PATH3):
            os.makedirs(PATH3)

    method = 3
    l_episodes = 250
    n_episodes = 10
    n_iterations = 20

    averages = {key: {} for key in ['wip', 'lrp', 'err', 'arm1', 'arm2']}
    for avg_key in averages:
        for param, values in param_sets.items():
            for value in values:
                averages[avg_key][f'{param}_{value}'] = []

    param_list = [
        (nt, ns, np, nc, ft_type, tt, ut, uo, th, fr, df, method, l_episodes, n_episodes, n_iterations, PATH3)
        for nt in param_sets['n_steps_set']
        for ns in param_sets['n_states_set']
        for np in param_sets['n_partitions_set']
        for nc in param_sets['armcoef_set']
        for ft_type in param_sets['f_type_set']
        for tt in param_sets['t_type_set']
        for ut in param_sets['u_type_set']
        for uo in param_sets['u_order_set']
        for th in param_sets['threshold_set']
        for fr in param_sets['fraction_set']
        for df in param_sets['nsrew_discount_set']
    ]

    total = len(param_list)

    num_cpus = cpu_count() - 1
    print(f"Using {num_cpus} CPUs")

    with Pool(num_cpus) as pool:
        for count, result in enumerate(pool.imap_unordered(run_combination, param_list), 1):
            wip_obj, lrp_obj, lrp_reg = result
            print(f"{count} / {total}")


if __name__ == '__main__':
    main()
