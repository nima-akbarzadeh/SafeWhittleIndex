import matplotlib.pyplot as plt
import joblib
import numpy as np


if __name__ == '__main__':

    transerror_l, wierrors_l, rew_l, obj_l, rew_ss, obj_ss = joblib.load('./output/learnsafetsrb_523310.5.joblib')

    trn_err = np.mean(transerror_l, axis=(0, 2))
    wis_err = np.mean(wierrors_l, axis=(0, 2))
    reg_obj = obj_ss - obj_l
    reg = np.cumsum(np.mean(reg_obj, axis=(0, 2)))
    cumreg = [reg[t] / (t + 1) for t in range(len(reg))]

    plt.figure(figsize=(8, 6))
    plt.plot(trn_err, linewidth=4)
    plt.xlabel('Episodes', fontsize=14, fontweight='bold')
    plt.ylabel('Max Transition Error', fontsize=14, fontweight='bold')
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(wis_err, linewidth=4)
    plt.xlabel('Episodes', fontsize=14, fontweight='bold')
    plt.ylabel('Max WI Error', fontsize=14, fontweight='bold')
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(reg, linewidth=8)
    plt.xlabel('Episodes', fontsize=14, fontweight='bold')
    plt.ylabel('Regret', fontsize=14, fontweight='bold')
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(cumreg, linewidth=8)
    plt.xlabel('Episodes', fontsize=14, fontweight='bold')
    plt.ylabel('Regret/T', fontsize=14, fontweight='bold')
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True)
    plt.show()

    wip_obj = np.mean(np.sum(obj_ss, axis=2), axis=0)
    lrp_obj = np.mean(np.sum(obj_l, axis=2), axis=0)
    wip_out = [sum(wip_obj[:t]) / t for t in range(1, 1 + len(wip_obj))]
    lrp_out = [sum(lrp_obj[:t]) / t for t in range(1, 1 + len(lrp_obj))]

    plt.figure(figsize=(8, 6))
    plt.plot(lrp_out, label='Learner', color='blue', linewidth=4)
    plt.plot(np.mean(wip_obj) * np.ones(len(lrp_out)), label='Oracle', color='black', linestyle='--', linewidth=4)
    plt.xlabel('Episodes', fontsize=14, fontweight='bold')
    plt.ylabel('Objective', fontsize=14, fontweight='bold')
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    plt.legend(prop={'weight': 'bold', 'size': 12})
    plt.grid(True)
    plt.show()
