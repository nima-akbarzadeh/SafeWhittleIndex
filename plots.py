import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd

if __name__ == '__main__':

    # transerror_l, wierrors_l, rew_l, obj_l, rew_ss, obj_ss = joblib.load('./20June24_Results/learnsafetsrb_543310.5.joblib')
    # # transerror_l, wierrors_l, rew_l, obj_l, rew_ss, obj_ss = joblib.load('./20June24_Results/learnsafetsrb_553310.5.joblib')
    # # transerror_l, wierrors_l, rew_l, obj_l, rew_ss, obj_ss = joblib.load('./20June24_Results/learnsafetsrb_5411310.5.joblib')
    #
    # # print(obj_ss)
    # # rew_ss = np.round(rew_ss, 2)
    # # obj_ss = np.round(obj_ss, 2)
    # # obj_l = np.round(obj_l, 2)
    # # rew_l = np.round(rew_l, 2)
    # # print(obj_ss)
    # # print(obj_ss - obj_l)
    #
    # trn_err = np.mean(np.max(transerror_l, axis=2), axis=0)
    # wis_err = np.mean(np.max(wierrors_l, axis=2), axis=0)
    # reg_obj = np.mean(obj_ss - obj_l, axis=(0, 2))
    # # print(reg_obj[-100:])
    # reg = np.cumsum(reg_obj)
    # regT = [reg[t] / (t + 1) for t in range(len(reg))]
    #
    # plt.figure(figsize=(8, 6))
    # plt.plot(trn_err, linewidth=4)
    # plt.xlabel('Episodes', fontsize=14, fontweight='bold')
    # plt.ylabel('Max Transition Error', fontsize=14, fontweight='bold')
    # plt.xticks(fontsize=12, fontweight='bold')
    # plt.yticks(fontsize=12, fontweight='bold')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    #
    # plt.figure(figsize=(8, 6))
    # plt.plot(wis_err, linewidth=4)
    # plt.xlabel('Episodes', fontsize=14, fontweight='bold')
    # plt.ylabel('Max WI Error', fontsize=14, fontweight='bold')
    # plt.xticks(fontsize=12, fontweight='bold')
    # plt.yticks(fontsize=12, fontweight='bold')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    #
    # plt.figure(figsize=(8, 6))
    # plt.plot(reg, linewidth=8)
    # plt.xlabel('Episodes', fontsize=14, fontweight='bold')
    # plt.ylabel('Regret', fontsize=14, fontweight='bold')
    # plt.xticks(fontsize=12, fontweight='bold')
    # plt.yticks(fontsize=12, fontweight='bold')
    # plt.ylim([0, 1])
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    #
    # plt.figure(figsize=(8, 6))
    # plt.plot(regT, linewidth=8)
    # plt.xlabel('Episodes', fontsize=14, fontweight='bold')
    # plt.ylabel('Regret/K', fontsize=14, fontweight='bold')
    # plt.xticks(fontsize=12, fontweight='bold')
    # plt.yticks(fontsize=12, fontweight='bold')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    #
    # # wip_obj = np.mean(np.sum(obj_ss, axis=2), axis=0)
    # # lrp_obj = np.mean(np.sum(obj_l, axis=2), axis=0)
    # # wip_out = [sum(wip_obj[:t]) / t for t in range(1, 1 + len(wip_obj))]
    # # lrp_out = [sum(lrp_obj[:t]) / t for t in range(1, 1 + len(lrp_obj))]
    # #
    # # plt.figure(figsize=(8, 6))
    # # plt.plot(lrp_out, label='Learner', color='blue', linewidth=4)
    # # plt.plot(np.mean(wip_obj) * np.ones(len(lrp_out)), label='Oracle', color='black', linestyle='--', linewidth=4)
    # # plt.xlabel('Episodes', fontsize=14, fontweight='bold')
    # # plt.ylabel('Objective', fontsize=14, fontweight='bold')
    # # plt.xticks(fontsize=12, fontweight='bold')
    # # plt.yticks(fontsize=12, fontweight='bold')
    # # plt.legend(prop={'weight': 'bold', 'size': 12})
    # # plt.grid(True)
    # # plt.show()

    ###################################################################################################################
    ###################################################################################################################
    ###################################################################################################################

    df = pd.read_excel('./output/Res_[3][2, 3, 4, 5][3, 4, 5].xlsx')
    y = df['MEAN-RelW']

    print(y.mean())

    # Define the boundaries for the histogram
    min_val = df['MEAN-RelW'].min()
    print(min_val)

    max_val = df['MEAN-RelW'].max()
    print(max_val)

    print(f"portion below zero: {sum(y.values < 0)/len(y)}")

    # Ensure 0 is included in the bins
    bins = list(np.linspace(min_val, max_val, num=15))
    if 0 not in bins:
        closest_to_zero = np.min(np.abs(bins))
        bins.remove(closest_to_zero)
        bins = np.sort(np.append(bins, 0))

    # Plot the histogram
    plt.hist(df['MEAN-RelW'], bins=bins, edgecolor='black')

    # Format the x-axis to have one decimal place
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
    plt.xticks(bins)

    plt.xticks(fontsize=14, fontweight='bold', rotation=90)
    plt.yticks(fontsize=14, fontweight='bold')

    # Reduce the whitespace between bins
    plt.hist(df['MEAN-RelW'], bins=bins, edgecolor='black', linewidth=0.5, color='blue')

    plt.grid(axis='y')
    plt.xlabel('Relative Improvement', fontsize=14, fontweight='bold')
    plt.ylabel('Frequency', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.show()

    ###################################################################################################################
    ###################################################################################################################
    ###################################################################################################################



