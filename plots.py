import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':

    # Define the parameters
    gamma = 0.5
    o = 8

    # Define the function
    def f(J):
        return 1 - gamma ** (-1 / o) * np.maximum(0, gamma - J) ** (1 / o)

    def g(J):
        numerator = 1 + np.exp(-o * (1 - gamma))
        denominator = 1 + np.exp(-o * (J - gamma))
        return numerator / denominator

    def h(J):
        return [1 if j >= gamma else 0 for j in J]

    # Generate values for J
    J_values = np.linspace(0, 1, 400)
    f_values = f(J_values)
    g_values = g(J_values)
    h_values = h(J_values)

    # Plot the function
    plt.figure(figsize=(10, 6))
    plt.plot(J_values, f_values, linewidth=8)
    plt.xlabel('Total Reward', fontsize=20, fontweight='bold')
    plt.ylabel('Utility Value', fontsize=20, fontweight='bold')
    plt.xticks(fontsize=16, fontweight='bold')
    plt.yticks(fontsize=16, fontweight='bold')
    plt.grid(True)
    plt.show()

    # Plot the function
    plt.figure(figsize=(10, 6))
    plt.plot(J_values, g_values, linewidth=8)
    plt.xlabel('Total Reward', fontsize=20, fontweight='bold')
    plt.ylabel('Utility Value', fontsize=20, fontweight='bold')
    plt.xticks(fontsize=16, fontweight='bold')
    plt.yticks(fontsize=16, fontweight='bold')
    plt.grid(True)
    plt.show()


    # Plot the function
    plt.figure(figsize=(10, 6))
    plt.plot(J_values, h_values, linewidth=8)
    # plt.plot(0.5, 0, marker='o', markersize=10, fillstyle='none')
    # plt.plot(0.5, 1, marker='o', markersize=10)
    plt.xlabel('Total Reward', fontsize=20, fontweight='bold')
    plt.ylabel('Utility Value', fontsize=20, fontweight='bold')
    plt.xticks(fontsize=16, fontweight='bold')
    plt.yticks(fontsize=16, fontweight='bold')
    plt.grid(True)
    plt.show()
