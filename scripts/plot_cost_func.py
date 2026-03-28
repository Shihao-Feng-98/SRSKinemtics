import numpy as np
import matplotlib.pyplot as plt


def get_next_psi(psi_pre, psi_lb, psi_ub, K=0.5, alpha=15.0):
    """
    K步长增益，应该关于距离是越来越大的，且和alpha相关
    alpha指数衰减因子，delta越小变化越快
    """
    psi_range = psi_ub - psi_lb
    temp1 = (psi_pre - psi_lb) / psi_range
    temp2 = (psi_ub - psi_pre) / psi_range
    delta = K * 0.5 * psi_range * (np.exp(-alpha*temp1) - np.exp(-alpha*temp2))
    return psi_pre + delta


if __name__ == "__main__":
    n = 200
    psi_lb = 0.0
    psi_ub = 1.0
    psi_pre = 0.01

    psi_array_50 = [psi_pre]
    for i in range(n):
        psi_next = get_next_psi(psi_pre, psi_lb, psi_ub, K=0.1, alpha=50.0)
        psi_array_50.append(psi_next)
        psi_pre = psi_next

    psi_pre = 0.01
    psi_array_20 = [psi_pre]
    for i in range(n):
        psi_next = get_next_psi(psi_pre, psi_lb, psi_ub, K=0.1, alpha=20.0)
        psi_array_20.append(psi_next)
        psi_pre = psi_next

    psi_pre = 0.01
    psi_array_10 = [psi_pre]
    for i in range(n):
        psi_next = get_next_psi(psi_pre, psi_lb, psi_ub, K=0.1, alpha=10.0)
        psi_array_10.append(psi_next)
        psi_pre = psi_next

    psi_pre = 0.01
    psi_array_5 = [psi_pre]
    for i in range(n):
        psi_next = get_next_psi(psi_pre, psi_lb, psi_ub, K=0.1, alpha=5.0)
        psi_array_5.append(psi_next)
        psi_pre = psi_next

    plt.plot(psi_array_50, label="alpha=50")
    plt.plot(psi_array_20, label="alpha=20")
    plt.plot(psi_array_10, label="alpha=10")
    plt.plot(psi_array_5, label="alpha=5")
    plt.xlabel("Iteration")
    plt.ylabel("Psi Value")
    plt.title("Cost Function Convergence")
    plt.grid()
    plt.legend()
    plt.show()