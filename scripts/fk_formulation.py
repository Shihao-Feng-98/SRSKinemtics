import sympy as sp
import numpy as np
from IPython import embed

def mdh_transform_(a, alpha, d, theta):
    ct, st = sp.cos(theta), sp.sin(theta)
    ca, sa = sp.cos(alpha), sp.sin(alpha)
    return np.array([[ct, -st, 0., a],
                    [st*ca, ct*ca, -sa, -d*sa],
                    [st*sa, ct*sa, ca, d*ca],
                    [0., 0., 0., 1.]])

if __name__ == "__main__":
    d1 = sp.symbols("d1", real=True)
    d2 = sp.symbols("d2", real=True)
    d3 = sp.symbols("d3", real=True)
    d4 = sp.symbols("d4", real=True)
    q1 = sp.symbols("q1", real=True)
    q2 = sp.symbols("q2", real=True)
    q3 = sp.symbols("q3", real=True)
    q4 = sp.symbols("q4", real=True)
    q5 = sp.symbols("q5", real=True)
    q6 = sp.symbols("q6", real=True)
    q7 = sp.symbols("q7", real=True)
    mdh_params_ = np.array([[0.,       0., d1, q1],
                            [0., -np.pi/2, 0., q2],
                            [0.,  np.pi/2, d2, q3],
                            [0., -np.pi/2, 0., q4],
                            [0.,  np.pi/2, d3, q5],
                            [0., -np.pi/2, 0., q6],
                            [0.,  np.pi/2, d4, q7]])

    T03 = np.eye(4)
    for i in range(3):
        T03 = T03 @ mdh_transform_(mdh_params_[i][0],
                                mdh_params_[i][1],
                                mdh_params_[i][2],
                                mdh_params_[i][3])

    T47 = np.eye(4)
    for i in range(4,7):
        T47 = T47 @ mdh_transform_(mdh_params_[i][0],
                                mdh_params_[i][1],
                                mdh_params_[i][2],
                                mdh_params_[i][3])

    embed()