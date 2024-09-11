import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def dxdts(n):
    if n == 0:
        return 't'
    else:
        str_dxdts = ''
    for i in range(n):
        dxdt = ''
        for j in range(i):
            dxdt += 'd/dt '
        str_dxdts += dxdt
    return str_dxdts + 'x' 

def rk(a, b, X0, m, f) -> pd.DataFrame:
    """
    Given an ODE problem:
    x(t) = g(t, x, x')
    x'(t) = h(t, x)

    Parameters
    ----------
    a: float
        The start of the invertal.
    b: float
        The end of the interval.
    x0: float
        Initial values of x(t) and x'(t) at t=a.
    m: int
        Number of interval partitions.
    """
    n = len(X0)
    h = (b - a)/m # step

    data = np.zeros((m+1, n+1))
    data[:, 0] = np.arange(a,b+h,h)
    data[0, 1:] = X0

    table = pd.DataFrame(data=data,
                        columns=[dxdts(j) for j in range(n+1)],
                        index=np.arange(m+1))

    for i in range(m):
        t = table.loc[i, 't']
        X = table.loc[i, 'x':]
        k1 = f(t, X)
        k2 = f(t + h/2, X + k1*h/2)
        k3 = f(t + h/2, X + k2*h/2)
        k4  = f(t + h, X + k3*h)
        k = (k1 + 2*(k2 + k3) + k4)/6
        X_next = X + h*k
        table.loc[i+1, 'x':] = X_next

    return table