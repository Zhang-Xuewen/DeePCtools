"""
Name: util.py
Author: Xuewen Zhang
Date:at 13/04/2024
version: 1.0.0
Description: utils of deepctools
"""
import casadi as cs
import numpy as np


def hankel(x, L):
    """
        ------Construct Hankel matrix------
        x: data sequence (data_size, x_dim)
        L: row dimension of the hankel matrix
        T: data samples of data x
        return: H(x): hankel matrix of x  H(x): (x_dim*L, T-L+1)
                H(x) = [x(0)   x(1) ... x(T-L)
                        x(1)   x(2) ... x(T-L+1)
                        .       .   .     .
                        .       .     .   .
                        .       .       . .
                        x(L-1) x(L) ... x(T-1)]
                Hankel matrix of order L has size:  (x_dim*L, T-L+1)
    """
    if not isinstance(x, np.ndarray):
        x = np.array(x)

    T, x_dim = x.shape

    Hx = np.zeros((L * x_dim, T - L + 1))
    for i in range(L):
        Hx[i * x_dim:(i + 1) * x_dim, :] = x[i:i + T - L + 1, :].T  # x need transpose to fit the hankel dimension
    return Hx

def safevertcat(x):
    """
    Safer wrapper for Casadi's vertcat.

    the input x is expected to be an iterable containing multiple things that
    should be concatenated together. This is in contrast to Casadi 3.0's new
    version of vertcat that accepts a variable number of arguments. We retain
    this (old, Casadi 2.4) behavior because it makes it easier to check types.

    If a single SX or MX object is passed, then this doesn't do anything.
    Otherwise, if all elements are numpy ndarrays, then numpy's concatenate
    is called. If anything isn't an array, then casadi.vertcat is called.
    """
    symtypes = set(["SX", "MX"])
    xtype = getattr(x, "type_name", lambda: None)()
    if xtype in symtypes:
        val = x
    elif (not isinstance(x, np.ndarray) and
          all(isinstance(a, np.ndarray) for a in x)):
        val = np.concatenate(x)
    else:
        val = cs.vertcat(*x)
    return val

# Now give the actual functions.
def rk4(f, x0, par, Delta=1, M=1):
    """
    Does M RK4 timesteps of function f with variables x0 and parameters par.

    The first argument of f must be var, followed by any number of parameters
    given in a list in order.

    Note that var and the output of f must add like numpy arrays.
    """
    h = Delta / M
    x = x0
    j = 0
    while j < M:  # For some reason, a for loop creates problems here.
        k1 = f(x, *par)
        k2 = f(x + k1 * h / 2, *par)
        k3 = f(x + k2 * h / 2, *par)
        k4 = f(x + k3 * h, *par)
        x = x + (k1 + 2 * k2 + 2 * k3 + k4) * h / 6
        j += 1
    return x