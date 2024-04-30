"""
Name: tutorial.py
Author: Xuewen Zhang
Date:at 19/03/2024
Description: A tutorial example to illustrate how to use deepctools
"""

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import casadi as cs

import deepctools as dpctools

class Plant():
    def __init__(self):
        """
            A discrete-time nonlinear model of polynomial single-input-single-output system:
                    y(t) = 4 * y(t-1) * u(t-1) - 0.5 * y(t-1) + 2 * u(t-1) * u(t) + u(t)
            The description of this system can be found in paper: https://ieeexplore.ieee.org/abstract/document/10319277
            # Note this plant is a nonlinear model which do not satisfy the assumption of Fundamental Lemma,
            # the control performance may not be good.
        """
        self.y0 = 0
        self.u0 = 0
        self.u = None
        self.lbu = -0.08   # bound of u, defined by myself
        self.ubu = 0.08

        self._formulate_plant()

    def _formulate_plant(self):
        """
            Here formulate the plant as a "casadi Function" since it is a simple system
            Or can formulate the plant as a model in "gym"
            To predict next y:
                y_cur = self.step(u_cur, u_pre, y_pre)
        """
        u_cur = cs.SX.sym('ut', 1)
        u_pre = cs.SX.sym('ut-1', 1)
        y_pre = cs.SX.sym('yt-1', 1)

        y_cur = 4 * cs.mtimes(y_pre, u_pre) - 0.5 * y_pre + 2 * cs.mtimes(u_pre, u_cur) + u_cur

        self.step = cs.Function('plant_step', [u_cur, u_pre, y_pre], [y_cur])

    def get_action(self):
        """
            Generate random control input of white noise (mean:0, variance:0.01)
            return: u
        """
        mean = 0
        std_var = 0.1
        u = np.random.normal(0, 0.1)
        u = np.clip(u, self.lbu, self.ubu)
        return u


    def generate_data(self, T, us=None):
        """
            Generate T step steps data
            if us is not None: generate data with constant u
            else: generate data with random u
            return: U, Y  |  [list]
        """
        U, Y = [], []

        U.append(self.u0)
        Y.append(self.y0)

        if us is None:
            get_action = self.get_action
        else:
            us = np.clip(us, self.lbu, self.ubu)
            get_action = lambda: us

        for i in tqdm(range(T)):
            u_cur = get_action()
            y_cur = self.step(u_cur, U[i], Y[i])
            U.append(u_cur)
            Y.append(y_cur.full()[0, 0])
        print('>> Data generation complete!')
        return U, Y





def main1():
    """
        A tutorial to illustrate how to use deepctools packages
        # Here no apply scale to the collected data
        # *The set-point will not change during control
        # Note this plant is a nonlinear model which do not satisfy the assumptions of Fundamental Lemma,
        # the control performance may not be good.
    """
    # ---------------------------setting---------------------------
    plant = Plant()
    # system parameters
    u_dim = 1
    y_dim = 1

    # DeePC config
    # feasible config:
    # good: {RDeePC:False, Tini:1, Np:5, T:5, uloss:uus}, T merely influence the performance as long as T>=5
    # good: {RDeePC:True, Tini:1, Np:1, T:600, uloss:du}, T will influence the steady-state loss
    # good: {RDeePC:True, Tini:1, Np:1, T:600, uloss:uus}, T will influence the steady-state loss
    # good: {RDeePC:True, Tini:1, Np:1, T:600, uloss:u}, T will influence the steady-state loss
    RDeePC = False         # if true, then Robust DeePC, if false, then DeePC
    uloss = "uus"          # loss of u in objective function, can be 'u', 'uus', 'du'
    Tini = 1
    Np = 1
    T = 5
    g_dim = T - Tini - Np + 1

    # Online parameters
    N = 150                               # entire control steps
    Nhold = 1                             # the steps of control action holds, action update interval
    Q = np.diag(np.tile([100], Np))        # weighting matrix of y
    R = np.diag(np.tile(0.5, Np))         # weighting matrix of u
    lambda_y = np.diag(np.tile(2, Tini))  # weighting matrix of noise of y
    lambda_g = 10 * np.eye(g_dim)         # weighting of the regulation of g

    # Offline data
    ud, yd = plant.generate_data(T-1)
    ud, yd = np.array(ud).reshape(-1, 1), np.array(yd).reshape(-1, 1)

    # online init data
    uini_0, yini_0 = plant.generate_data(Tini-1, us=0.02)
    uini = np.array(uini_0).reshape(-1, 1)
    yini = np.array(yini_0).reshape(-1, 1)
    us = 0.045
    ys = 0.037159090909090906

    # init deepc tools
    dpc_args = [u_dim, y_dim, T, Tini, Np, np.array([ys]), ud, yd, Q, R]
    dpc_kwargs = dict(us=np.array([us]),
                      lambda_g=lambda_g,
                      lambda_y=lambda_y,
                      ineqconidx={'u': [0]},
                      ineqconbd={'lbu': [plant.lbu], 'ubu': [plant.ubu]}
                      )
    dpc = dpctools.deepctools(*dpc_args, **dpc_kwargs)

    # init and formulate deepc solver
    dpc_opts = {
        'ipopt.max_iter': 100,  # 50
        'ipopt.tol': 1e-5,
        'ipopt.print_level': 1,
        'print_time': 0,
        'ipopt.acceptable_tol': 1e-8,
        'ipopt.acceptable_obj_change_tol': 1e-6,
    }
    if RDeePC:  # if true: Robust DeePC
        dpc.init_RDeePCsolver(uloss=uloss, opts=dpc_opts)
    else:
        dpc.init_DeePCsolver(uloss=uloss, opts=dpc_opts)

    # ---------------------------online DeePC control loop---------------------------
    Uf, Yf = [plant.u0], [plant.y0]  # u and y trajectory from Tini to Tini+N
    cost_y = [np.linalg.norm(plant.y0 - ys)]
    t_solve = []

    print('----------------DeePC loop----------------')
    with tqdm(total=int(N / Nhold), desc=f'DeePC: {N}', unit='opt') as t:
        for i in range(0, N, Nhold):
            # Solve the optimization  obtain the optimized operator g
            u_opt, g_opt, t_s = dpc.solver_step(uini, yini)

            t_solve.append(t_s)

            print('\n> %d | %s | iter: %d | Solved time: %f' % (
            i, dpc.solver.stats()['return_status'], dpc.solver.stats()['iter_count'], t_solve[-1]))

            # Obtain the optimized control inputs                                 # (u_dim*Np, 1)
            u_cur = u_opt[0]

            # apply to plant
            for j in range(Nhold):
                # plant simulation
                y_cur = plant.step(u_cur, Uf[-1], Yf[-1])

                y_cur = y_cur.full()[0, 0]
                Uf.append(u_cur)
                Yf.append(y_cur)

                uini = np.concatenate((uini[1:, :], np.array([u_cur]).reshape(-1, 1)), axis=0)
                yini = np.concatenate((yini[1:, :], np.array([y_cur]).reshape(-1, 1)), axis=0)

                cost = np.linalg.norm(y_cur - ys)
                cost_y.append(cost)

            # update tqdm progress bar
            t.set_postfix(loss={"y cost": cost})
            t.update(1)

    t_solve_mean = np.array([np.mean(t_solve)])
    print(f'>> Loop finished, mean solve time: {t_solve_mean}.')

    # plot the figure
    upath = np.concatenate((np.array(uini_0), Uf), axis=0)
    ypath = np.concatenate((np.array(yini_0), Yf), axis=0)
    costpath = []
    for i in range(Tini):
        costpath.append(np.linalg.norm(yini_0[i] - ys))
    costpath.extend(cost_y)

    fig1, ax = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(8, 6))
    t = np.arange(N + Tini + 1)
    # plot y
    ax[0].plot(t, ypath, color='red', label='DeePC')
    ax[0].axhline(y=ys, color='blue', linestyle='--', label='Reference')
    ax[0].axvline(x=Tini, color='g', linestyle=':', label='Tini')
    ax[0].set_xlabel('Steps')
    ax[0].set_ylabel('y')
    ax[0].legend()
    # plot u
    ax[1].plot(t, upath, color='red', label='DeePC')
    ax[1].axhline(y=us, color='blue', linestyle='--', label='Reference')
    ax[1].axvline(x=Tini, color='g', linestyle=':', label='Tini')
    ax[1].set_xlabel('Steps')
    ax[1].set_ylabel('u')
    # plot loss
    ax[2].plot(t, costpath, color='red', label='DeePC')
    ax[2].axvline(x=Tini, color='g', linestyle=':', label='Tini')
    ax[2].set_xlabel('Steps')
    ax[2].set_ylabel('y loss')
    plt.show()



def main2():
    """
        A tutorial to illustrate how to use deepctools packages
        # Here no apply scale to the collected data
        # *The set-point will change during control
        # Note this plant is a nonlinear model which do not satisfy the assumptions of Fundamental Lemma,
        # the control performance may not be good.
    """
    # ---------------------------setting---------------------------
    plant = Plant()
    # system parameters
    u_dim = 1
    y_dim = 1

    # DeePC config
    # feasible config:
    # good: {RDeePC:False, Tini:1, Np:5, T:5, uloss:uus}, T merely influence the performance as long as T>=5
    # good: {RDeePC:True, Tini:1, Np:1, T:600, uloss:du}, T will influence the steady-state loss
    # good: {RDeePC:True, Tini:1, Np:1, T:600, uloss:uus}, T will influence the steady-state loss
    # good: {RDeePC:True, Tini:1, Np:1, T:600, uloss:u}, T will influence the steady-state loss
    RDeePC = False         # if true, then Robust DeePC, if false, then DeePC
    uloss = "uus"          # loss of u in objective function, can be 'u', 'uus', 'du'
    Tini = 1
    Np = 1
    T = 5
    g_dim = T - Tini - Np + 1

    # Online parameters
    N = 150                               # entire control steps
    Nhold = 1                             # the steps of control action holds, action update interval
    Q = np.diag(np.tile([100], Np))        # weighting matrix of y
    R = np.diag(np.tile(0.5, Np))         # weighting matrix of u
    lambda_y = np.diag(np.tile(2, Tini))  # weighting matrix of noise of y
    lambda_g = 10 * np.eye(g_dim)         # weighting of the regulation of g

    # Offline data
    ud, yd = plant.generate_data(T-1)
    ud, yd = np.array(ud).reshape(-1, 1), np.array(yd).reshape(-1, 1)

    # online init data
    uini_0, yini_0 = plant.generate_data(Tini-1, us=0.02)
    uini = np.array(uini_0).copy().reshape(-1, 1)
    yini = np.array(yini_0).copy().reshape(-1, 1)
    
    # set-point change
    sp_change_t = 50
    us1 = 0.045
    ys1 = 0.037159090909090906
    
    us2 = 0.07
    ys2 = 0.06540983606557378
    
    uref_all = np.concatenate((np.tile(us1, (sp_change_t + Tini, 1)).reshape(-1, 1), np.tile(us2, (N - sp_change_t + 1, 1)).reshape(-1, 1)))
    yref_all = np.concatenate((np.tile(ys1, (sp_change_t + Tini, 1)).reshape(-1, 1), np.tile(ys2, (N - sp_change_t + 1, 1)).reshape(-1, 1)))

    # init deepc tools
    dpc_args = [u_dim, y_dim, T, Tini, Np, ud, yd, Q, R]
    dpc_kwargs = dict(
                      lambda_g=lambda_g,
                      lambda_y=lambda_y,
                      sp_change=True,
                      ineqconidx={'u': [0]},
                      ineqconbd={'lbu': [plant.lbu], 'ubu': [plant.ubu]}
                      )
    dpc = dpctools.deepctools(*dpc_args, **dpc_kwargs)

    # init and formulate deepc solver
    dpc_opts = {
        'ipopt.max_iter': 100,  # 50
        'ipopt.tol': 1e-5,
        'ipopt.print_level': 1,
        'print_time': 0,
        'ipopt.acceptable_tol': 1e-8,
        'ipopt.acceptable_obj_change_tol': 1e-6,
    }
    if RDeePC:  # if true: Robust DeePC
        dpc.init_RDeePCsolver(uloss=uloss, opts=dpc_opts)
    else:
        dpc.init_DeePCsolver(uloss=uloss, opts=dpc_opts)

    # ---------------------------online DeePC control loop---------------------------
    Uf, Yf = [plant.u0], [plant.y0]  # u and y trajectory from Tini to Tini+N
    cost_y = [np.linalg.norm(plant.y0 - yref_all[Tini])]
    t_solve = []

    print('----------------DeePC loop----------------')
    with tqdm(total=int(N / Nhold), desc=f'DeePC: {N}', unit='opt') as t:
        for i in range(0, N, Nhold):
            # Solve the optimization  obtain the optimized operator g
            uref = uref_all[i + Tini:i + Tini + Np, :]
            yref = yref_all[i + Tini:i + Tini + Np, :]
            u_opt, g_opt, t_s = dpc.solver_step(uini, yini, uref, yref)

            t_solve.append(t_s)

            print('\n> %d | %s | iter: %d | Solved time: %f' % (
            i, dpc.solver.stats()['return_status'], dpc.solver.stats()['iter_count'], t_solve[-1]))

            # Obtain the optimized control inputs                                 # (u_dim*Np, 1)
            u_cur = u_opt[0]

            # apply to plant
            for j in range(Nhold):
                # plant simulation
                y_cur = plant.step(u_cur, Uf[-1], Yf[-1])

                y_cur = y_cur.full()[0, 0]
                Uf.append(u_cur)
                Yf.append(y_cur)

                uini = np.concatenate((uini[1:, :], np.array([u_cur]).reshape(-1, 1)), axis=0)
                yini = np.concatenate((yini[1:, :], np.array([y_cur]).reshape(-1, 1)), axis=0)

                cost = np.linalg.norm(y_cur - yref_all[i * Nhold + j])
                cost_y.append(cost)

            # update tqdm progress bar
            t.set_postfix(loss={"y cost": cost})
            t.update(1)

    t_solve_mean = np.array([np.mean(t_solve)])
    print(f'>> Loop finished, mean solve time: {t_solve_mean}.')

    # plot the figure
    upath = np.concatenate((np.array(uini_0), Uf), axis=0)
    ypath = np.concatenate((np.array(yini_0), Yf), axis=0)
    costpath = []
    for i in range(Tini):
        costpath.append(np.linalg.norm(yini_0[i] - yref[i]))
    costpath.extend(cost_y)

    fig1, ax = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(8, 6))
    t = np.arange(N + Tini + 1)
    
    yref_all_step, t_step = dpctools.data_to_step(yref_all.reshape(-1, 1), t=t) 
    upath_step = dpctools.data_to_step(upath.reshape(-1, 1))
    uref_all_step = dpctools.data_to_step(uref_all.reshape(-1, 1))
    
    # plot y
    ax[0].plot(t, ypath, color='red', label='DeePC')
    # ax[0].axhline(y=ys, color='blue', linestyle='--', label='Reference')
    ax[0].plot(t_step, yref_all_step, color='blue', linestyle='--', label='Reference')
    ax[0].axvline(x=Tini, color='g', linestyle=':', label='Tini')
    ax[0].set_xlabel('Steps')
    ax[0].set_ylabel('y')
    ax[0].legend()
    # plot u
    ax[1].plot(t_step, upath_step, color='red', label='DeePC')
    # ax[1].axhline(y=us, color='blue', linestyle='--', label='Reference')
    ax[1].plot(t_step, uref_all_step, color='blue', linestyle='--', label='Reference')
    ax[1].axvline(x=Tini, color='g', linestyle=':', label='Tini')
    ax[1].set_xlabel('Steps')
    ax[1].set_ylabel('u')
    # plot loss
    ax[2].plot(t, costpath, color='red', label='DeePC')
    ax[2].axvline(x=Tini, color='g', linestyle=':', label='Tini')
    ax[2].set_xlabel('Steps')
    ax[2].set_ylabel('y loss')
    plt.show()



if __name__=='__main__':
    main()





