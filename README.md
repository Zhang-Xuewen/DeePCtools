# DeePCtools
Developed a wrapped package for DeePC implementation, such as mpctools.
Also contains some general functions for building casadi Functions and discrete models.

## I. Functions
deeptools \
    |___hankel \
    |___deepctools \
    |___getCasadiFunc \
    |___DiscreteSimulator

### 1. hankel(x, L)

> Construct hankel matrix of order L based on data x

### 2. deepctools(*args, **kwargs)

> Including DeePC and Robust DeePC design
> 
> Objective function: $J = \Vert y - y^r \Vert_Q^2 + \Vert u_{loss} \Vert_R^2 + \mathcal{o}(\sigma_y, g)$
>
> $u_{loss}$ can be:
> 
>        'u': $u_{loss}=u$
>
>        'uus': $u_{loss}=u-u^r$
>
>        'du': $u_{loss}=\delta u$
> 
> > Construct the nlp solver for DeePC using casadi ipopt sovler
> 
> > only formulate the solver once at the first begining. 
>
> In the online loop, no need to reformulate the NLP problem which saves lots of computational time.
> 
> > Each iteration, only need provide updated parameters: $u_{ini}$, $y_{ini}$.

Note: There is a tutorial file to implement and use deepctools in **tutorial.py**.

### 3. getCasadiFunc()
> Construct the Function using casadi

### 4. DiscreteSimulator()
> Construct the discrete system simulator for predicting next step

## II. Tutorial

> A discrete-time nonlinear model of polynomial single-input-single-output system: 

        `y(t) = 4 * y(t-1) * u(t-1) - 0.5 * y(t-1) + 2 * u(t-1) * u(t) + u(t)`

> The description of this system can be found in [paper](https://ieeexplore.ieee.org/abstract/document/10319277)

- Note this plant is a nonlinear model which do not satisfy the assumption of Fundamental Lemma, the control performance may not be good.

Tutorial results:
```
     Feasible DeePC config: 
     good: {RDeePC:False, Tini:1, Np:5, T:5, uloss:uus}, T merely influence the performance as long as T>=5 
     good: {RDeePC:True, Tini:1, Np:1, T:600, uloss:du}, T will influence the steady-state 
     good: {RDeePC:True, Tini:1, Np:1, T:600, uloss:uus}, T will influence the steady-state 
     good: {RDeePC:True, Tini:1, Np:1, T:600, uloss:u}, T will influence the steady-state 
```

![peformance](https://github.com/QiYuan-Zhang/DeePCtools/assets/53491122/b662fe31-b2ee-43b2-9c38-98673b2ddfb1)

