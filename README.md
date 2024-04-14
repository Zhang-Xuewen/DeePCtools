# DeePCtools
Developed a wrapped package for DeePC implementation, such as mpctools.
Also contains some general functions for building casadi Functions and discrete models.

## I. Functions
deeptools
    |___hankel
    |___deepctools
    |___getCasadiFunc
    |___DiscreteSimulator

### 1. hankel(x, L)

> Construct hankel matrix of order L based on data x

### 2. deepctools(*args, **kwargs)

> Contains DeePC and Robust DeePC design
> 
> loss on u can be 'u': ||u||_R^2 'uus': ||u-us||_R^2 'du': ||du||_R^2
> 
> > Construct the nlp solver for DeePC using casadi ipopt sovler
> 
> > only formulate the solver once
> 
> > when solve, provide, uini, yini parameters

Note: can refer to the tutorial.py

### 3. getCasadiFunc()
> Construct the Function using casadi

### 4. DiscreteSimulator()
> Construct the discrete system simulator for predicting next step

## II. Tutorial

> A discrete-time nonlinear model of polynomial single-input-single-output system:
        `y(t) = 4 * y(t-1) * u(t-1) - 0.5 * y(t-1) + 2 * u(t-1) * u(t) + u(t)`
> The description of this system can be found in [paper](https://ieeexplore.ieee.org/abstract/document/10319277)

- Note this plant is a nonlinear model which do not satisfy the assumption of Fundamental Lemma, the control performance may not be good.

    

