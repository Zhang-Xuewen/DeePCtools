# DeePCtools
A wrapped package for Data-enabled predictive control (DeePC) implementation. Including **DeePC** and **Robust DeePC** design with multiple objective functions.

If you have questions, remarks, technical issues etc. feel free to use the issues page of this repository. I am looking forward to your feedback and the discussion.

> Introduction: [link](https://qiyuan-zhang.github.io/my-toolbox/2024/04/15/Developed-deepctools.html)
> 
> PyPI site: [link](https://pypi.org/project/deepctools/)

## I. How to use

This package operates within the Python framework.

### 1. Required packages

- Numpy
- Matplotlib
- CasADi &emsp; &emsp;     <-- 3 <= __version__ <= 4

### 2. Usage

- Download the [*deepctools*](https://github.com/QiYuan-Zhang/DeePCtools/tree/8dbc2458966214bf9885f4d622e20c3b840641e2/deepctools) file and save it to your project directory.

- Or install using pip

```
    pip install deepctools
```
Then you can use the deepctools in your python project.

## II. deepctools toolbox organization
```
. 
└── deeptools 
    ├── hankel 
    ├── deepctools 
    │   ├── initialize_DeePCsolver
    │   ├── initialize_RDeePCsolver
    │   └── solver_step
    ├── getCasadiFunc 
    └── DiscreteSimulator
```

Under this file and function, there is detailed explanation of the usage, inputs, outputs of the functions. 

### 1. hankel(x, L)

Construct Hankel matrix of order L based on data x 

- The data x: $x \in \mathbb{R}^{T, n_x}$
- The Hankel matrix: $H_L(x) \in \mathbb{R}^{n_x  L \times T - L + 1}$

### 2. deepctools(*args, **kwargs)

Formulate and solve the DeePC problem, including **DeePC** and **Robust DeePC** design.

Construct the nlp solver for DeePC using CasADi IPOPT sovler, only formulate the solver once at the first beginning. 

In the online loop, no need to reformulate the NLP problem which saves lots of computational time.

Each iteration, only need provide updated parameters: $u_{ini}$, $y_{ini}$ (, $u_{ref}$, $y_{ref}$ if set-point changes during control).

> Objective function: $J = \Vert y - y^r \Vert_Q^2 + \Vert u_{loss} \Vert_R^2 + \mathcal{o}(\sigma_y, g)$

> $u_{loss}$ can be:

```       
        'u': u

        'uus': u - u_ref

        'du': delta u
``` 

There is a tutorial file in [`tutorial.py`](./tutorial.py).

#### a. initialize_DeePCsolver(uloss, opts)  

Formulate the DeePC design with different loss on control inputs.

The optmization problem can be formulated as:

```
        Standard DeePC design:                        |            Equivalent expression
     min J  =  || y - yref ||_Q^2 + || uloss ||_R^2   |   min J  =  || Uf*g - yref ||_Q^2 + || uloss ||_R^2
         s.t.   [Up]       [uini]                     |      s.t.   Up * g = uini
                [Yp] * g = [yini]                     |             Yp * g = yini
                [Uf]       [ u  ]                     |             ulb <= u <= uub
                [Yf]       [ y  ]                     |             ylb <= y <= yub
                ulb <= u <= uub                       |
                ylb <= y <= yub                       |  uloss = (u)   or    (u - uref)    or    (du)
```


#### b. initialize_RDeePCsolver(uloss, opts)  

Formulate the Robust DeePC design with slack variables and different loss on control inputs.

The optmization problem can be formulated as:

```
              Robust DeePC design:                    |            Equivalent expression
     min J  =  || y - yref ||_Q^2 + || uloss ||_R^2   |   min J  =  || Uf*g - ys ||_Q^2 + || uloss ||_R^2
                 + lambda_y||sigma_y||_2^2            |             + lambda_y||Yp*g-yini||_2^2
                 + lambda_g||g||_2^2                  |             + lambda_g||g||_2^2
         s.t.   [Up]       [uini]     [   0   ]       |      s.t.   Up * g = uini
                [Yp] * g = [yini]  +  [sigma_y]       |             ulb <= u <= uub
                [Uf]       [ u  ]     [   0   ]       |             ylb <= y <= yub
                [Yf]       [ y  ]     [   0   ]       |
                ulb <= u <= uub                       |
                ylb <= y <= yub                       |  uloss = (u)   or    (u - uref)    or    (du)
```

#### c. solver_step(uini, yini) 

Solve the optimization problem for one step, and output the optimized control inputs, operator g, and solving time.


### 3. getCasadiFunc(*args, **kwargs)

Construct the Function using CasADi

### 4. DiscreteSimulator(*args, **kwargs)

Construct the discrete system simulator for predicting next step

## III. Tutorial 

This is a tutorial example to illustrate how to use the *deepctools* to develop and implement DeePC design to different processes.

### 1. Plant

A simple discrete-time nonlinear model of polynomial single-input-single-output system is used: 

```
        y(t) = 4 * y(t-1) * u(t-1) - 0.5 * y(t-1) + 2 * u(t-1) * u(t) + u(t)
```

The model has been crafted as a `Plant` class to facilitate its utilization.

Notice:

- This system is adopted from the [paper](https://ieeexplore.ieee.org/abstract/document/10319277).
- Note this plant is a nonlinear model which do not satisfy the assumption of Fundamental Lemma, the control performance can be bad.
- For your own project, you can replace this plant to your own system.


### 2. DeePC designs

Within the sample code, you have the option to specify either DeePC or Robust DeePC design, along with various objective functions. This segment is implemented within the `main` function.

### 3. Tutorial results

Feasible DeePC config: 
```
     DeePC        |  {Tini:1, Np:5, T:5, uloss:uus}    | T merely influence the performance as long as T>=5 
     Robust DeePC |  {Tini:1, Np:1, T:600, uloss:du}   | T will influence the steady-state loss 
     Robust DeePC |  {Tini:1, Np:1, T:600, uloss:uus}  | T will influence the steady-state loss
     Robust DeePC |  {Tini:1, Np:1, T:600, uloss:u}    | T will influence the steady-state loss
```

Figure of control peformance under first config:

![peformance](https://github.com/QiYuan-Zhang/DeePCtools/assets/53491122/b662fe31-b2ee-43b2-9c38-98673b2ddfb1)


## Version update

#### **1.0.0**: initial commit

#### 1.0.1: debug

#### 1.0.2: debug

#### 1.0.3: debug

#### 1.0.4: add uref, yref which can tracking varying set-points

#### 1.0.5: debug

#### **1.1.0**: add SVD-based dimension reduction for DeePC

#### 1.1.1: debug SVD-based dimension reduction to ensure the rank properties

#### 1.1.2: debug SVD-based dimension reduction to ensure the rank properties



## License

The project is released under the APACHE license. See [LICENSE](LICENSE) for details.