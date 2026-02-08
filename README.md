# Project 3 : Monte Carlo Option Princing Terminal

## 1. Objective

The goal of this project is to implement a professional-grade simulation tool to price various financial derivatives using the Monte Carlo method. The tool handles both standard European options and complex path-dependent "exotic" options.

## 2. Mathematical Framework

The underlying asset price $S_t$ is assumed to follow a Geometric Brownian Motion (GBM), discretized as follows :

$$S_t = S_{t-1} exp^{(\mu -\frac{\sigma^2}{2}dt + \sigma \sqrt{dt} W)}$$

Where : 
- $\mu$ : Expected annual return (drift). 
- $\sigma$ : Annual volatility. 
- $W$ : Random variable following a standard normal distribution $N(0,1)$

## 3. Supported Instruments

- Vanilla Call : Standard right to buy the asset. 
- Tunnel : A strategy involving a cap and a floor to limit the gain/loss variance. 
- Himalaya : A path-dependant option based on the average performance of the asset over multiple dates. 
- Napoleon : An option paying a coupon adjusted by the worst periodic performance. 

## 4. Technical Implementation

- Language : Python 3.14.0. 
- Engine : Vectorized **NumPy** operations for high-speed simulations. 
- Interface : **Streamlit** web application for real-time adjustment. 
- Outputs : Estimated Option Price
    - Statistical Error (99% confidence level). 
    - Convergence Graph : Visualizing the stabilization of the price as the number of simulations increases. 


# Project 4 : PDE Numerical Solver (Finite Difference)

## 1. Objective

This project focuses on the numerical resolution of Partial Differential Equations (PDEs) applied to finance. By using finite difference methods, we compute the price of financial instruments on a grid without relying on random simulations. 

## 2. Numerical Scheme

We implement the **Crank-Nicholson** scheme ($\theta = 0.5$), which offers a gigh degree of stability and second-order temporal accuracy. The resulting tridiagonal system is solved efficiently at each time using the **Thomas Algorithm**. 

## 3. Implemented Models

The tool reproduces 4 classic financial models provided in the course annexes :
- Black & Scholes : For European equity options. 
- Cox, INersoll, Ross (CIR) : For mean-reverting interest rates with non-negativity constraints. 
- Vasicek : For mean-reverting interest rates. 
- Merton : For firm value and credit risk modeling. 

## 4. Key Parameters (Annex Compliance)

The solver is calibrated using the mandatory parameters from the project specifications:
- CIR : $\kappa = 0.8, \theta = 0.10, \sigma = 0.5, \lambda = 0.05$. 
- Vasicek : $a = 0.95, b = 0.10, \sigma = 0.2, \lambda = 0.05$. 
- Black-Scholes : $K = 100, \sigma = 0.20, r = 0.08$. 

## 5. technical Stack

- Language : Python 3.14.0
- Architecture : Modular design with separate classes for models and solvers. 
- Visualization : Price curves at $t=0$ compared to terminal payoffs at $T$. 
