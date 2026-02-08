import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# Import des modules locaux
from models import GeometricBrownianMotion, CIRModel, VasicekModel, BlackScholesPDEModel, MertonModel
from instruments import VanillaCall, TunnelOption, HimalayaOption, NapoleonOption
from solvers import MonteCarloPricer, PDESolver

# Configuration de la page
st.set_page_config(page_title="Risk Management Terminal", layout="wide")
st.title("Professional Financial Engineering Tool")

# --- SIDEBAR : Paramètres Dynamiques ---
with st.sidebar:
    st.header("1. Global Settings")
    project = st.radio("Select Project Module:", ["Project 3: Monte Carlo", "Project 4: PDE (Thomas)"])
    
    st.divider()
    
    st.header("2. Market Environment")
    s0 = st.number_input("Initial Price / Spot Rate ($S_0$)", value=100.0)
    r_base = st.number_input("Base Risk-free Rate ($r$)", value=0.03, format="%.4f")
    
    # Paramètres spécifiques au Projet 3 (GBM)
    if project == "Project 3: Monte Carlo":
        st.subheader("GBM Parameters")
        mu_sidebar = st.number_input("Drift ($\mu$)", value=0.05)
        sigma_sidebar = st.number_input("Volatility ($\sigma$)", value=0.20)
        
    st.divider()
    st.header("3. Simulation Settings")

    if project == "Project 3: Monte Carlo":
        n_sims = st.slider("Number of Simulations", 1000, 100000, 20000)
    
    maturity = st.number_input("Maturity (Years)", value=1.0, step=0.1)

# --- MODULE PROJET 3 : MONTE CARLO ---
if project == "Project 3: Monte Carlo":
    st.header("Monte Carlo Pricing Engine")
    
    option_type = st.selectbox(
        "Choose an exotic or vanilla instrument", 
        ["Vanilla Call", "Tunnel", "Himalaya", "Napoleon"]
    )

    # Initialisation de l'instrument
    inst = None

    # Affichage séquentiel (Pleine largeur)
    if option_type == "Vanilla Call":
        st.info("**Vanilla Call:** Gives the holder the right to buy an asset at strike $K$. Payoff at $T$: $\max(S_T - K, 0)$.")
        k = st.number_input("Strike ($K$)", value=100.0)
        inst = VanillaCall(k, maturity)

    elif option_type == "Tunnel":
        st.info("**Tunnel:** A strategy limiting price exposure. Payoff is a Vanilla Call capped by a Maximum (Cap) and protected by a Minimum (Floor).")
        k = st.number_input("Strike ($K$)", value=100.0)
        cap = st.number_input("Cap (Max Gain)", value=20.0)
        floor = st.number_input("Floor (Min Gain)", value=0.0)
        inst = TunnelOption(k, maturity, cap, floor)

    elif option_type == "Himalaya":
        st.info("**Himalaya:** Path-dependent option. The payoff is the average of the best performances over $N$ observation dates.")
        k = st.number_input("Notional ($K$)", value=100.0)
        n_obs = st.number_input("Number of Observation Dates", value=12, step=1)
        inst = HimalayaOption(k, maturity, n_obs)

    elif option_type == "Napoleon":
        st.info("**Napoleon:** Pays a fixed coupon adjusted by the *worst* performance of the asset over $N$ sub-periods.")
        k = st.number_input("Notional ($K$)", value=100.0)
        coupon = st.number_input("Fixed Coupon Rate", value=0.08)
        n_obs = st.number_input("Number of Observation Dates", value=12, step=1)
        inst = NapoleonOption(k, maturity, coupon, n_obs)

    if st.button("Run Monte Carlo Simulation"):
        with st.spinner('Simulating paths...'):
            model = GeometricBrownianMotion(s0, mu_sidebar, sigma_sidebar)
            pricer = MonteCarloPricer(model, inst, r_base)
            price, error, convergence = pricer.price(n_sims)
            
            res_col1, res_col2 = st.columns(2)
            res_col1.metric("Estimated Price", f"{price:.4f} €")
            res_col2.metric("99% Confidence Error (+/-)", f"{error:.4f} €")
            
            st.subheader("Convergence Analysis")
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(convergence, color='#1f77b4', label='Simulated Price')
            ax.axhline(price, color='red', linestyle='--', label='Final Estimate')
            ax.set_xlabel("Number of Simulations")
            ax.legend()
            st.pyplot(fig)

# --- MODULE PROJET 4 : PDE SOLVER ---
else:
    st.header("PDE Finite Difference Solver (Crank-Nicholson)")
    
    model_name = st.selectbox("Select PDE Model", ["CIR", "Vasicek", "Black-Scholes", "Merton"])
    
    params = {}

    # Affichage séquentiel (Pleine largeur)
    if model_name == "CIR":
        st.info("**Cox-Ingersoll-Ross (CIR):** Mean-reverting model for interest rates ensuring non-negative rates. Formula: $dr_t = \kappa(\\theta - r_t)dt + \sigma\sqrt{r_t}dW_t$.")
        kappa = st.number_input("Kappa ($\kappa$ - Speed of reversion)", value=0.8)
        theta = st.number_input("Theta ($\\theta$ - Long term mean)", value=0.10)
        sigma_pde = st.number_input("Sigma ($\sigma$ - Volatility)", value=0.5)
        lmbda = st.number_input("Lambda ($\lambda$ - Risk premium)", value=0.05)
        params = {'kappa': kappa, 'theta': theta, 'sigma': sigma_pde, 'lambda': lmbda}
        x_range = (0.0, 1.0)

    elif model_name == "Vasicek":
        st.info("**Vasicek Model:** Mean-reverting model for rates where rates can theoretically become negative. Formula: $dr_t = a(b - r_t)dt + \sigma dW_t$.")
        a_param = st.number_input("a (Speed of reversion)", value=0.95)
        b_param = st.number_input("b (Long term mean)", value=0.10)
        sigma_pde = st.number_input("Sigma ($\sigma$ - Volatility)", value=0.2)
        lmbda = st.number_input("Lambda ($\lambda$ - Risk premium)", value=0.05)
        params = {'a': a_param, 'b': b_param, 'sigma': sigma_pde, 'lambda': lmbda}
        x_range = (0.0, 1.0)

    elif model_name == "Black-Scholes":
        st.info("**Black-Scholes PDE:** The fundamental equation for European options. Solved on a price grid.")
        r_pde = st.number_input("Risk-free Rate ($r$)", value=0.08)
        sigma_pde = st.number_input("Volatility ($\sigma$)", value=0.20)
        k_pde = st.number_input("Strike ($K$)", value=100.0)
        params = {'r': r_pde, 'sigma': sigma_pde, 'k': k_pde}
        x_range = (s0 * 0.5, s0 * 1.5)
    
    elif model_name == "Merton":
        st.info("**Merton Model:** Credit risk model in which equity is viewed as a call option on the firm's assets. Debt $D$ plays the role of the strike price.")
        r_merton = st.number_input("Risk-free Rate ($r$)", value=0.05)
        sigma_merton = st.number_input("Asset Volatility ($\sigma$)", value=0.30)
        debt = st.number_input("Debt Level ($D$)", value=80.0)
        params = {'r': r_merton, 'sigma': sigma_merton, 'debt': debt}
        # La grille x représente ici la valeur des actifs de la firme (V)
        x_range = (debt * 0.5, s0 * 1.5)

    if st.button("Solve PDE System"):
        with st.spinner('Solving tridiagonal system...'):
            nx, nt = 100, 200
            x_grid = np.linspace(x_range[0], x_range[1], nx)
            t_grid = np.linspace(0, maturity, nt)
            
            if model_name == "Black-Scholes":
                v_terminal = np.maximum(x_grid - params['k'], 0)
            else:
                v_terminal = np.ones_like(x_grid)
                
            if model_name == "CIR": model = CIRModel(params)
            elif model_name == "Vasicek": model = VasicekModel(params)
            elif model_name == "Black-Scholes": model = BlackScholesPDEModel(params)
            elif model_name == "Merton": model = MertonModel(params)
            
            solver = PDESolver(model, x_grid, t_grid)
            v_final = solver.solve(v_terminal)
            
            st.subheader("Price Curve Visualization")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(x_grid, v_terminal, '--', label='Payoff at Maturity (T)', alpha=0.5)
            ax.plot(x_grid, v_final, label='Calculated Price (t=0)', color='green', linewidth=2)
            ax.set_xlabel("Asset Price / Rate")
            ax.set_ylabel("Value")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            st.success("Numerical solution converged using Thomas algorithm (Crank-Nicholson).")