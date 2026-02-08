import numpy as np
from scipy.stats import norm

class MonteCarloPricer:
    def __init__(self, model, instrument, r):
        self.model = model
        self.instrument = instrument
        self.r = r

    def price(self, n_sims, n_steps=252):
        dt = self.instrument.t / n_steps
        paths = self.model.generate_paths(dt, n_steps, n_sims)
        
        payoffs = self.instrument.get_payoff(paths)
        
        # Actualisation (Discounting)
        discounted_payoffs = payoffs * np.exp(-self.r * self.instrument.t)
        
        price = np.mean(discounted_payoffs)
        std_dev = np.std(discounted_payoffs)
        
        # Erreur à 99% : quantile 2.575 * erreur type 
        error_99 = 2.575 * (std_dev / np.sqrt(n_sims))
        
        # Pour le graphique de convergence 
        cumulative_means = np.cumsum(discounted_payoffs) / np.arange(1, n_sims + 1)
        
        return price, error_99, cumulative_means
    
def thomas_algorithm(a, b, c, d):
    """
    Résout Ax = d où A est une matrice tridiagonale.
    a: diagonale inférieure, b: diagonale principale, c: diagonale supérieure.
    """
    nf = len(d)
    ac, bc, cc, dc = map(np.array, (a, b, c, d))
    for it in range(1, nf):
        mc = ac[it-1] / bc[it-1]
        bc[it] = bc[it] - mc * cc[it-1]
        dc[it] = dc[it] - mc * dc[it-1]
    
    xc = bc
    xc[-1] = dc[-1] / bc[-1]
    for il in range(nf-2, -1, -1):
        xc[il] = (dc[il] - cc[il] * xc[il+1]) / bc[il]
    return xc

class PDESolver:
    def __init__(self, model, x_grid, t_grid):
        self.model = model
        self.x = x_grid
        self.t = t_grid
        self.dx = x_grid[1] - x_grid[0]
        self.dt = t_grid[1] - t_grid[0]

    def solve(self, terminal_condition):
        Nx = len(self.x)
        Nt = len(self.t)
        v = terminal_condition.copy()
        
        # Paramètre theta pour Crank-Nicholson (0.5 = Schéma centré)
        theta_cn = 0.5 

        # Récupère les coefficients PDE du modèle : a(x), b(x), c(x), d(x)
        a, b, c, d = self.model.get_coeffs(self.x)
        
        # Précalcul les poids de discrétisation pour la matrice tridiagonale
        alpha = (a / self.dx**2) - (b / (2 * self.dx))
        beta = (-2 * a / self.dx**2) - c
        gamma = (a / self.dx**2) + (b / (2 * self.dx))
        
        # Time-stepping (à l'envers, de T vers 0 pour le pricing)
        for n in range(Nt - 1):
            # Matrices pour Crank-Nicholson: (I - dt*theta*M) V_new = (I + dt*(1-theta)*M) V_old
            # Left-hand side (Implicit)
            L_lower = -self.dt * theta_cn * alpha[1:]
            L_main = 1 - self.dt * theta_cn * beta
            L_upper = -self.dt * theta_cn * gamma[:-1]
            
            # Terme explicite (RHS)
            RHS = v + self.dt * (1 - theta_cn) * (alpha * np.roll(v, 1) + beta * v + gamma * np.roll(v, -1))
            
            # Conditions aux limites (Dirichlet simplifiées)
            RHS[0], RHS[-1] = v[0], v[-1]
            L_main[0], L_main[-1] = 1.0, 1.0
            L_lower[0], L_upper[-1] = 0.0, 0.0
            
            # Résolution avec Thomas
            v = thomas_algorithm(L_lower, L_main, L_upper, RHS)
            
        return v