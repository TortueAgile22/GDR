import numpy as np

class GeometricBrownianMotion:
    def __init__(self, s0, mu, sigma):
        self.s0 = s0
        self.mu = mu
        self.sigma = sigma

    def generate_paths(self, dt, n_steps, n_sims):
        """
        Génère des trajectoires selon le modèle de diffusion ci-dessous :
        S_t = S_{t-1} * exp((mu - sigma^2/2) * dt + sigma * sqrt(dt) * W)
        """
        # W suit une loi normale centrée réduite
        w = np.random.standard_normal((n_steps, n_sims))
        
        # Calcul de l'exponentielle
        drift = (self.mu - 0.5 * self.sigma**2) * dt
        diffusion = self.sigma * np.sqrt(dt) * w
        
        # Simulation pas à pas
        increments = np.exp(drift + diffusion)
        paths = np.zeros((n_steps + 1, n_sims))
        paths[0] = self.s0
        
        for t in range(1, n_steps + 1):
            paths[t] = paths[t-1] * increments[t-1]
            
        return paths
    
class PDEModel:
    """Classe de base pour les paramètres de l'EDP : 
    0.5*sigma^2*V'' + (mu-lambda*sigma)*V' - r*V + d = 0
    """
    def __init__(self, params):
        self.params = params

class CIRModel(PDEModel):
    def get_coeffs(self, x):
        # Paramètres selon l'Annexe Page 5
        kappa = self.params['kappa']
        theta = self.params['theta']
        sigma = self.params['sigma']
        lmbda = self.params['lambda']
        
        # mu(t,x) = kappa * (theta - x)
        # sigma(t,x) = sigma * sqrt(x)
        # lambda(t,x) = lambda * sqrt(x) / sigma
        
        # aProc(t,x) = 0.5 * sigma(t,x)^2
        a = 0.5 * (sigma**2) * x
        # bProc(t,x) = mu(t,x) - lambda(t,x)*sigma(t,x)
        # b = kappa*(theta - x) - (lambda*sqrt(x)/sigma)*(sigma*sqrt(x)) = kappa*(theta-x) - lambda*x
        b = kappa * (theta - x) - (lmbda * x)
        # cProc(t,x) = r(t,x) = x
        c = x
        return a, b, c, 0

class VasicekModel(PDEModel):
    def get_coeffs(self, x):
        # Paramètres selon l'Annexe Page 5-6
        a_speed = self.params['a']
        b_mean = self.params['b']
        sigma = self.params['sigma']
        lmbda = self.params['lambda']
        
        # bprime = b - lambda*sigma/a
        b_prime = b_mean - (lmbda * sigma / a_speed)
        
        # aProc(t,x) = 0.5 * sigma^2
        a = 0.5 * (sigma**2) * np.ones_like(x)
        # bProc(t,x) = a * (bprime - x)
        b = a_speed * (b_prime - x)
        # cProc(t,x) = r(t,x) = x
        c = x
        return a, b, c, 0

class BlackScholesPDEModel(PDEModel):
    def get_coeffs(self, x):
        r = self.params['r']
        sigma = self.params['sigma']
        
        # aProc(t,x) = 0.5 * sigma^2 * x^2
        a = 0.5 * (sigma**2) * (x**2)
        # bProc(t,x) = r * x
        b = r * x
        # cProc(t,x) = r
        c = r * np.ones_like(x)
        return a, b, c, 0


class MertonModel(PDEModel):
    def get_coeffs(self, x):
        # Les coefficients sont identiques à Black-Scholes pour la valeur de la firme
        r = self.params['r']
        sigma = self.params['sigma']
        
        # aProc(t,x) = 0.5 * sigma^2 * x^2
        a = 0.5 * (sigma**2) * (x**2)
        # bProc(t,x) = r * x
        b = r * x
        # cProc(t,x) = r
        c = r * np.ones_like(x)
        return a, b, c, 0