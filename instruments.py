import numpy as np

class Option:
    def __init__(self, strike, maturity):
        self.k = strike
        self.t = maturity

class VanillaCall(Option):
    def get_payoff(self, paths):
        # Payoff classique : Max(S_T - K, 0)
        return np.maximum(paths[-1] - self.k, 0)

class TunnelOption(Option):
    def __init__(self, strike, maturity, cap, floor):
        super().__init__(strike, maturity)
        self.cap = cap
        self.floor = floor

    def get_payoff(self, paths):
        # Payoff bridé entre une borne basse et une borne haute
        payoff_vanilla = np.maximum(paths[-1] - self.k, 0)
        return np.clip(payoff_vanilla, self.floor, self.cap)

class HimalayaOption(Option):
    def __init__(self, strike, maturity, n_observations):
        super().__init__(strike, maturity)
        self.n_obs = n_observations

    def get_payoff(self, paths):
        # On divise la trajectoire en n périodes
        # On prend la performance max de chaque période
        indices = np.linspace(0, len(paths)-1, self.n_obs + 1, dtype=int)
        performances = []
        for i in range(1, len(indices)):
            period_chunk = paths[indices[i-1]:indices[i]]
            # Performance max locale
            perf = np.max(period_chunk, axis=0) / paths[indices[i-1]] - 1
            performances.append(np.maximum(perf, 0))
        return self.k * np.mean(performances, axis=0)

class NapoleonOption(Option):
    def __init__(self, strike, maturity, coupon, n_observations):
        super().__init__(strike, maturity)
        self.coupon = coupon
        self.n_obs = n_observations

    def get_payoff(self, paths):
        # Coupon fixe + la pire performance des sous-périodes
        indices = np.linspace(0, len(paths)-1, self.n_obs + 1, dtype=int)
        periodic_returns = []
        for i in range(1, len(indices)):
            ret = (paths[indices[i]] / paths[indices[i-1]]) - 1
            periodic_returns.append(ret)
        
        worst_performance = np.min(periodic_returns, axis=0)
        return self.k * (self.coupon + worst_performance)