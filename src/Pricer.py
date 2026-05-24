import numpy as np
from scipy.stats import norm

from EuropeanOption import EuropeanOption


class Pricer:
    # Returns the ideal call or put prices for a perfectly hedged position to break even
    # The model assumes Volatility remains the same, which is unrealistic in the real world.
    # Returns True if put-call parity is violated beyond tolerance.
    # Parity: C + K*exp(-rT) = P + S*exp(-qT)
    @staticmethod
    def parity_violation(call_price, put_price, option: EuropeanOption, tolerance: float = 0.01):
        """Returns True if put-call parity is violated beyond tolerance.
        Parity: C + K*exp(-rT) = P + S*exp(-qT)
        """
        discounted_K = option.K * np.exp(-option.r * option.T)
        discounted_S = option.S * np.exp(-option.q * option.T)
        return abs(call_price + discounted_K - (put_price + discounted_S)) > tolerance
    

    # Black-Scholes price with continuous dividend yield q.
    # the Drift is (r - q + sigma^2/2); spot term gets an exp(-qT) factor.
    @staticmethod
    def black_scholes_price(option: EuropeanOption):

        SD = option.sigma * np.sqrt(option.T)
        discounted_K = option.K * np.exp(-option.r * option.T)
        discounted_S = option.S * np.exp(-option.q * option.T)

        # Note that d1 and d2 are both log-normal adjusted, more details in notes. 
        d1 = (np.log(option.S / option.K) + (option.r - option.q + (option.sigma ** 2) / 2) * option.T) / SD
        d2 = d1 - SD

        if option.option_type == "call":
            return discounted_S * norm.cdf(d1) - discounted_K * norm.cdf(d2)
        return discounted_K * norm.cdf(-d2) - discounted_S * norm.cdf(-d1)

