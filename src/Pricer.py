import numpy as np
from scipy.stats import norm

from EuropeanOption import EuropeanOption
from scipy.optimize import brentq

class IVSolverError(RuntimeError):
    # Raised when implied volatility cannot be recovered from the given price.
    pass


class Pricer:
    # Returns the ideal call or put prices for a perfectly hedged position to break even
    # The model assumes Volatility remains the same, which is unrealistic in the real world.
    # Returns True if put-call parity is violated beyond tolerance.
    # Parity: C + K*exp(-rT) = P + S*exp(-qT)
    @staticmethod
    def parity_violation(call_price, put_price, option: EuropeanOption, tolerance: float = 0.01):

        discounted_K = option.K * np.exp(-option.r * option.T)
        discounted_S = option.S * np.exp(-option.q * option.T)
        return abs(call_price + discounted_K - (put_price + discounted_S)) > tolerance
    

    # Black-Scholes price with continuous dividend yield q.
    # the Drift (d1) is (r - q + sigma^2/2); spot term gets an exp(-qT) factor.
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
    

    # Vega requires computing d1 based on the option's current sigma 
    # Returns dollar change per unit change in sigma (not per 1%)
    # ∂price/∂sigma = vega
    @staticmethod
    def vega(option: EuropeanOption):

        if option.T == 0 or option.sigma == 0:
            return 0.0  # By definition: no time or no vol = no vega 
        
        SD = option.sigma * np.sqrt(option.T)
        d1 = (np.log(option.S / option.K) + (option.r - option.q + (option.sigma ** 2) / 2) * option.T) / SD
        
        return option.S * np.exp(-option.q * option.T) * np.sqrt(option.T) * norm.pdf(d1)
    

    # Returns the implied volatility (sigma)
    @staticmethod
    def implied_volatility(price, option: EuropeanOption):

        # throws exception when given a price that is impossible to calculate
        if price > option.upper_bound() or price < option.lower_bound():
            raise IVSolverError("The price given is out of the range possible")
        
        # specs
        current_sigma = 0.2
        tolerance = 1e-8

        # Essentially the error, the objective for Brent's
        def objective(sigma):
            return Pricer._bs_price_at_sigma(sigma, option) - price

    
        # Newton's (numerical) method 
        for _ in range(100):
            error = objective(current_sigma)

            if abs(error) <= tolerance:
                return current_sigma
            
            v = Pricer._vega_at_sigma(current_sigma, option)
            if abs(v) < 1e-10:
                break  # vega vanished, try Brent
    
            current_sigma = current_sigma - error / v
    
            if current_sigma <= 0 or current_sigma > 5.0:
                break  # sigma left range, try Brent

            # Calculate vega for the numerical method
            current_sigma = current_sigma - error / Pricer._vega_at_sigma(current_sigma, option)

        try:
            return brentq(objective, 1e-6, 5.0) # arbitrary numbers, lower and upper bound will never happen
        
        except ValueError:
            raise IVSolverError("Both Newton and Brent failed")
        

    @staticmethod
    def _bs_price_at_sigma(sigma, option):
        SD = sigma * np.sqrt(option.T)
        discounted_K = option.K * np.exp(-option.r * option.T)
        discounted_S = option.S * np.exp(-option.q * option.T)

        # Note that d1 and d2 are both log-normal adjusted, more details in notes. 
        d1 = (np.log(option.S / option.K) + (option.r - option.q + (sigma ** 2) / 2) * option.T) / SD
        d2 = d1 - SD

        if option.option_type == "call":
            return discounted_S * norm.cdf(d1) - discounted_K * norm.cdf(d2)
        return discounted_K * norm.cdf(-d2) - discounted_S * norm.cdf(-d1)

    @staticmethod
    def _vega_at_sigma(sigma, option):

        if option.T == 0 or sigma == 0:
            return 0.0  # By definition: no time or no vol = no vega 
        
        SD = sigma * np.sqrt(option.T)
        d1 = (np.log(option.S / option.K) + (option.r - option.q + (sigma ** 2) / 2) * option.T) / SD
        
        return option.S * np.exp(-option.q * option.T) * np.sqrt(option.T) * norm.pdf(d1)




      