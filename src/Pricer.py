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
    

    # Vega requires computing d1 based on the option's current sigma 
    # Returns dollar change per unit change in sigma (not per 1%)
    @staticmethod
    def vega(option: EuropeanOption):

        if option.T == 0 or option.sigma == 0:
            return 0.0  # By definition: no time or no vol = no vega 
        
        SD = option.sigma * np.sqrt(option.T)
        d1 = (np.log(option.S / option.K) + (option.r - option.q + (option.sigma ** 2) / 2) * option.T) / SD
        
        return option.S * np.exp(-option.q * option.T) * np.sqrt(option.T) * norm.pdf(d1)
    

    # Recover the volatility implied by a market price.
    # Strategy: Newton-Raphson first using quadratic convergence when it works,
    # with Brent's method as a bracketed fallback when Newton fails.
    # Raises IVSolverError if the price violates no-arbitrage bounds
    # or if both solvers fail.
    @staticmethod
    def implied_volatility(
        market_price: float,
        option: EuropeanOption,
        tolerance: float = 1e-6,
        max_iterations: int = 100,
        initial_guess: float = 0.20, # given generally 0.1 ~ 0.5
    ):
        # Reject prices outside no arbitrage bounds immediately, no sigma can produce them
        lower = option.lower_bound()
        upper = option.upper_bound()
        if market_price < lower - 1e-10 or market_price > upper + 1e-10:
            raise IVSolverError(
                f"Market price {market_price:.6f} is outside no-arbitrage bounds "
                f"[{lower:.6f}, {upper:.6f}]"
            )
        
        # Newton's Methods
        sigma = initial_guess
        for _ in range(max_iterations):
            # Compute price at current sigma without mutating the option
            price = Pricer._bs_price_at_sigma(option, sigma)
            error = price - market_price
            
            if abs(error) < tolerance:
                return sigma  # Converged
            
            # Compute vega at current sigma
            v = Pricer._vega_at_sigma(option, sigma)
            if abs(v) < 1e-8:
                break  # Vega vanished; Newton can't move
            
            sigma = sigma - error / v
            if sigma <= 0 or sigma > 5.0:
                break  # Diverged out of plausible range
        
        # Brent's Method
        def objective(s):
            return Pricer._bs_price_at_sigma(option, s) - market_price
        
        lo, hi = 1e-6, 5.0
        try:
            f_lo = objective(lo)
            f_hi = objective(hi)
        except Exception as e:
            raise IVSolverError(f"Black-Scholes failed during bracketing: {e}")
        
        if f_lo * f_hi > 0:
            # No sign change in the bracket: no IV in [lo, hi]
            raise IVSolverError(
                f"No implied vol bracketed in [{lo}, {hi}]. "
                f"f(lo)={f_lo:.6f}, f(hi)={f_hi:.6f}. "
                "Price may be at a no-arbitrage boundary where vega is ~0."
            )
        
        try:
            return brentq(objective, lo, hi, xtol=tolerance, maxiter=max_iterations)
        except Exception as e:
            raise IVSolverError(f"Brent's method failed: {e}")

    # helper method: Black-Scholes price using a candidate sigma, without mutating the option.
    # Private helper for the IV solver — lets us try different sigmas
    # without touching option.sigma
    @staticmethod
    def _bs_price_at_sigma(option: EuropeanOption, sigma: float):
      
        SD = sigma * np.sqrt(option.T)
        discounted_K = option.K * np.exp(-option.r * option.T)
        discounted_S = option.S * np.exp(-option.q * option.T)
        d1 = (np.log(option.S / option.K) + (option.r - option.q + sigma**2 / 2) * option.T) / SD
        d2 = d1 - SD
        
        if option.option_type == "call":
            return discounted_S * norm.cdf(d1) - discounted_K * norm.cdf(d2)
        return discounted_K * norm.cdf(-d2) - discounted_S * norm.cdf(-d1)

    # helper method: Vega using a candidate sigma. Private helper for the IV solver
    @staticmethod
    def _vega_at_sigma(option: EuropeanOption, sigma: float):
        
        if option.T == 0 or sigma == 0:
            return 0.0
        SD = sigma * np.sqrt(option.T)
        d1 = (np.log(option.S / option.K) + (option.r - option.q + sigma**2 / 2) * option.T) / SD
        return option.S * np.exp(-option.q * option.T) * np.sqrt(option.T) * norm.pdf(d1)