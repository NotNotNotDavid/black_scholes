import numpy as np
import math 
from scipy.stats import norm

from EuropeanOption import EuropeanOption

class Pricer:

    def arbitrage_checker(call_price, put_price, option : EuropeanOption):
        accuracy = 0.01
        DISCOUNTED_K = option.K * math.exp(-option.r * option.T)

        return abs(call_price + DISCOUNTED_K - (put_price + option.S)) > accuracy
    
    
    # Returns the ideal call or put prices for a perfectly hedged position to break even
    # The model assumes Volatility remains the same, which is unrealistic in the real world.
    def black_scholes_price(option : EuropeanOption):

        SD = option.sigma * np.sqrt(option.T)
        DISCOUNTED_K = option.K * math.exp(-option.r * option.T)

        # Note that d1 and d2 are both log-normal adjusted, more details in notes. 
        d1 = (np.log(option.S / option.K) + (option.r + (option.sigma ** 2) / 2) * option.T) / SD
        d2 = d1 - SD

        if option.option_type == "call":
            return option.S * norm.cdf(d1) - DISCOUNTED_K * norm.cdf(d2) # the call price
        return DISCOUNTED_K * norm.cdf(-d2) - option.S * norm.cdf(-d1) # the put price

