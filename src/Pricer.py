import numpy as np
import math 

from EuropeanOption import EuropeanOption

class Pricer:

    def arbitrage_checker(call_price, put_price, option : EuropeanOption):
        accuracy = 0.01
        DISCOUNTED_K = option.K * math.exp(-option.r * option.T)

        return abs(call_price + DISCOUNTED_K - (put_price + option.S)) > accuracy

