import numpy as np
import math

class EuropeanOption:
    
    def __init__(self, S: float, K: float, T: float, r: float, sigma: float, option_type: str):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        
        self.option_type = option_type.lower()
        
        if self.option_type not in ['call', 'put']:
            raise ValueError("option_type must be either 'call' or 'put'")
        

    def intrinsic_value(self): 
        if self.option_type == "call": 
            return max(self.S - self.K, 0.0)
        return max(self.K - self.S, 0.0)
    
        
    def upper_bound(self): # no upper bound as guaranteed to be positive
        if self.option_type == "call":
            return self.S
        return self.K * math.exp(-self.r * self.T)
    

    def lower_bound(self):
        DISCOUNTED_K = self.K * math.exp(-self.r * self.T)

        if self.option_type == "call":
            return max(self.S - DISCOUNTED_K, 0.0)
        return max(DISCOUNTED_K - self.S, 0.0)
    
    
    # Returns the Call or Put price depending on the option, 
    # e.g. put_call_parity(20) on a call option is saying that a put is priced at 20
    def put_call_parity(self, opposite_price: float):
        DISCOUNTED_K = self.K * math.exp(-self.r * self.T)

        if self.option_type == "call":
            return opposite_price + self.S - DISCOUNTED_K
        return opposite_price + DISCOUNTED_K - self.S
    
        
        



    


        