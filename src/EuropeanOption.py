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
        discounted_K = self.K * math.exp(-self.r * self.T)

        if self.option_type == "call":
            return max(self.S - discounted_K, 0.0)
        return max(discounted_K - self.S, 0.0)


    


        