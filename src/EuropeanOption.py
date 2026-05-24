import numpy as np

class EuropeanOption:

    def __init__(self, S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0, option_type: str = "call"):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.q = q
        self.option_type = option_type.lower()

        if self.option_type not in ['call', 'put']:
            raise ValueError("option_type must be either 'call' or 'put'")

    def intrinsic_value(self):
        if self.option_type == "call":
            return max(self.S - self.K, 0.0)
        return max(self.K - self.S, 0.0)
    

    def upper_bound(self):
        if self.option_type == "call":
            return self.S * np.exp(-self.q * self.T)
        return self.K * np.exp(-self.r * self.T)


    def lower_bound(self):
        discounted_K = self.K * np.exp(-self.r * self.T)
        discounted_S = self.S * np.exp(-self.q * self.T)

        if self.option_type == "call":
            return max(discounted_S - discounted_K, 0.0)
        return max(discounted_K - discounted_S, 0.0)


    # Returns the call or put price implied by put-call parity given the price.
    # E.g. parity_counterpart_price(20) on a call option returns the put price if the call trades at 20.
    def parity_counterpart_price(self, current_option_price: float):
        discounted_K = self.K * np.exp(-self.r * self.T)
        discounted_S = self.S * np.exp(-self.q * self.T)

        if self.option_type == "call":
            # P = C + K*exp(-rT) - S*exp(-qT)
            return current_option_price + discounted_K - discounted_S
        # C = P + S*exp(-qT) - K*exp(-rT)
        return current_option_price + discounted_S - discounted_K