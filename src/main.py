import math

from EuropeanOption import EuropeanOption
from Pricer import Pricer


# option1 = EuropeanOption(20, 20, 1, 0.01, 0.01, "call")

# print(Pricer.arbitrage_checker(191, 129, option1))

# option2_no_arb = EuropeanOption(31, 30, 3/12, 0.1, 0.01, "call")

# print(Pricer.arbitrage_checker(3, 1.26, option2_no_arb))

# print(Pricer.black_scholes_price(option2_no_arb))

# Test for Black Scholes ----

call_contract = EuropeanOption(S=100, K=100, T=0.5, r=0.05, sigma=0.20, option_type="call")
put_contract  = EuropeanOption(S=100, K=100, T=0.5, r=0.05, sigma=0.20, option_type="put")

# Get Values
c_price = Pricer.black_scholes_price(call_contract)
# p_price = Pricer.black_scholes_price(put_contract)

p_price_double_check = call_contract.put_call_parity(c_price)

print(f"Theoretical Call Price: ${c_price:.2f}")
print(f"Theoretical Put Price:  ${p_price_double_check:.2f}")

# Check arbitrage
is_arbitrage = Pricer.arbitrage_checker(call_price=c_price, put_price=p_price_double_check, option=call_contract)
print(f"Is there an algebraic bug in my calculus? {is_arbitrage}")




