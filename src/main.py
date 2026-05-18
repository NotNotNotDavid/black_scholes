import math

from EuropeanOption import EuropeanOption
from Pricer import Pricer


option1 = EuropeanOption(20, 20, 1, 0.01, 0.01, "call")

print(Pricer.arbitrage_checker(191, 129, option1))

option2_no_arb = EuropeanOption(31, 30, 3/12, 0.1, 0.01, "call")

print(Pricer.arbitrage_checker(3, 1.26, option2_no_arb))