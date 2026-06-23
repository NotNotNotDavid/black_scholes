# tests/test_pricer.py
import math
import pytest
from EuropeanOption import EuropeanOption
from Pricer import Pricer, IVSolverError

# to run tests: pytest tests/test_pricer.py in terminal

# --- Reference values from Hull's textbook ---

def test_bs_call_textbook_value():
    """Hull Chapter 15 Example 15.6: S=42, K=40, T=0.5, r=0.10, sigma=0.20.
    Hull gives c ≈ 4.76 for the call."""
    option = EuropeanOption(S=42, K=40, T=0.5, r=0.10, sigma=0.20, option_type="call")
    price = Pricer.black_scholes_price(option)
    assert price == pytest.approx(4.76, abs=0.01)


def test_bs_put_textbook_value():
    """Same parameters; Hull gives p ≈ 0.81 for the put."""
    option = EuropeanOption(S=42, K=40, T=0.5, r=0.10, sigma=0.20, option_type="put")
    price = Pricer.black_scholes_price(option)
    assert price == pytest.approx(0.81, abs=0.01)


# --- Monotonicity: prices should move in the expected direction ---

def test_call_price_increases_with_spot():
    low = EuropeanOption(S=95, K=100, T=0.5, r=0.05, sigma=0.20, option_type="call")
    high = EuropeanOption(S=105, K=100, T=0.5, r=0.05, sigma=0.20, option_type="call")
    assert Pricer.black_scholes_price(high) > Pricer.black_scholes_price(low)


def test_put_price_decreases_with_spot():
    low = EuropeanOption(S=95, K=100, T=0.5, r=0.05, sigma=0.20, option_type="put")
    high = EuropeanOption(S=105, K=100, T=0.5, r=0.05, sigma=0.20, option_type="put")
    assert Pricer.black_scholes_price(high) < Pricer.black_scholes_price(low)


def test_call_price_increases_with_volatility():
    low_vol = EuropeanOption(S=100, K=100, T=0.5, r=0.05, sigma=0.10, option_type="call")
    high_vol = EuropeanOption(S=100, K=100, T=0.5, r=0.05, sigma=0.50, option_type="call")
    assert Pricer.black_scholes_price(high_vol) > Pricer.black_scholes_price(low_vol)


def test_put_price_increases_with_volatility():
    """Vega is positive for both calls and puts — higher vol = higher price."""
    low_vol = EuropeanOption(S=100, K=100, T=0.5, r=0.05, sigma=0.10, option_type="put")
    high_vol = EuropeanOption(S=100, K=100, T=0.5, r=0.05, sigma=0.50, option_type="put")
    assert Pricer.black_scholes_price(high_vol) > Pricer.black_scholes_price(low_vol)


# --- Bounds: computed prices must respect no-arbitrage ---


def test_call_price_within_bounds():
    option = EuropeanOption(S=100, K=100, T=0.5, r=0.05, sigma=0.20, option_type="call")
    price = Pricer.black_scholes_price(option)
    assert option.lower_bound() <= price <= option.upper_bound()


def test_put_price_within_bounds():
    option = EuropeanOption(S=100, K=100, T=0.5, r=0.05, sigma=0.20, option_type="put")
    price = Pricer.black_scholes_price(option)
    assert option.lower_bound() <= price <= option.upper_bound()


def test_call_price_within_bounds_with_dividend():
    option = EuropeanOption(S=100, K=100, T=1.0, r=0.05, sigma=0.20, option_type="call", q=0.03)
    price = Pricer.black_scholes_price(option)
    assert option.lower_bound() <= price <= option.upper_bound()


# --- Put-call parity satisfied by BS prices ---


def test_parity_holds_no_dividend():
    """BS prices on a call and a put with identical parameters must satisfy parity exactly."""
    call = EuropeanOption(S=100, K=100, T=0.5, r=0.05, sigma=0.20, option_type="call")
    put = EuropeanOption(S=100, K=100, T=0.5, r=0.05, sigma=0.20, option_type="put")
    c = Pricer.black_scholes_price(call)
    p = Pricer.black_scholes_price(put)
    # C + K*exp(-rT) = P + S (q=0)
    assert c + 100 * math.exp(-0.05 * 0.5) == pytest.approx(p + 100)


def test_parity_holds_with_dividend():
    call = EuropeanOption(S=100, K=100, T=1.0, r=0.05, sigma=0.20, option_type="call", q=0.02)
    put = EuropeanOption(S=100, K=100, T=1.0, r=0.05, sigma=0.20, option_type="put", q=0.02)
    c = Pricer.black_scholes_price(call)
    p = Pricer.black_scholes_price(put)
    lhs = c + 100 * math.exp(-0.05)
    rhs = p + 100 * math.exp(-0.02)
    assert lhs == pytest.approx(rhs, abs=1e-10)


def test_parity_violation_flags_inconsistent_prices():
    """parity_violation should return True for clearly bad prices."""
    option = EuropeanOption(S=100, K=100, T=0.5, r=0.05, sigma=0.20, option_type="call")
    fair_call = Pricer.black_scholes_price(option)
    wrong_put = fair_call + 5.0  # arbitrary, breaks parity
    assert Pricer.parity_violation(fair_call, wrong_put, option) == True


def test_parity_violation_not_flagged_for_correct_prices():
    call = EuropeanOption(S=100, K=100, T=0.5, r=0.05, sigma=0.20, option_type="call", q=0.02)
    put = EuropeanOption(S=100, K=100, T=0.5, r=0.05, sigma=0.20, option_type="put", q=0.02)
    c = Pricer.black_scholes_price(call)
    p = Pricer.black_scholes_price(put)
    assert Pricer.parity_violation(c, p, call) == False

# --- IV Calculator tests (pytest tests/test_pricer.py -v -k "iv")---

def test_iv_atm():
    true_sigma = 0.2
    option = EuropeanOption(S=100, K=100, T=0.5, r=0.05, sigma= true_sigma, option_type="call", q=0.00)
    bs_price = Pricer.black_scholes_price(option)
    calculated_sigma = Pricer.implied_volatility(bs_price, option)
    assert true_sigma == pytest.approx(calculated_sigma, abs = 1e-5)

def test_iv_no_alter_original():
    true_sigma = 0.2
    option = EuropeanOption(S=100, K=100, T=0.5, r=0.05, sigma= true_sigma, option_type="call", q=0.00)
    bs_price = Pricer.black_scholes_price(option)
    calculated_sigma = Pricer.implied_volatility(bs_price, option)
    assert true_sigma == option.sigma


def test_iv_otm_call():
    true_sigma = 0.2
    option = EuropeanOption(S=100, K=105, T=0.5, r=0.05, sigma= true_sigma, option_type="call", q=0.00)
    bs_price = Pricer.black_scholes_price(option)
    calculated_sigma = Pricer.implied_volatility(bs_price, option)
    assert true_sigma == pytest.approx(calculated_sigma, abs = 1e-5)


def test_iv_otm_put():
    true_sigma = 0.2
    option = EuropeanOption(S=105, K=100, T=0.5, r=0.05, sigma=true_sigma, option_type="put", q=0.00)
    bs_price = Pricer.black_scholes_price(option)
    calculated_sigma = Pricer.implied_volatility(bs_price, option)
    assert true_sigma == pytest.approx(calculated_sigma, abs = 1e-5)

def test_iv_itm_call():
    true_sigma = 0.2
    option = EuropeanOption(S=116, K=100, T=0.5, r=0.05, sigma=true_sigma, option_type="call", q=0.00)
    bs_price = Pricer.black_scholes_price(option)
    calculated_sigma = Pricer.implied_volatility(bs_price, option)
    assert true_sigma == pytest.approx(calculated_sigma, abs = 1e-5)

def test_iv_itm_put():
    true_sigma = 0.2
    option = EuropeanOption(S=90, K=100, T=0.5, r=0.05, sigma=true_sigma, option_type="put", q=0.00)
    bs_price = Pricer.black_scholes_price(option)
    calculated_sigma = Pricer.implied_volatility(bs_price, option)
    assert true_sigma == pytest.approx(calculated_sigma, abs = 1e-5)

def test_iv_atm_dividends():
    true_sigma = 0.2
    option = EuropeanOption(S=100, K=100, T=0.5, r=0.05, sigma=true_sigma, option_type="call", q=0.02)
    bs_price = Pricer.black_scholes_price(option)
    calculated_sigma = Pricer.implied_volatility(bs_price, option)
    assert true_sigma == pytest.approx(calculated_sigma, abs = 1e-5)

def test_iv_atm_dividends_high_sigma():
    true_sigma = 0.92
    option = EuropeanOption(S=100, K=100, T=0.5, r=0.05, sigma=true_sigma, option_type="call", q=0.02)
    bs_price = Pricer.black_scholes_price(option)
    calculated_sigma = Pricer.implied_volatility(bs_price, option)
    assert true_sigma == pytest.approx(calculated_sigma, abs = 1e-5)

def test_iv_atm_high_sigma():
    true_sigma = 0.83
    option = EuropeanOption(S=100, K=100, T=0.5, r=0.05, sigma=true_sigma, option_type="call", q=0.00)
    bs_price = Pricer.black_scholes_price(option)
    calculated_sigma = Pricer.implied_volatility(bs_price, option)
    assert true_sigma == pytest.approx(calculated_sigma, abs = 1e-5)

def test_iv_atm_low_sigma():
    true_sigma = 0.01
    option = EuropeanOption(S=100, K=100, T=0.5, r=0.05, sigma=true_sigma, option_type="call", q=0.00)
    bs_price = Pricer.black_scholes_price(option)
    calculated_sigma = Pricer.implied_volatility(bs_price, option)
    assert true_sigma == pytest.approx(calculated_sigma, abs = 1e-5)

def test_iv_out_of_bounds():
    true_sigma = 0.2
    option = EuropeanOption(S=100, K=100, T=0.5, r=0.05, sigma=true_sigma, option_type="call", q=0.00)
    bs_price_wrong = Pricer.black_scholes_price(option) + 1000
    with pytest.raises(IVSolverError): 
        Pricer.implied_volatility(bs_price_wrong, option)

def test_iv_rejects_negative_price():
    option = EuropeanOption(S=100, K=100, T=0.5, r=0.05, sigma=0.20, option_type="call", q=0.00)
    with pytest.raises(IVSolverError):
        Pricer.implied_volatility(-1.0, option)

def test_iv_deep_itm_call():
    true_sigma = 0.2
    option = EuropeanOption(S=140, K=100, T=0.5, r=0.05, sigma=true_sigma, option_type="call", q=0.00)
    bs_price = Pricer.black_scholes_price(option)
    calculated_sigma = Pricer.implied_volatility(bs_price, option)
    assert true_sigma == pytest.approx(calculated_sigma, abs = 1e-5)

def test_iv_deep_itm_put():
    true_sigma = 0.2
    option = EuropeanOption(S=100, K=50, T=0.5, r=0.05, sigma=true_sigma, option_type="put", q=0.00)
    bs_price = Pricer.black_scholes_price(option)
    calculated_sigma = Pricer.implied_volatility(bs_price, option)
    assert true_sigma == pytest.approx(calculated_sigma, abs = 1e-5)

def test_iv_deep_itm_extreme():
    true_sigma = 0.88
    option = EuropeanOption(S=100, K=35, T=0.5, r=0.05, sigma=true_sigma, option_type="put", q=0.00)
    bs_price = Pricer.black_scholes_price(option)
    calculated_sigma = Pricer.implied_volatility(bs_price, option)
    assert true_sigma == pytest.approx(calculated_sigma, abs = 1e-5)