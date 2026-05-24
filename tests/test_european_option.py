import numpy as np
import pytest
from EuropeanOption import EuropeanOption

# to run tests: pytest tests/ in terminal

def test_call_intrinsic_value_in_the_money():
    option = EuropeanOption(S=120, K=100, T=0.5, r=0.05, sigma=0.20, option_type="call")
    assert option.intrinsic_value() == 20.0

def test_call_intrinsic_value_otm():
    option = EuropeanOption(S=80, K=100, T=0.5, r=0.05, sigma=0.20, option_type="call")
    assert option.intrinsic_value() == 0.0


def test_call_intrinsic_value_atm():
    option = EuropeanOption(S=100, K=100, T=0.5, r=0.05, sigma=0.20, option_type="call")
    assert option.intrinsic_value() == 0.0


def test_put_intrinsic_value_itm():
    option = EuropeanOption(S=80, K=100, T=0.5, r=0.05, sigma=0.20, option_type="put")
    assert option.intrinsic_value() == 20.0


def test_put_intrinsic_value_otm():
    option = EuropeanOption(S=120, K=100, T=0.5, r=0.05, sigma=0.20, option_type="put")
    assert option.intrinsic_value() == 0.0

def test_call_upper_bound_no_dividend():
    option = EuropeanOption(S=100, K=100, T=1.0, r=0.05, sigma=0.20, option_type="call")
    assert option.upper_bound() == 100.0


def test_call_upper_bound_with_dividend():
    option = EuropeanOption(S=100, K=100, T=1.0, r=0.05, sigma=0.20, option_type="call", q=0.03)
    expected = 100 * np.exp(-0.03)
    assert option.upper_bound() == pytest.approx(expected)


def test_call_upper_bound_decreases_with_dividend():
    """Higher dividend yield → lower call upper bound."""
    no_div = EuropeanOption(S=100, K=100, T=1.0, r=0.05, sigma=0.20, option_type="call", q=0.0)
    with_div = EuropeanOption(S=100, K=100, T=1.0, r=0.05, sigma=0.20, option_type="call", q=0.03)
    assert with_div.upper_bound() < no_div.upper_bound()


def test_put_upper_bound():
    option = EuropeanOption(S=100, K=100, T=1.0, r=0.05, sigma=0.20, option_type="put")
    assert option.upper_bound() == pytest.approx(100 * np.exp(-0.05))


def test_call_lower_bound_itm():
    option = EuropeanOption(S=120, K=100, T=1.0, r=0.05, sigma=0.20, option_type="call")
    expected = 120 - 100 * np.exp(-0.05)
    assert option.lower_bound() == pytest.approx(expected)


def test_call_lower_bound_clamped_to_zero():
    option = EuropeanOption(S=50, K=100, T=1.0, r=0.05, sigma=0.20, option_type="call")
    assert option.lower_bound() == 0.0


def test_put_lower_bound_itm():
    option = EuropeanOption(S=80, K=100, T=1.0, r=0.05, sigma=0.20, option_type="put")
    expected = 100 * np.exp(-0.05) - 80
    assert option.lower_bound() == pytest.approx(expected)

def test_parity_counterpart_no_dividend():
    """Given a call price, the derived put should satisfy the parity equation."""
    call = EuropeanOption(S=100, K=100, T=0.5, r=0.05, sigma=0.20, option_type="call")
    c_price = 7.0
    p_price = call.parity_counterpart_price(c_price)
    # Verify: C + K*exp(-rT) = P + S (q=0)
    assert c_price + 100 * np.exp(-0.05 * 0.5) == pytest.approx(p_price + 100)


def test_parity_counterpart_with_dividend():
    call = EuropeanOption(S=100, K=100, T=1.0, r=0.05, sigma=0.20, option_type="call", q=0.02)
    c_price = 7.0
    p_price = call.parity_counterpart_price(c_price)
    lhs = c_price + 100 * np.exp(-0.05)
    rhs = p_price + 100 * np.exp(-0.02)
    assert lhs == pytest.approx(rhs)


def test_parity_roundtrip():
    """call → put → call should recover the original price."""
    call = EuropeanOption(S=100, K=100, T=0.5, r=0.05, sigma=0.20, option_type="call", q=0.02)
    put = EuropeanOption(S=100, K=100, T=0.5, r=0.05, sigma=0.20, option_type="put", q=0.02)
    original = 7.0
    derived = call.parity_counterpart_price(original)
    recovered = put.parity_counterpart_price(derived)
    assert recovered == pytest.approx(original)

def test_accepts_call_and_put():
    EuropeanOption(S=100, K=100, T=0.5, r=0.05, sigma=0.20, option_type="call")
    EuropeanOption(S=100, K=100, T=0.5, r=0.05, sigma=0.20, option_type="put")


def test_case_insensitive_option_type():
    option = EuropeanOption(S=100, K=100, T=0.5, r=0.05, sigma=0.20, option_type="CALL")
    assert option.option_type == "call"


def test_rejects_invalid_option_type():
    with pytest.raises(ValueError):
        EuropeanOption(S=100, K=100, T=0.5, r=0.05, sigma=0.20, option_type="straddle")


def test_dividend_defaults_to_zero():
    """Backward-compat: not passing q should be the same as q=0."""
    option = EuropeanOption(S=100, K=100, T=0.5, r=0.05, sigma=0.20, option_type="call")
    assert option.q == 0.0