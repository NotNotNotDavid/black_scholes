# Options Pricing & Volatility Analysis Framework

A Python project for pricing European options and analyzing implied
volatility, intended as the foundation for an empirical study of the
variance risk premium in SPX index options.

## Status

This project is still in development! The phases below indicate what
has been built and what is planned (and also might change!).

| Component                            | Status          |
| ------------------------------------ | --------------- |
| Black-Scholes pricer (with dividend yield) | Complete  |
| Put-call parity & no-arbitrage bounds | Complete       |
| Implied volatility solver (Newton-Raphson + Brent fallback) | Complete |
| Greeks (Delta, Gamma, Vega, Theta, Rho, second-order) | In progress |
| Implied volatility surface construction | Planned     |
| Variance risk premium empirical study | Planned        |
| PCA of IV surface dynamics           | Planned         |

Last updated: Jun 23th 2026

## Research Direction

The empirical question driving this project: **does the implied
volatility embedded in SPX options systematically overstate subsequently
realized volatility, and what does the structure of that bias look like
across strikes and maturities?**

This phenomenon, the variance risk premium, is well-documented in the
academic literature. The project aims to characterize it on recent SPX
data to reveal that human fear is a important bias to take account to.

## Current Capabilities

**Pricing.** Closed-form Black-Scholes-Merton with continuous dividend
yield, appropriate for index options. Handles calls and puts, validates
inputs at construction, and computes no-arbitrage bounds and put-call
parity counterpart prices directly on the option object.

**Implied volatility inversion.** Newton-Raphson with analytical vega as
the derivative, falling back to Brent's method when Newton fails (deep
ITM/OTM, near-expiry, or vega ~ 0 cases). Rejects prices outside the
no-arbitrage bounds.


## Repository layout

```text
.
├── src/
│   ├── EuropeanOption.py    # Option contract: bounds, parity, intrinsic value
│   └── Pricer.py            # Black-Scholes, vega, implied volatility solver
└── tests/
    ├── test_european_option.py
    └── test_pricer.py
conftest.py            # pytest path configuration

```

## Running the code

```bash
pip install numpy scipy pytest
pytest tests/          # run the validation suite
```

Python 3.10+ 
Dependencies: numpy, scipy, and pytest for the tests

## Methodology notes

**Dividend convention.** We use a continuous dividend yield
`q`, which is the standard convention for index options (SPX, NDX) and
the model nearly used by all major data sources. This
differs from Hull's dividend treatment in Chapter 11, which
applies to short-dated single-name options with known dividend dates.
For the SPX, the empirical work this project targets, continuous
yield is the appropriate choice.

**IV solver design.** Newton-Raphson alone fails on a meaningful
percentage of real world options due to vanishing vega in deep
ITM/OTM regimes. The Brent fallback handles these cases. For very deep
ITM or OTM, IV is genuinely not recoverable to high precision because
the price is constant in sigma to floating-point accuracy. This is a
limitation of the inversion problem, not the solver.

**Model limitations.** Black-Scholes assumes constant volatility,
log-normal returns, and no jumps. The volatility observed in real
SPX data is direct evidence of model misspecification. This project
uses BS as a baseline. 

## Limitations

**American/European Optioins.** This project will mainly focus on European options.
As extensions to American options via binomial trees can be considered in the future.


## References

- Hull, J. C. *Options, Futures, and Other Derivatives*, Global ed. 11th ed.

---

*This is an ongoing project! The README will be updated as components
are completed.*