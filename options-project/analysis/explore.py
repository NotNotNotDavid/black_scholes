import yfinance as yf
tk = yf.Ticker("^SPX")
print(tk.options)                    # expiries, or empty tuple?
chain = tk.option_chain(tk.options[0])
print(chain.calls.columns.tolist())
print(chain.calls[["strike","bid","ask","volume","openInterest","impliedVolatility"]].head())