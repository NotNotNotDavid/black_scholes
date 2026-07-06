import yfinance as yf
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

# collect.py lives at: options-project/src/pipeline/collect.py
# parents[0] = pipeline, parents[1] = src, parents[2] = options-project
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

def collect_one_expiry(tk, expiry, snapshot_ts, S, r):

    chain = tk.option_chain(expiry)

    calls_df = chain.calls.copy()
    calls_df["option_type"] = "call"

    puts_df = chain.puts.copy()
    puts_df["option_type"] = "put"

    both_df = pd.concat([calls_df, puts_df], ignore_index=True)

    both_df = both_df.rename(columns={
        "strike": "K",
        "openInterest": "open_interest",
        "impliedVolatility": "iv_yfinance",  # stored as a baseline, not trusted, will use own calculations
    })

    both_df["mid"] = (both_df["bid"] + both_df["ask"]) / 2
    both_df["expiration"] = pd.to_datetime(expiry)
    both_df["snapshot_ts"] = snapshot_ts
    both_df["snapshot_date"] = pd.to_datetime(snapshot_ts.date())
    both_df["S"] = S
    both_df["r"] = r
    both_df["T"] = (both_df["expiration"] - both_df["snapshot_date"]).dt.days / 365.0


    cols = ["snapshot_date", "snapshot_ts", "option_type", "expiration", "T",
            "K", "bid", "ask", "mid", "volume", "open_interest", "S", "r", "iv_yfinance"]
    return both_df[cols]


def main():
    tk = yf.Ticker("^SPX")
    snapshot_ts = datetime.now(timezone.utc)

    S = tk.fast_info["lastPrice"]
    r = 0.037  # placeholder until FRED data, as of July 1st's data

    frame = []

    for expiry in tk.options:
        try:
            df = collect_one_expiry(tk, expiry, snapshot_ts, S, r)
            frame.append(df)

        except Exception as e:
            print(f"Skipped {expiry}: {e}")
            continue

    dayfile = pd.concat(frame, ignore_index=True)

    RAW_DIR.mkdir(parents=True, exist_ok=True)   # creates data/raw/ if missing
    filepath = RAW_DIR / f"spx_{snapshot_ts.date()}.parquet"
    dayfile.to_parquet(filepath, index=False)

    print(f"Saved snapshot to {filepath}")
    print(f"{len(dayfile)} rows collected across {len(tk.options)} expiries")
    print(dayfile.head())


if __name__ == "__main__":
    main()