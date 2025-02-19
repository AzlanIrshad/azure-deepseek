import yfinance as yf
import pandas as pd
import numpy as np

# Function to fetch stock data
def get_stock_data(ticker="AAPL", start="2023-01-01", end="2025-02-18"):
    stock = yf.Ticker(ticker)
    df = stock.history(start=start, end=end)
    df["Price Change %"] = df["Close"].pct_change()
    df["Target"] = np.where(df["Price Change %"] > 0, 1, 0)  # 1 = Up, 0 = Down
    df = df.dropna()
    return df

# Fetch data and save it for training
if __name__ == "__main__":
    df = get_stock_data()
    df.to_csv("stock_data.csv", index=False)
    print("Stock data saved successfully!")
