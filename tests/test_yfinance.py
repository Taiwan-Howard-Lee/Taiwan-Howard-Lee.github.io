import yfinance as yf

# Get data for Apple (AAPL)
ticker = yf.Ticker("AAPL")

# Get historical market data
hist = ticker.history(period="1mo")

print(hist)

# Get company information
print(ticker.info)
