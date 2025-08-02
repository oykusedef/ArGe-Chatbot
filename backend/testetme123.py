import yfinance as yf
ticker = yf.Ticker("AKBNK.IS")
hist = ticker.history(period="1y")
print(hist.head())
print(hist.tail())
print("Satır sayısı:", len(hist))
print(hist[['Close']].isna().sum())