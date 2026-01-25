from flask import Flask, request, jsonify
import yfinance as yf
import pandas as pd
import numpy as np

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze_stock():
    data = request.get_json()
    ticker = data.get('ticker', 'AAPL')
    
    print(f"Python: Analyzing {ticker}...")
    
    # Download stock data
    stock = yf.Ticker(ticker)
    hist = stock.history(period="3mo")
    
    # Calculate simple indicators
    current_price = float(hist['Close'].iloc[-1])
    avg_price = float(hist['Close'].mean())
    volatility = float(hist['Close'].std())
    
    # Simple prediction logic
    if current_price < avg_price * 0.95:
        signal = "BUY"
        confidence = 0.72
    elif current_price > avg_price * 1.05:
        signal = "SELL"
        confidence = 0.68
    else:
        signal = "HOLD"
        confidence = 0.55
    
    result = {
        "service": "Python ML",
        "ticker": ticker,
        "current_price": round(current_price, 2),
        "average_price": round(avg_price, 2),
        "volatility": round(volatility, 2),
        "signal": signal,
        "confidence": confidence,
        "indicators": {
            "rsi": round(np.random.uniform(30, 70), 2),  # Simplified for demo
            "macd": round(np.random.uniform(-2, 2), 2)
        }
    }
    
    print(f"Python: Returning {signal} signal")
    return jsonify(result)

if __name__ == '__main__':
    print("Python service starting on port 5001...")
    app.run(host='0.0.0.0', port=5001, debug=True)
