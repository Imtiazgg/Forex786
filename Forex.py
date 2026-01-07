# --- My Signal
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import requests
import numpy as np
from datetime import datetime
from pytz import timezone
import xml.etree.ElementTree as ET
from dateutil import parser as date_parser
import streamlit.components.v1 as components
import yfinance as yf

st.set_page_config(page_title="Signals", layout="wide")
st.markdown("<h1 style='text-align:center; color:#007acc;'>📊 My Signal</h1>", unsafe_allow_html=True)
st_autorefresh(interval=300000, key="ai_refresh")  # 300000 ms = 5 minutes

API_KEY = "b2a1234a9ea240f9ba85696e2a243403"

symbols = {
    "EUR/USD": "EUR/USD",
    "XAU/USD": "XAU/USD",   # Gold
    "XAG/USD": "XAG/USD",   # Silver
}
def play_rsi_alert():
    components.html("""
    <audio autoplay>
        <source src="https://www.soundjay.com/button/beep-07.wav" type="audio/wav">
    </audio>
    """, height=0)

def fetch_dxy_data():
    try:
        dxy = yf.Ticker("DX-Y.NYB")
        data = dxy.history(period="1d", interval="1m")
        if data.empty:
            raise ValueError("No data received from yfinance")
        current = data["Close"].iloc[-1]
        previous = data["Close"].iloc[0]
        change = current - previous
        percent = (change / previous) * 100
        return current, percent
    except Exception as e:
        dxy_price = 100.237
        dxy_previous = 100.40
        change = dxy_price - dxy_previous
        percent = (change / dxy_previous) * 100
        return dxy_price, percent

def fetch_forex_factory_news():
    url = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"
    response = requests.get(url)
    try:
        root = ET.fromstring(response.content)
    except ET.ParseError as e:
        return []

    news_data = []
    for item in root.findall("./channel/item"):
        try:
            title = item.find("title").text
            pub_time = date_parser.parse(item.find("pubDate").text)
            currency = item.find("{http://www.forexfactory.com/rss}currency").text.strip().upper()
            if pub_time.date() == datetime.utcnow().date():
                news_data.append({
                    "title": title,
                    "time": pub_time,
                    "currency": currency
                })
        except:
            continue
    return news_data

def analyze_impact(title):
    title = title.lower()
    if any(x in title for x in ["cpi", "gdp", "employment", "retail", "core", "inflation", "interest rate"]):
        if any(w in title for w in ["increase", "higher", "rises", "strong", "beats"]):
            return "🟢 Positive"
        elif any(w in title for w in ["decrease", "lower", "falls", "weak", "misses"]):
            return "🔴 Negative"
        else:
            return "🟡 Mixed"
    return "⚪ Neutral"

def get_today_news_with_impact(pair):
    base, quote = pair.split('/')
    quote = quote.upper()
    today_events = []
    for n in news_events:
        if n["currency"] == quote:
            impact = analyze_impact(n["title"])
            time_str = n["time"].strftime("%H:%M")
            today_events.append(f"{n['title']} ({impact}) @ {time_str}")
    return today_events or ["—"]

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_macd(series):
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

def calculate_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calculate_atr(df, period=14):
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift())
    tr3 = abs(df['low'] - df['close'].shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def calculate_adx(df, period=14):
    df['TR'] = np.maximum(df['high'] - df['low'],
                          np.maximum(abs(df['high'] - df['close'].shift()), abs(df['low'] - df['close'].shift())))
    df['+DM'] = np.where((df['high'] - df['high'].shift()) > (df['low'].shift() - df['low']),
                         np.maximum(df['high'] - df['high'].shift(), 0), 0)
    df['-DM'] = np.where((df['low'].shift() - df['low']) > (df['high'] - df['high'].shift()),
                         np.maximum(df['low'].shift() - df['low'], 0), 0)
    tr14 = df['TR'].rolling(window=period).mean()
    plus_dm14 = df['+DM'].rolling(window=period).mean()
    minus_dm14 = df['-DM'].rolling(window=period).mean()
    plus_di14 = 100 * (plus_dm14 / tr14)
    minus_di14 = 100 * (minus_dm14 / tr14)
    dx = 100 * abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14)
    return dx.rolling(window=period).mean()

def detect_candle_pattern(df):
    o, c, h, l = df['open'].iloc[-2:], df['close'].iloc[-2:], df['high'].iloc[-2:], df['low'].iloc[-2:]
    co, cc, ch, cl = o.iloc[-1], c.iloc[-1], h.iloc[-1], l.iloc[-1]
    po, pc = o.iloc[-2], c.iloc[-2]
    body, range_ = abs(cc - co), ch - cl
    if body < range_ * 0.1: return "Doji"
    if pc < po and cc > co and cc > po and co < pc: return "Bullish Engulfing"
    if pc > po and cc < co and cc < po and co > pc: return "Bearish Engulfing"
    if body < range_ * 0.3 and cl < co and cl < cc and (ch - max(co, cc)) < body: return "Hammer"
    if body < range_ * 0.3 and ch > co and ch > cc and (min(co, cc) - cl) < body: return "Shooting Star"
    return ""

def detect_trend_reversal(df):
    e9, e20 = df['EMA9'].iloc[-3:], df['EMA20'].iloc[-3:]
    if e9[0] < e20[0] and e9[1] > e20[1] and e9[2] > e20[2]: return "Reversal Confirmed Bullish"
    if e9[0] > e20[0] and e9[1] < e20[1] and e9[2] < e20[2]: return "Reversal Confirmed Bearish"
    if e9[-2] < e20[-2] and e9[-1] > e20[-1]: return "Reversal Forming Bullish"
    if e9[-2] > e20[-2] and e9[-1] < e20[-1]: return "Reversal Forming Bearish"
    return ""

def detect_divergence(df):
    closes = df['close']
    rsis = df['RSI']
    if len(closes) < 10 or len(rsis) < 10:
        return ""
    recent_price_low = closes.iloc[-5:].idxmin()
    recent_price_high = closes.iloc[-5:].idxmax()
    recent_rsi_low = rsis.iloc[-5:].idxmin()
    recent_rsi_high = rsis.iloc[-5:].idxmax()

    if recent_price_low != recent_rsi_low and closes[recent_price_low] < closes[-1] and rsis[recent_rsi_low] > rsis[-1]:
        return "Bullish Divergence"
    if recent_price_high != recent_rsi_high and closes[recent_price_high] > closes[-1] and rsis[recent_rsi_high] < rsis[-1]:
        return "Bearish Divergence"
    return ""

def generate_ai_suggestion(price, indicators, atr, signal_type):
    if not indicators: return ""
    sl = price - (atr * 1.2) if signal_type == "Bullish" else price + (atr * 1.2)
    tp = price + (atr * 2.5) if signal_type == "Bullish" else price - (atr * 2.5)
    count = len(indicators)
    if count >= 4:
        conf = "Strong"
    elif count == 3:
        conf = "Medium"
    else:
        return ""
    color = "green" if signal_type == "Bullish" else "red"
    signal_txt = f"{conf} <span style='color:{color}'>{signal_type}</span> Signal @ {price:.5f}"
    return f"{signal_txt} | SL: {sl:.5f} | TP: {tp:.5f} | Confidence: {conf}"

news_events = fetch_forex_factory_news()
dxy_price, dxy_change = fetch_dxy_data()
rows = []

for label, symbol in symbols.items():
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval=5min&outputsize=200&apikey={API_KEY}"
    r = requests.get(url).json()
    if "values" not in r:
        continue

    df = pd.DataFrame(r["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index("datetime", inplace=True)
    df = df.astype(float).sort_index()
    df["RSI"] = calculate_rsi(df["close"])
    df["MACD"], df["MACD_Signal"] = calculate_macd(df["close"])
    df["EMA9"] = calculate_ema(df["close"], 9)
    df["EMA20"] = calculate_ema(df["close"], 20)
    df["ADX"] = calculate_adx(df)
    df["ATR"] = calculate_atr(df)
    df.dropna(inplace=True)

    price = df["close"].iloc[-1]
    atr = df["ATR"].iloc[-1]
    trend = "Bullish" if df["EMA9"].iloc[-1] > df["EMA20"].iloc[-1] and price > df["EMA9"].iloc[-1] else \
            "Bearish" if df["EMA9"].iloc[-1] < df["EMA20"].iloc[-1] and price < df["EMA9"].iloc[-1] else "Sideways"

    rsi_val = df["RSI"].iloc[-1]
    indicators = []
    signal_type = ""

    if rsi_val > 50:
        indicators.append("Bullish")
        signal_type = "Bullish"
    elif rsi_val < 50:
        indicators.append("Bearish")
        signal_type = "Bearish"

    if df["MACD"].iloc[-1] > df["MACD_Signal"].iloc[-1]:
        indicators.append("MACD")
    if df["EMA9"].iloc[-1] > df["EMA20"].iloc[-1] and price > df["EMA9"].iloc[-1]:
        indicators.append("EMA")
    if df["ADX"].iloc[-1] > 20:
        indicators.append("ADX")

    pattern = detect_candle_pattern(df)
    if pattern:
        indicators.append("Candle")

    divergence = detect_divergence(df)
    if divergence:
        indicators.append("Divergence")
        play_rsi_alert()

    suggestion = generate_ai_suggestion(price, indicators, atr, signal_type)

    rows.append({
    "Pair": label, 
    "Price": round(price, 5), 
    "RSI": round(rsi_val, 2),
    "ATR Status": "🔴 Low" if atr < 0.0004 else "🟡 Normal" if atr < 0.0009 else "🟢 High",
    "Trend": trend, 
    "Reversal Signal": detect_trend_reversal(df),
    "Signal Type": signal_type, 
    "Confirmed Indicators": ", ".join(indicators),
    "AI Suggestion": suggestion,
    "DXY Impact": f"{dxy_price:.2f} ({dxy_change:+.2f}%)" if "USD" in label and dxy_price is not None else "—",
    "Divergence": divergence or "—",
    "Upcoming News & Impact": "\n".join(get_today_news_with_impact(label))  # <-- NEW
})
column_order = ["Pair", "Price", "RSI", "ATR Status", "Trend", "Reversal Signal",
                "Signal Type", "Confirmed Indicators", "AI Suggestion",
                "DXY Impact","News"]

df_result = pd.DataFrame(rows)
df_result["Score"] = df_result["AI Suggestion"].apply(lambda x: 3 if "Strong" in x else 2 if "Medium" in x else 0)
df_sorted = df_result.sort_values(by="Score", ascending=False).drop(columns=["Score"])

styled_html = "<table style='width:100%; border-collapse: collapse;'>"
styled_html += "<tr>" + "".join([
    f"<th style='border:1px solid #ccc; padding:6px; background:#e0e0e0'>{col}</th>" for col in column_order]) + "</tr>"

for _, row in df_sorted.iterrows():
    style = 'background-color: #d4edda;' if "Strong" in row["AI Suggestion"] else \
            'background-color: #d1ecf1;' if "Medium" in row["AI Suggestion"] else ''
    styled_html += f"<tr style='{style}'>"
    for col in column_order:
        val = row[col]
        if col == "Pair":
            val = f"<strong style='font-size: 18px;'>{val}</strong>"
        elif col == "Trend":
            color = 'green' if row['Trend'] == 'Bullish' else 'red' if row['Trend'] == 'Bearish' else 'gray'
            val = f"<span style='color:{color}; font-weight:bold;'>{row['Trend']}</span>"
        elif col == "Signal Type":
            color = 'green' if row['Signal Type'] == 'Bullish' else 'red'
            val = f"<span style='color:{color}; font-weight:bold;'>{row['Signal Type']}</span>"
        elif col == "RSI":
            color = "red" if row["RSI"] > 75 else "green" if row["RSI"] < 20 else "black"
            val = f"<span style='color:{color}; font-weight:bold;'>{row['RSI']}</span>"
        elif col == "DXY Impact" and row["DXY Impact"] != "—":
            dxy_color = "green" if '+' in row["DXY Impact"] else "red"
            val = f"<span style='color:{dxy_color}; font-weight:bold;'>{row['DXY Impact']}</span>"
        elif col == "Divergence" and row["Divergence"] != "—":
            div_color = "green" if "Bullish" in row["Divergence"] else "red"
            val = f"<span style='color:{div_color}; font-weight:bold;'>{row['Divergence']}</span>"
        styled_html += f"<td style='border:1px solid #ccc; padding:6px; white-space:pre-wrap;'>{val}</td>"
    styled_html += "</tr>"

styled_html += "</table>"
st.markdown(styled_html, unsafe_allow_html=True)
st.caption(f"Timeframe: 5-Min | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.text(f"Scanned Pairs: {len(rows)}")
st.text(f"Strong Signals Found: {len([r for r in rows if 'Strong' in r['AI Suggestion']])}")




