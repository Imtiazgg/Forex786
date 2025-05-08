# --- Signals with H & I (15-Min Timeframe | ATR + Candle Pattern Name) ---
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import requests
import numpy as np
from datetime import datetime, timedelta
from scipy.signal import argrelextrema
from pytz import timezone

st.set_page_config(page_title="Forex AI Signals", layout="wide")
st.markdown("<h1 style='text-align:center; color:#007acc;'>📊 Forex AI Signals (15-Min Timeframe)</h1>",
            unsafe_allow_html=True)
st_autorefresh(interval=120000, key="auto_refresh")

API_KEY = "b2a1234a9ea240f9ba85696e2a243403"
symbols = {
    "EUR/USD": "EUR/USD", "GBP/USD": "GBP/USD", "USD/JPY": "USD/JPY",
    "AUD/USD": "AUD/USD", "USD/CAD": "USD/CAD", "USD/CHF": "USD/CHF",
    "XAU/USD": "XAU/USD", "WTI/USD": "WTI/USD", "EUR/JPY": "EUR/JPY", "NZD/USD": "NZD/USD",
    "EUR/GBP": "EUR/GBP", "EUR/CAD": "EUR/CAD",
    "GBP/JPY": "GBP/JPY", "EUR/AUD": "EUR/AUD", "AUD/JPY": "AUD/JPY",
    "GBP/NZD": "GBP/NZD", "EUR/NZD": "EUR/NZD",  "XAG/USD": "XAG/USD",  # ✅ Silver added

}
news_events = {
    "EUR/USD": [{"time": "10:30", "title": "Euro CPI Data"}],
    "GBP/USD": [{"time": "11:00", "title": "BoE Governor Speech"}],
    "USD/JPY": [{"time": "13:00", "title": "US Jobless Claims"}],
    "AUD/USD": [{"time": "08:00", "title": "RBA Statement"}],
    "USD/CAD": [{"time": "15:00", "title": "Canada Trade Balance"}],
    "USD/CHF": [{"time": "14:00", "title": "US Fed Chair Remarks"}],
    "XAU/USD": [{"time": "13:30", "title": "Gold Reserve Report"}],
    "WTI/USD": [{"time": "12:30", "title": "Crude Oil Inventory"}],
    "EUR/JPY": [{"time": "09:00", "title": "ECB Bulletin"}],
    "NZD/USD": [{"time": "07:30", "title": "NZ Employment Report"}],
    "EUR/GBP": [{"time": "10:00", "title": "UK & Euro Economic Sentiment"}],
    "EUR/CAD": [{"time": "14:30", "title": "ECB & BoC Press Conference"}],
    "GBP/JPY": [{"time": "11:30", "title": "UK Inflation & JPY Trade Balance"}],
    "EUR/AUD": [{"time": "09:45", "title": "Eurozone GDP vs Australia CPI"}],
    "AUD/JPY": [{"time": "08:15", "title": "RBA vs BOJ Rate Outlook"}],
    "GBP/NZD": [{"time": "07:50", "title": "BoE vs RBNZ Rate Forecast"}],
    "EUR/NZD": [{"time": "10:20", "title": "Euro Retail vs NZ Growth Data"}],
    "XAG/USD": [{"time": "13:45", "title": "Silver Market Update"}],
}
def fetch_data(symbol, interval="15min", outputsize=200):
    url = "https://api.twelvedata.com/time_series"
    params = {"symbol": symbol, "interval": interval, "outputsize": outputsize, "apikey": API_KEY}
    r = requests.get(url, params=params)
    data = r.json()
    if "values" not in data:
        return None
    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index("datetime", inplace=True)
    df = df.astype(float).sort_index()
    return df


def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calculate_atr(df, period=14):
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift())
    tr3 = abs(df['low'] - df['close'].shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr


def calculate_macd(series):
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal


def calculate_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


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
    adx = dx.rolling(window=period).mean()
    return adx


def detect_candle_pattern(df):
    o, c, h, l = df['open'].iloc[-2:], df['close'].iloc[-2:], df['high'].iloc[-2:], df['low'].iloc[-2:]
    current_open = o.iloc[-1]
    current_close = c.iloc[-1]
    current_high = h.iloc[-1]
    current_low = l.iloc[-1]

    body = abs(current_close - current_open)
    range_ = current_high - current_low

    previous_open = o.iloc[-2]
    previous_close = c.iloc[-2]

    if body < range_ * 0.1:
        return "Doji"
    if previous_close < previous_open and current_close > current_open and current_close > previous_open and current_open < previous_close:
        return "Bullish Engulfing"
    if previous_close > previous_open and current_close < current_open and current_close < previous_open and current_open > previous_close:
        return "Bearish Engulfing"
    if body < range_ * 0.3 and (current_low < current_open and current_low < current_close) and (
            current_high - max(current_open, current_close)) < body:
        return "Hammer"
    if body < range_ * 0.3 and (current_high > current_open and current_high > current_close) and (
            min(current_open, current_close) - current_low) < body:
        return "Shooting Star"
    return ""


def detect_trend_reversal(df):
    if len(df) < 3:
        return ""
    e9 = df['EMA9'].iloc[-3:]
    e20 = df['EMA20'].iloc[-3:]
    if e9[0] < e20[0] and e9[1] > e20[1] and e9[2] > e20[2]:
        return "Reversal Confirmed Bullish"
    elif e9[0] > e20[0] and e9[1] < e20[1] and e9[2] < e20[2]:
        return "Reversal Confirmed Bearish"
    elif e9[-2] < e20[-2] and e9[-1] > e20[-1]:
        return "Reversal Forming Bullish"
    elif e9[-2] > e20[-2] and e9[-1] < e20[-1]:
        return "Reversal Forming Bearish"
    return ""
def generate_ai_suggestion(price, indicators, atr):
    if not indicators:
        return ""
    sl = price - (atr * 1.2) if "Bullish" in indicators else price + (atr * 1.2)
    tp = price + (atr * 2.5) if "Bullish" in indicators else price - (atr * 2.5)
    count = len(indicators)
    if count >= 4:
        confidence = "Strong"
    elif count == 3:
        confidence = "Medium"
    elif count == 2:
        confidence = "Low"
    else:
        return ""
    return f"{confidence} Signal @ {price:.5f} | SL: {sl:.5f} | TP: {tp:.5f} | Confidence: {confidence}"


def check_news_alert(pair):
    now = datetime.now(timezone('Asia/Karachi'))
    alert_list = []
    for event in news_events.get(pair, []):
        try:
            event_time = datetime.strptime(event["time"], "%H:%M").replace(
                year=now.year, month=now.month, day=now.day, tzinfo=timezone('Asia/Karachi'))
            if timedelta(0) <= (event_time - now) <= timedelta(minutes=30):
                alert_list.append(f"{event['title']} @ {event['time']}")
        except:
            continue
    return " | ".join(alert_list) if alert_list else ""


# ---------------- Table Rows ---------------- #
rows = []
for label, symbol in symbols.items():
    df = fetch_data(symbol, interval="15min")
    if df is not None:
        df['RSI'] = calculate_rsi(df['close'])
        df['MACD'], df['MACD_Signal'] = calculate_macd(df['close'])
        df['EMA9'] = calculate_ema(df['close'], 9)
        df['EMA20'] = calculate_ema(df['close'], 20)
        df['ADX'] = calculate_adx(df)
        df['ATR'] = calculate_atr(df)
        df = df.dropna()

        price_now = df['close'].iloc[-1]
        atr_value = df['ATR'].iloc[-1]
        atr_status = "🔴 Low" if atr_value < 0.0004 else "🟡 Normal" if atr_value < 0.0009 else "🟢 High"

        reversal = detect_trend_reversal(df)
        pattern = detect_candle_pattern(df)
        candle_pattern = pattern if pattern else "—"

        indicators = []
        if df['RSI'].iloc[-1] > 50:
            indicators.append("Bullish")
        elif df['RSI'].iloc[-1] < 50:
            indicators.append("Bearish")

        if df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1]:
            indicators.append("MACD")
        if df['EMA9'].iloc[-1] > df['EMA20'].iloc[-1] and price_now > df['EMA9'].iloc[-1]:
            indicators.append("EMA")
        if df['ADX'].iloc[-1] > 20:
            indicators.append("ADX")
        if pattern:
            indicators.append("Candle")

        trend = (
            "Bullish" if df['EMA9'].iloc[-1] > df['EMA20'].iloc[-1] and price_now > df['EMA9'].iloc[-1]
            else "Bearish" if df['EMA9'].iloc[-1] < df['EMA20'].iloc[-1] and price_now < df['EMA9'].iloc[-1]
            else "Sideways"
        )

        ai_suggestion = generate_ai_suggestion(price_now, indicators, atr_value)

        rows.append({
            "Pair": label, "Price": round(price_now, 5), "RSI": round(df['RSI'].iloc[-1], 2),
            "ATR": round(atr_value, 5), "ATR Status": atr_status,
            "Trend": trend,
            "Reversal Signal": reversal,
            "Confirmed Indicators": ", ".join(indicators),
            "Candle Pattern": candle_pattern,
            "AI Suggestion": ai_suggestion,
            "News Alert": check_news_alert(label)
        })
# ---------------- Table Layout ---------------- #
column_order = [
    "Pair", "Price", "RSI", "ATR", "ATR Status", "Trend", "Reversal Signal",
    "Confirmed Indicators", "Candle Pattern", "AI Suggestion", "News Alert"
]

styled_html = "<table style='width:100%; border-collapse: collapse;'>"
styled_html += "<tr>" + "".join([
    f"<th style='border: 1px solid #ccc; padding: 6px; background-color:#e0e0e0'>{col}</th>"
    for col in column_order
]) + "</tr>"


def style_row(row):
    ai = row['AI Suggestion']
    trend = row['Trend']

    if pd.notna(ai) and "Confidence: Strong" in ai:
        return 'background-color: #d4edda;'  # Light Green
    if pd.notna(ai) and "Confidence: Medium" in ai:
        return 'background-color: #d1ecf1;'  # Light Blue
    if "Reversal" in row['Reversal Signal']:
        return 'background-color: #fff0b3;'  # Yellow for Reversal Signals
    return ''

def trend_color_text(trend):
    color = "green" if trend == "Bullish" else "red" if trend == "Bearish" else "gray"
    return f"<span style='color:{color}; font-weight:bold;'>{trend}</span>"


df_result = pd.DataFrame(rows)
if "Pair" in df_result.columns:
    df_sorted = df_result.sort_values(by="Pair", na_position='last')
else:
    df_sorted = df_result

for _, row in df_sorted.iterrows():
    style = style_row(row)
    styled_html += f"<tr style='{style}'>"
    for col in column_order:
        val = row[col]
        if col == "Pair":
            val = f"<strong style='font-size: 18px;'>{val}</strong>"
        elif col == "Trend":
            val = trend_color_text(val)
        styled_html += f"<td style='border: 1px solid #ccc; padding: 6px;'>{val}</td>"
    styled_html += "</tr>"

styled_html += "</table>"
st.markdown(styled_html, unsafe_allow_html=True)

st.caption(f"Timeframe: 15-Min | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.text(f"Scanned Pairs: {len(rows)}")
strongs = [r for r in rows if "Confidence: Strong" in r["AI Suggestion"]]
st.text(f"Strong Signals Found: {len(strongs)}")
