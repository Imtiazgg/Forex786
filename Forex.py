# --- My Signal (Updated)
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import requests
import numpy as np
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET
from dateutil import parser as date_parser
import streamlit.components.v1 as components
import yfinance as yf

st.set_page_config(page_title="Signals", layout="wide")
st.markdown("<h1 style='text-align:center; color:#007acc;'>📊 My Signal</h1>", unsafe_allow_html=True)
st_autorefresh(interval=300000, key="ai_refresh")  # 5-min refresh

# Prefer putting this in .streamlit/secrets.toml as:
# TWELVEDATA_API_KEY = "your_api_key"
API_KEY = st.secrets["TWELVEDATA_API_KEY"]
symbols = {
    "EUR/USD": "EUR/USD",
    "XAU/USD": "XAU/USD",
    "XAG/USD": "XAG/USD",
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
            raise ValueError("No data received")
        current = float(data["Close"].iloc[-1])
        previous = float(data["Close"].iloc[0])
        change = current - previous
        percent = (change / previous) * 100 if previous != 0 else 0
        return current, percent
    except Exception:
        dxy_price = 100.237
        dxy_previous = 100.40
        change = dxy_price - dxy_previous
        percent = (change / dxy_previous) * 100 if dxy_previous != 0 else 0
        return dxy_price, percent


def fetch_forex_factory_news():
    url = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        root = ET.fromstring(response.content)
    except Exception:
        return []

    news_data = []
    for item in root.findall("./channel/item"):
        try:
            title = item.find("title").text
            pub_time = date_parser.parse(item.find("pubDate").text)
            currency_tag = item.find("{http://www.forexfactory.com/rss}currency")
            currency = currency_tag.text.strip().upper() if currency_tag is not None and currency_tag.text else ""
            news_data.append({"title": title, "time": pub_time, "currency": currency})
        except Exception:
            continue
    return news_data


def analyze_impact(title):
    title = title.lower()
    if any(x in title for x in ["cpi", "gdp", "employment", "retail", "core", "inflation", "interest rate", "nfp", "pmi"]):
        if any(w in title for w in ["increase", "higher", "rises", "strong", "beats", "hawkish"]):
            return "🟢 Positive"
        elif any(w in title for w in ["decrease", "lower", "falls", "weak", "misses", "dovish"]):
            return "🔴 Negative"
        else:
            return "🟡 Mixed"
    return "⚪ Neutral"


def get_upcoming_news_with_impact(pair):
    """Return news from now until next 24h for the given pair."""
    try:
        _, quote = pair.split('/')
        quote = quote.upper()
    except ValueError:
        return ["—"]

    now = datetime.utcnow()
    next_24h = now + timedelta(hours=24)
    upcoming_events = []

    for n in news_events:
        if n["currency"] == quote and now <= n["time"].replace(tzinfo=None) <= next_24h:
            impact = analyze_impact(n["title"])
            time_str = n["time"].strftime("%Y-%m-%d %H:%M")
            upcoming_events.append(f"{n['title']} ({impact}) @ {time_str}")

    return upcoming_events if upcoming_events else ["—"]


def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


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
    tr2 = (df['high'] - df['close'].shift()).abs()
    tr3 = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def calculate_adx(df, period=14):
    temp = df.copy()
    temp['TR'] = np.maximum(
        temp['high'] - temp['low'],
        np.maximum((temp['high'] - temp['close'].shift()).abs(), (temp['low'] - temp['close'].shift()).abs())
    )
    temp['+DM'] = np.where(
        (temp['high'] - temp['high'].shift()) > (temp['low'].shift() - temp['low']),
        np.maximum(temp['high'] - temp['high'].shift(), 0),
        0
    )
    temp['-DM'] = np.where(
        (temp['low'].shift() - temp['low']) > (temp['high'] - temp['high'].shift()),
        np.maximum(temp['low'].shift() - temp['low'], 0),
        0
    )

    tr14 = temp['TR'].rolling(window=period).mean()
    plus_dm14 = temp['+DM'].rolling(window=period).mean()
    minus_dm14 = temp['-DM'].rolling(window=period).mean()

    plus_di14 = 100 * (plus_dm14 / tr14.replace(0, np.nan))
    minus_di14 = 100 * (minus_dm14 / tr14.replace(0, np.nan))

    dx = 100 * (plus_di14 - minus_di14).abs() / (plus_di14 + minus_di14).replace(0, np.nan)
    return dx.rolling(window=period).mean()


def detect_trend_reversal(df):
    if len(df) < 3:
        return ""
    e9 = df['EMA9'].iloc[-3:].tolist()
    e20 = df['EMA20'].iloc[-3:].tolist()

    if e9[0] < e20[0] and e9[1] > e20[1] and e9[2] > e20[2]:
        return "Reversal Confirmed Bullish"
    if e9[0] > e20[0] and e9[1] < e20[1] and e9[2] < e20[2]:
        return "Reversal Confirmed Bearish"
    if e9[1] < e20[1] and e9[2] > e20[2]:
        return "Reversal Forming Bullish"
    if e9[1] > e20[1] and e9[2] < e20[2]:
        return "Reversal Forming Bearish"
    return ""


def detect_divergence(df):
    if len(df) < 10:
        return ""

    closes = df['close'].iloc[-5:]
    rsis = df['RSI'].iloc[-5:]

    price_low_idx = closes.idxmin()
    price_high_idx = closes.idxmax()
    rsi_low_idx = rsis.idxmin()
    rsi_high_idx = rsis.idxmax()

    current_close = df['close'].iloc[-1]
    current_rsi = df['RSI'].iloc[-1]

    if price_low_idx != rsi_low_idx:
        if df.loc[price_low_idx, 'close'] < current_close and df.loc[rsi_low_idx, 'RSI'] > current_rsi:
            return "Bullish Divergence"

    if price_high_idx != rsi_high_idx:
        if df.loc[price_high_idx, 'close'] > current_close and df.loc[rsi_high_idx, 'RSI'] < current_rsi:
            return "Bearish Divergence"

    return ""


def add_liquidity_features(df, lookback=20):
    temp = df.copy()

    temp["liq_high"] = temp["high"].rolling(lookback).max().shift(1)
    temp["liq_low"] = temp["low"].rolling(lookback).min().shift(1)

    temp["sweep_high"] = (temp["high"] > temp["liq_high"]) & (temp["close"] < temp["liq_high"])
    temp["sweep_low"] = (temp["low"] < temp["liq_low"]) & (temp["close"] > temp["liq_low"])

    return temp


def add_bos_features(df, swing_lookback=5):
    temp = df.copy()

    temp["prev_swing_high"] = temp["high"].rolling(swing_lookback).max().shift(1)
    temp["prev_swing_low"] = temp["low"].rolling(swing_lookback).min().shift(1)

    temp["bos_up"] = temp["close"] > temp["prev_swing_high"]
    temp["bos_down"] = temp["close"] < temp["prev_swing_low"]

    return temp


def get_smart_money_signal(df, trend):
    last = df.iloc[-1]

    if last["sweep_low"] and last["bos_up"]:
        return "STRONG BUY 🔥", "Sell-side liquidity taken + BOS up"
    elif last["sweep_high"] and last["bos_down"]:
        return "STRONG SELL 🔥", "Buy-side liquidity taken + BOS down"
    elif last["sweep_low"]:
        return "BUY SETUP 👀", "Liquidity grabbed below"
    elif last["sweep_high"]:
        return "SELL SETUP 👀", "Liquidity grabbed above"
    elif trend == "Bullish" and last["bos_up"]:
        return "BULLISH CONTINUATION ✅", "Trend + structure continuation"
    elif trend == "Bearish" and last["bos_down"]:
        return "BEARISH CONTINUATION ✅", "Trend + structure continuation"

    return "WAIT ⚠️", "No clean smart-money setup"


def get_liquidity_status(df):
    last = df.iloc[-1]

    if last["sweep_low"]:
        return "Sell-side liquidity taken"
    elif last["sweep_high"]:
        return "Buy-side liquidity taken"
    elif pd.notna(last["liq_high"]) and pd.notna(last["liq_low"]):
        dist_high = abs(last["close"] - last["liq_high"])
        dist_low = abs(last["close"] - last["liq_low"])
        if dist_high < dist_low:
            return "Near buy-side liquidity"
        else:
            return "Near sell-side liquidity"
    return "—"


def generate_ai_suggestion(price, indicators, atr, signal_type, smart_signal):
    bullish_words = ["BUY", "BULLISH"]
    bearish_words = ["SELL", "BEARISH"]

    if any(word in smart_signal for word in bullish_words):
        signal_type = "Bullish"
    elif any(word in smart_signal for word in bearish_words):
        signal_type = "Bearish"
    elif signal_type not in ["Bullish", "Bearish"]:
        signal_type = "Neutral"

    if signal_type == "Bullish":
        sl = price - (atr * 1.2)
        tp = price + (atr * 2.5)
    elif signal_type == "Bearish":
        sl = price + (atr * 1.2)
        tp = price - (atr * 2.5)
    else:
        sl = price
        tp = price

    score = len(indicators)
    if "STRONG" in smart_signal:
        score += 2
    elif "SETUP" in smart_signal or "CONTINUATION" in smart_signal:
        score += 1

    if score >= 5:
        conf = "Strong"
    elif score >= 3:
        conf = "Medium"
    else:
        conf = "Weak"

    color = "green" if signal_type == "Bullish" else "red" if signal_type == "Bearish" else "gray"
    signal_txt = f"{conf} <span style='color:{color}'>{signal_type}</span> Signal @ {price:.5f}"

    if signal_type == "Neutral":
        return f"{signal_txt} | Wait for confirmation | Confidence: {conf}"

    return f"{signal_txt} | SL: {sl:.5f} | TP: {tp:.5f} | Confidence: {conf}"


news_events = fetch_forex_factory_news()
dxy_price, dxy_change = fetch_dxy_data()
rows = []

for label, symbol in symbols.items():
    try:
        url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval=5min&outputsize=200&apikey={API_KEY}"
        r = requests.get(url, timeout=20).json()

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
        df = add_liquidity_features(df, lookback=20)
        df = add_bos_features(df, swing_lookback=5)
        df.dropna(inplace=True)

        if df.empty:
            continue

        price = df["close"].iloc[-1]
        atr = df["ATR"].iloc[-1]
        rsi_val = df["RSI"].iloc[-1]

        trend = (
            "Bullish" if df["EMA9"].iloc[-1] > df["EMA20"].iloc[-1] and price > df["EMA9"].iloc[-1]
            else "Bearish" if df["EMA9"].iloc[-1] < df["EMA20"].iloc[-1] and price < df["EMA9"].iloc[-1]
            else "Sideways"
        )

        indicators = []
        signal_type = ""

        if rsi_val > 50:
            indicators.append("Bullish RSI")
            signal_type = "Bullish"
        elif rsi_val < 50:
            indicators.append("Bearish RSI")
            signal_type = "Bearish"

        if df["MACD"].iloc[-1] > df["MACD_Signal"].iloc[-1]:
            indicators.append("Bullish MACD")
        else:
            indicators.append("Bearish MACD")

        if df["EMA9"].iloc[-1] > df["EMA20"].iloc[-1] and price > df["EMA9"].iloc[-1]:
            indicators.append("Bullish EMA")
        elif df["EMA9"].iloc[-1] < df["EMA20"].iloc[-1] and price < df["EMA9"].iloc[-1]:
            indicators.append("Bearish EMA")

        if df["ADX"].iloc[-1] > 20:
            indicators.append("ADX Strong")

        divergence = detect_divergence(df)
        if divergence:
            indicators.append("Divergence")
            play_rsi_alert()

        smart_signal, smart_reason = get_smart_money_signal(df, trend)
        liquidity_status = get_liquidity_status(df)

        suggestion = generate_ai_suggestion(price, indicators, atr, signal_type, smart_signal)

        rows.append({
            "Pair": label,
            "Price": round(price, 5),
            "RSI": round(rsi_val, 2),
            "ATR Status": "🔴 Low" if atr < 0.0004 else "🟡 Normal" if atr < 0.0009 else "🟢 High",
            "Trend": trend,
            "Liquidity": liquidity_status,
            "Sweep High": "Yes" if df["sweep_high"].iloc[-1] else "No",
            "Sweep Low": "Yes" if df["sweep_low"].iloc[-1] else "No",
            "BOS Up": "Yes" if df["bos_up"].iloc[-1] else "No",
            "BOS Down": "Yes" if df["bos_down"].iloc[-1] else "No",
            "Smart Money Signal": smart_signal,
            "Smart Reason": smart_reason,
            "Reversal Signal": detect_trend_reversal(df),
            "Signal Type": signal_type if signal_type else "Neutral",
            "Confirmed Indicators": ", ".join(indicators),
            "AI Suggestion": suggestion,
            "DXY Impact": f"{dxy_price:.2f} ({dxy_change:+.2f}%)" if "USD" in label and dxy_price is not None else "—",
            "Divergence": divergence or "—",
            "Upcoming News & Impact": "\n".join(get_upcoming_news_with_impact(label))
        })

    except Exception as e:
        st.warning(f"{label} data fetch error: {e}")

column_order = [
    "Pair", "Price", "RSI", "ATR Status", "Trend", "Liquidity",
    "Sweep High", "Sweep Low", "BOS Up", "BOS Down",
    "Smart Money Signal", "Smart Reason",
    "Reversal Signal", "Signal Type", "Confirmed Indicators",
    "AI Suggestion", "DXY Impact", "Divergence", "Upcoming News & Impact"
]

df_result = pd.DataFrame(rows)

if df_result.empty:
    st.error("No valid market data found.")
else:
    # --- HTML table formatting
    styled_html = "<table style='width:100%; border-collapse: collapse;'>"
    styled_html += "<tr>" + "".join([
        f"<th style='border:1px solid #ccc; padding:6px; background:#e0e0e0'>{col}</th>"
        for col in column_order
    ]) + "</tr>"

    for _, row in df_result.iterrows():
        style = (
            'background-color: #d4edda;' if "Strong" in str(row["AI Suggestion"])
            else 'background-color: #cce5ff;' if "Medium" in str(row["AI Suggestion"])
            else 'background-color: #f8f9fa;'
        )

        styled_html += f"<tr style='{style}'>"

        for col in column_order:
            val = row[col]

            if col == "Pair":
                val = f"<strong style='font-size: 18px;'>{val}</strong>"

            elif col == "Trend":
                color = 'green' if row['Trend'] == 'Bullish' else 'red' if row['Trend'] == 'Bearish' else 'gray'
                val = f"<span style='color:{color}; font-weight:bold;'>{row['Trend']}</span>"

            elif col == "Signal Type":
                color = 'green' if row['Signal Type'] == 'Bullish' else 'red' if row['Signal Type'] == 'Bearish' else 'gray'
                val = f"<span style='color:{color}; font-weight:bold;'>{row['Signal Type']}</span>"

            elif col == "RSI":
                color = "red" if row["RSI"] > 75 else "green" if row["RSI"] < 20 else "black"
                val = f"<span style='color:{color}; font-weight:bold;'>{row['RSI']}</span>"

            elif col == "DXY Impact" and row["DXY Impact"] != "—":
                dxy_color = "green" if '+' in str(row["DXY Impact"]) else "red"
                val = f"<span style='color:{dxy_color}; font-weight:bold;'>{row['DXY Impact']}</span>"

            elif col == "Divergence" and row["Divergence"] != "—":
                div_color = "green" if "Bullish" in str(row["Divergence"]) else "red"
                val = f"<span style='color:{div_color}; font-weight:bold;'>{row['Divergence']}</span>"

            elif col == "Smart Money Signal":
                if "BUY" in str(row["Smart Money Signal"]) or "BULLISH" in str(row["Smart Money Signal"]):
                    val = f"<span style='color:green; font-weight:bold;'>{row['Smart Money Signal']}</span>"
                elif "SELL" in str(row["Smart Money Signal"]) or "BEARISH" in str(row["Smart Money Signal"]):
                    val = f"<span style='color:red; font-weight:bold;'>{row['Smart Money Signal']}</span>"
                else:
                    val = f"<span style='color:gray; font-weight:bold;'>{row['Smart Money Signal']}</span>"

            elif col == "Liquidity" and row["Liquidity"] != "—":
                liq = str(row["Liquidity"]).lower()
                liq_color = "green" if "sell-side" in liq else "red" if "buy-side" in liq else "black"
                val = f"<span style='color:{liq_color}; font-weight:bold;'>{row['Liquidity']}</span>"

            elif col in ["Sweep High", "Sweep Low", "BOS Up", "BOS Down"]:
                tf_color = "green" if row[col] == "Yes" else "gray"
                val = f"<span style='color:{tf_color}; font-weight:bold;'>{row[col]}</span>"

            styled_html += f"<td style='border:1px solid #ccc; padding:6px; white-space:pre-wrap;'>{val}</td>"

        styled_html += "</tr>"

    styled_html += "</table>"
    st.markdown(styled_html, unsafe_allow_html=True)

    st.caption(f"Timeframe: 5-Min | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.text(f"Scanned Pairs: {len(rows)}")
    st.text(f"Strong Signals Found: {len([r for r in rows if 'Strong' in r['AI Suggestion']])}")
