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
st_autorefresh(interval=300000, key="ai_refresh")  # 5 min

API_KEY = "b2a1234a9ea240f9ba85696e2a243403"

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

# --- DXY fetch
def fetch_dxy_data():
    try:
        dxy = yf.Ticker("DX-Y.NYB")
        data = dxy.history(period="1d", interval="1m")
        current = data["Close"].iloc[-1]
        previous = data["Close"].iloc[0]
        change = current - previous
        percent = (change / previous) * 100
        return current, percent
    except:
        return 100.237, -0.16

# --- Forex Factory news fetch
def fetch_forex_factory_news():
    url = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"
    response = requests.get(url)
    try:
        root = ET.fromstring(response.content)
    except:
        return []

    news_data = []
    for item in root.findall("./channel/item"):
        try:
            title = item.find("title").text
            pub_time = date_parser.parse(item.find("pubDate").text)
            currency_tag = item.find("{http://www.forexfactory.com/rss}currency")
            currency = currency_tag.text.strip().upper() if currency_tag is not None else ""
            if pub_time.date() >= datetime.utcnow().date():  # today or upcoming
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

# --- Get news for each pair
def get_news_for_pair(pair):
    base, quote = pair.split('/')
    news_list = []
    for n in news_events:
        if quote in n["currency"]:  # match USD/XAU/XAG
            impact = analyze_impact(n["title"])
            time_str = n["time"].strftime("%H:%M")
            news_list.append(f"{n['title']} ({impact}) @ {time_str}")
    return news_list or ["—"]

# --- Technical indicators
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta>0, 0)
    loss = -delta.where(delta<0, 0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1+rs))

def calculate_macd(series):
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

def calculate_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calculate_atr(df, period=14):
    tr = pd.concat([df['high']-df['low'], 
                    abs(df['high']-df['close'].shift()), 
                    abs(df['low']-df['close'].shift())], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def calculate_adx(df, period=14):
    df['TR'] = np.maximum(df['high']-df['low'], 
                          np.maximum(abs(df['high']-df['close'].shift()), abs(df['low']-df['close'].shift())))
    df['+DM'] = np.where((df['high']-df['high'].shift()) > (df['low'].shift()-df['low']), np.maximum(df['high']-df['high'].shift(),0),0)
    df['-DM'] = np.where((df['low'].shift()-df['low']) > (df['high']-df['high'].shift()), np.maximum(df['low'].shift()-df['low'],0),0)
    tr14 = df['TR'].rolling(period).mean()
    plus_dm14 = df['+DM'].rolling(period).mean()
    minus_dm14 = df['-DM'].rolling(period).mean()
    plus_di14 = 100*(plus_dm14/tr14)
    minus_di14 = 100*(minus_dm14/tr14)
    dx = 100*abs(plus_di14-minus_di14)/(plus_di14+minus_di14)
    return dx.rolling(period).mean()

# --- Trend & divergence
def detect_trend_reversal(df):
    e9, e20 = df['EMA9'].iloc[-3:], df['EMA20'].iloc[-3:]
    if e9[0]<e20[0] and e9[1]>e20[1] and e9[2]>e20[2]: return "Reversal Confirmed Bullish"
    if e9[0]>e20[0] and e9[1]<e20[1] and e9[2]<e20[2]: return "Reversal Confirmed Bearish"
    if e9[-2]<e20[-2] and e9[-1]>e20[-1]: return "Reversal Forming Bullish"
    if e9[-2]>e20[-2] and e9[-1]<e20[-1]: return "Reversal Forming Bearish"
    return ""

def detect_divergence(df):
    closes = df['close']
    rsis = df['RSI']
    if len(closes)<10: return ""
    low_idx = closes.iloc[-5:].idxmin()
    high_idx = closes.iloc[-5:].idxmax()
    rsi_low_idx = rsis.iloc[-5:].idxmin()
    rsi_high_idx = rsis.iloc[-5:].idxmax()
    if low_idx != rsi_low_idx and closes[low_idx]<closes[-1] and rsis[rsi_low_idx]>rsis[-1]:
        return "Bullish Divergence"
    if high_idx != rsi_high_idx and closes[high_idx]>closes[-1] and rsis[rsi_high_idx]<rsis[-1]:
        return "Bearish Divergence"
    return ""

def generate_ai_suggestion(price, indicators, atr, signal_type):
    if not indicators: return ""
    sl = price-(atr*1.2) if signal_type=="Bullish" else price+(atr*1.2)
    tp = price+(atr*2.5) if signal_type=="Bullish" else price-(atr*2.5)
    count = len(indicators)
    if count>=4: conf="Strong"
    elif count==3: conf="Medium"
    else: return ""
    color="green" if signal_type=="Bullish" else "red"
    return f"{conf} <span style='color:{color}'>{signal_type}</span> Signal @ {price:.5f} | SL: {sl:.5f} | TP: {tp:.5f} | Confidence: {conf}"

# --- Main
news_events = fetch_forex_factory_news()
dxy_price, dxy_change = fetch_dxy_data()
rows = []

for label, symbol in symbols.items():
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval=5min&outputsize=200&apikey={API_KEY}"
    r = requests.get(url).json()
    if "values" not in r: continue

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
    trend = "Bullish" if df["EMA9"].iloc[-1]>df["EMA20"].iloc[-1] and price>df["EMA9"].iloc[-1] else \
            "Bearish" if df["EMA9"].iloc[-1]<df["EMA20"].iloc[-1] and price<df["EMA9"].iloc[-1] else "Sideways"

    rsi_val = df["RSI"].iloc[-1]
    indicators = []
    signal_type = ""
    if rsi_val>50: indicators.append("Bullish"); signal_type="Bullish"
    elif rsi_val<50: indicators.append("Bearish"); signal_type="Bearish"
    if df["MACD"].iloc[-1]>df["MACD_Signal"].iloc[-1]: indicators.append("MACD")
    if df["EMA9"].iloc[-1]>df["EMA20"].iloc[-1] and price>df["EMA9"].iloc[-1]: indicators.append("EMA")
    if df["ADX"].iloc[-1]>20: indicators.append("ADX")

    divergence = detect_divergence(df)
    if divergence: indicators.append("Divergence"); play_rsi_alert()
    suggestion = generate_ai_suggestion(price, indicators, atr, signal_type)

    rows.append({
        "Pair": label,
        "Price": round(price,5),
        "RSI": round(rsi_val,2),
        "ATR Status": "🔴 Low" if atr<0.0004 else "🟡 Normal" if atr<0.0009 else "🟢 High",
        "Trend": trend,
        "Reversal Signal": detect_trend_reversal(df),
        "Signal Type": signal_type,
        "Confirmed Indicators": ", ".join(indicators),
        "AI Suggestion": suggestion,
        "DXY Impact": f"{dxy_price:.2f} ({dxy_change:+.2f}%)" if "USD" in label and dxy_price is not None else "—",
        "Divergence": divergence or "—",
        "Upcoming News & Impact": "\n".join(get_news_for_pair(label))  # ✅ working now
    })

# --- Columns & Styling
column_order = ["Pair","Price","RSI","ATR Status","Trend","Reversal Signal","Signal Type",
                "Confirmed Indicators","AI Suggestion","DXY Impact","Divergence","Upcoming News & Impact"]

df_result = pd.DataFrame(rows)
df_result["Score"] = df_result["AI Suggestion"].apply(lambda x: 3 if "Strong" in x else 2 if "Medium" in x else 0)
df_sorted = df_result.sort_values(by="Score", ascending=False).drop(columns=["Score"])

styled_html = "<table style='width:100%; border-collapse: collapse;'>"
styled_html += "<tr>" + "".join([f"<th style='border:1px solid #ccc; padding:6px; background:#e0e0e0'>{c}</th>" for c in column_order])+"</tr>"

for _, row in df_sorted.iterrows():
    style = 'background-color: #d4edda;' if "Strong" in str(row["AI Suggestion"]) else 'background-color: #d1ecf1;' if "Medium" in str(row["AI Suggestion"]) else ''
    styled_html += f"<tr style='{style}'>"
    for col in column_order:
        val = row[col]
        if col=="Pair": val = f"<strong style='font-size:18px'>{val}</strong>"
        elif col=="Trend": val = f"<span style='color:{'green' if row['Trend']=='Bullish' else 'red' if row['Trend']=='Bearish' else 'gray'}; font-weight:bold'>{val}</span>"
        elif col=="Signal Type": val = f"<span style='color:{'green' if row['Signal Type']=='Bullish' else 'red'}; font-weight:bold'>{val}</span>"
        elif col=="RSI": val = f"<span style='color:{'red' if row['RSI']>75 else 'green' if row['RSI']<20 else 'black'}; font-weight:bold'>{val}</span>"
        elif col=="DXY Impact" and val!="—": val = f"<span style='color:{'green' if '+' in val else 'red'}; font-weight:bold'>{val}</span>"
        elif col=="Divergence" and val!="—": val = f"<span style='color:{'green' if 'Bullish' in val else 'red'}; font-weight:bold'>{val}</span>"
        styled_html += f"<td style='border:1px solid #ccc; padding:6px; white-space:pre-wrap'>{val}</td>"
    styled_html += "</tr>"
styled_html += "</table>"

st.markdown(styled_html, unsafe_allow_html=True)
st.caption(f"Timeframe: 5-Min | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.text(f"Scanned Pairs: {len(rows)}")
st.text(f"Strong Signals Found: {len([r for r in rows if r['AI Suggestion'] and 'Strong' in r['AI Suggestion']])}")
