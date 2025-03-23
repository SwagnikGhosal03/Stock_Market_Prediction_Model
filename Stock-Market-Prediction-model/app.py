import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import load_model  # type: ignore
import streamlit as st
import matplotlib.pyplot as plt
import requests
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import datetime
import google.generativeai as genai
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon', quiet=True)

model = load_model('Stock Predictions Model.keras')

st.title('📈 Stock Market Predictor')

stock = st.text_input('Enter Stock Symbol:', 'GOOG').upper()
start = '2012-01-01'
end=datetime.today().strftime('%Y-%m-%d')

st.subheader('📊 Stock Data')
try:
    data = yf.download(stock, start, end)
    if data.empty:
        st.warning(f"No data found for '{stock}'. Please check the stock symbol.")
        st.stop()
    st.write(data)
except Exception as e:
    st.error(f"Error fetching stock data: {e}")
    st.stop()

def compute_moving_averages(data):
    return {
        "50 Days": data['Close'].rolling(50).mean(),
        "100 Days": data['Close'].rolling(100).mean(),
        "200 Days": data['Close'].rolling(200).mean()
    }

ma_values = compute_moving_averages(data)


#news API
st.subheader("📰 Stock News & Sentiment Analysis")

NEWS_API_KEY = "23a76dd2d6844e4a8d60b79e6ec2e3ca"

def fetch_stock_news(stock):
    NEWS_API_URL = f'https://newsapi.org/v2/everything?q={stock}&language=en&sortBy=publishedAt&apiKey=23a76dd2d6844e4a8d60b79e6ec2e3ca'
    response = requests.get(NEWS_API_URL)
    if response.status_code == 200:
        return response.json().get("articles", [])
    return []

#sentiment score
def analyze_sentiment(news_articles):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = []
    for article in news_articles:
        score = analyzer.polarity_scores(article["title"])["compound"]
        sentiment_scores.append(score)
    return sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0

news_articles = fetch_stock_news(stock)
sentiment_score = analyze_sentiment(news_articles)

st.subheader("Latest News")
if news_articles:
    for article in news_articles[:7]: 
        st.markdown(f"**[{article['title']}]({article['url']})**")
else:
    st.warning("No news found.")

st.subheader("Sentiment Analysis")
st.write(f"Sentiment Score: {sentiment_score:.2f}")

if sentiment_score > 0.05:
    st.success("Positive Sentiment 😊")
elif sentiment_score < -0.05:
    st.error("Negative Sentiment 😠")
else:
    st.warning("Neutral Sentiment 😐")

def compute_macd(data):
    short_ema = data['Close'].ewm(span=12, adjust=False).mean()
    long_ema = data['Close'].ewm(span=26, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

data['MACD'], data['Signal Line'] = compute_macd(data)

def compute_rsi(data, window=14):
    delta = data['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

data['RSI'] = compute_rsi(data)

# Chat-bot
genai.configure(api_key="AIzaSyDo_kSyVQkErJsgToh-CrFmpS8vT-pGeYg")
def get_stock_price(ticker):
    return str(yf.Ticker(ticker).history(period='1y').iloc[-1].Close)

def calculate_sma(ticker,window):
    data=yf.Ticker(ticker).history(period='1y').iloc[-1].Close
    return str(data.rolling(window=window).mean().iloc[-1])

def calculate_ema(ticker,window):
    data=yf.Ticker(ticker).history(period='1y').iloc[-1].Close
    return str(data.ewm(span=window,adjust=False).mean().iloc[-1])

def calculate_rsi(ticker, window=14):
    data = yf.Ticker(ticker).history(period='1y')['Close']  
    delta = data.diff()  
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ema_up = up.ewm(com=window - 1, adjust=False).mean()
    ema_down = down.ewm(com=window - 1, adjust=False).mean()

    rs = ema_up / ema_down
    rsi = 100 - (100 / (1 + rs))
    return str(rsi.iloc[-1])  


def calculate_macd(ticker):
    data = yf.Ticker(ticker).history(period='1y')['Close']  
    short_ema = data.ewm(span=12, adjust=False).mean()  
    long_ema = data.ewm(span=26, adjust=False).mean()  
    macd = short_ema - long_ema 
    signal = macd.ewm(span=9, adjust=False).mean()  
    macd_histogram = macd - signal 
    return f'{round(macd.iloc[-1], 2)}, {round(signal.iloc[-1], 2)}, {round(macd_histogram.iloc[-1], 2)}'

def plot_stock_price(ticker):
    data=yf.Ticker(ticker).history(period='1y').iloc[-1]
    plt.figure(figsize=(10,5))
    plt.plot(data.index,data.Close)
    plt.title(f'{ticker} Stock Price Over Last Year')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.grid(True)
    plt.savefig('stock.png')
    plt.close()

    
def predict_stock_price(stock_symbol, model):
        try:
            end_date = pd.to_datetime('today').strftime('%Y-%m-%d')
            start_date = (pd.to_datetime('today') - pd.DateOffset(days=60)).strftime('%Y-%m-%d')
            data = yf.download(stock_symbol, start=start_date, end=end_date)
            if data.empty:
                return "No sufficient data to make a prediction."
            recent_data = np.array(data['Close']).reshape(1, -1, 1) 
            predicted_price = model.predict(recent_data)
            return f"Predicted Price for {stock_symbol}: ${predicted_price[0][0]:.2f}"
        except Exception as e:
            return f"Prediction Error: {e}"
company_to_ticker = {
    "APPLE": "AAPL",
    "TESLA": "TSLA",
    "GOOGLE": "GOOG",
    "ALPHABET": "GOOG",
    "AMAZON": "AMZN",
    "MICROSOFT": "MSFT",
    "META": "META",
    "FACEBOOK": "META",
    "NVIDIA": "NVDA",
    "NETFLIX": "NFLX",
    "AMD": "AMD",
    "INTEL": "INTC",
    "IBM": "IBM",
    "ADOBE": "ADBE",
    "ORACLE": "ORCL",
    "CISCO": "CSCO",
    "SALESFORCE": "CRM",

    # Financial Sector
    "JPMORGAN": "JPM",
    "VISA": "V",
    "MASTERCARD": "MA",
    "BANK OF AMERICA": "BAC",
    "GOLDMAN SACHS": "GS",
    "AMERICAN EXPRESS": "AXP",
    "WELLS FARGO": "WFC",

    #  Automotive Sector
    "FORD": "F",
    "GENERAL MOTORS": "GM",
    "TOYOTA": "TM",
    "FERRARI": "RACE",
    "HONDA": "HMC",

    #Energy Sector
    "EXXON MOBIL": "XOM",
    "CHEVRON": "CVX",
    "BP": "BP",
    "SHELL": "SHEL",
    "TOTALENERGIES": "TTE",
    "CONOCOPHILLIPS": "COP",

    #Healthcare & Pharma
    "PFIZER": "PFE",
    "MODERNA": "MRNA",
    "JOHNSON & JOHNSON": "JNJ",
    "ASTRAZENECA": "AZN",
    "GILEAD": "GILD",
    "MERCK": "MRK",
    "ABBVIE": "ABBV",
    "NOVARTIS": "NVS",

    # Consumer & Retail
    "WALMART": "WMT",
    "TARGET": "TGT",
    "COSTCO": "COST",
    "PROCTER & GAMBLE": "PG",
    "COCA COLA": "KO",
    "PEPSICO": "PEP",
    "MCDONALDS": "MCD",
    "STARBUCKS": "SBUX",
    "DISNEY": "DIS",
    "NIKE": "NKE",
    "ADIDAS": "ADDYY",

    #  Travel & Hospitality
    "DELTA AIR LINES": "DAL",
    "UNITED AIRLINES": "UAL",
    "SOUTHWEST AIRLINES": "LUV",
    "HILTON": "HLT",
    "MARRIOTT": "MAR",
    "BOOKING HOLDINGS": "BKNG",

    # Industrials & Manufacturing
    "CATERPILLAR": "CAT",
    "BOEING": "BA",
    "LOCKHEED MARTIN": "LMT",
    "GENERAL ELECTRIC": "GE",
    "3M": "MMM",
    "HONEYWELL": "HON",

    #  Telecom & Media
    "VERIZON": "VZ",
    "AT&T": "T",
    "T-MOBILE": "TMUS",
    "COMCAST": "CMCSA",

    #  Materials & Commodities
    "FREEPORT-MCMORAN": "FCX",
    "NEWMONT CORPORATION": "NEM",
    "BARRICK GOLD": "GOLD",

    #  Real Estate & REITs
    "REALTY INCOME": "O",
    "PROLOGIS": "PLD",
    "SIMON PROPERTY GROUP": "SPG",

    #  Utilities
    "DUKE ENERGY": "DUK",
    "EXELON": "EXC",
    "SOUTHERN COMPANY": "SO",

    #  Crypto & Digital Payments
    "COINBASE": "COIN",
    "PAYPAL": "PYPL",
    "BLOCK": "SQ",
}

known_tickers = set(company_to_ticker.values())  
def extract_ticker(user_input):
    words = user_input.upper().split()  
    for word in words:
        if word in known_tickers:  
            return word
        elif word in company_to_ticker: 
            return company_to_ticker[word] 
    return stock  

functions=[
        {
            'name':'get_stock_price',
            'description':'Gets the latest stock price given the ticker symbol of a company',
            'parameters':{
                'type':'object',
                'properties':{
                    'ticker':{
                        'type':'string',
                        'description':'The stock ticker symbol for a company (for example AAPL for Apple). Note:FB is renamed to META'
                    }
                },
                'required':['ticker']
            },
        },
        {
            'name':'calculate_sma',
            'description':'Calculates the simple moving average for a given stock ticker and a window',
            'parameters':{
                'type':'object',
                'properties':{
                    'ticker':{
                        'type':'string',
                        'description':'The stock ticker symbol for a company(e.g., AAPL for Apple)',
                    },
                    'window':{
                       'type':'integer',
                       'description':'The timeframe to consider when calculating the sma',
                    }
                },
                'required':['ticker','window']
            }
        },
        {
            'name':'calculate_ema',
            'description':'calculate the exponential moving average for a given stock ticker and a window',
            'parameters':{
                'ticker':{
                    'type':'string',
                    'description':'The stock ticker symbol for a company(e.g., AAPL for Apple)',
                },
                'window':{
                    'type':'integer',
                     'description':'The timeframe to consider when calculating the ema',
                },
            },
            'required':['ticker','window'],
        },
        {
            'name':'calculate_rsi',
            'description':'calculate the rsi for a given stock ticker',
            'parameters':{
                'type':'object',
                'properties':{
                    'ticker':{
                        'type':'string',
                        'description':'The stock ticker symbol for a company(e.g., AAPL for Apple)',
                    },
                },
                'required':['ticker'],
            },
        },
         {
            'name':'calculate_macd',
            'description':'calculate the macd for a given stock ticker',
            'parameters':{
                'type':'object',
                'properties':{
                    'ticker':{
                        'type':'string',
                        'description':'The stock ticker symbol for a company(e.g., AAPL for Apple)',
                    },
                },
                'required':['ticker'],
            },
        },
        {
            'name':'plot_stock_price',
            'description':'plot the stock price for the last year given the ticker symbol of a company',
            'parameters':{
                'type':'object',
                'properties':{
                    'ticker':{
                        'type':'string',
                        'description':'The stock ticker symbol for a company(e.g., AAPL for Apple)',
                    },
                },
                'required':['ticker'],
            },
        },
        {
    'name': 'predict_stock_price',
    'description': 'Predicts the next day stock price using a trained model.',
    'parameters': {
        'type': 'object',
        'properties': {
            'stock_symbol': {
                'type': 'string',
                'description': 'The stock ticker symbol (e.g., AAPL for Apple).'
            }
        },
        'required': ['stock_symbol']
    }
},
    ]
available_functions={
        'predict_stock_price':predict_stock_price,
        'get_stock_price': get_stock_price,
        'calculate_sma':calculate_sma,
        'calculate_ema':calculate_ema,
        'calculate_rsi':calculate_rsi,
        'calculate_macd':calculate_macd,
        'plot_stock_price':plot_stock_price,
    }
if 'messages' not in st.session_state:
    st.session_state['messages'] = [] 

st.title('Stock Analysis Chatbot Assistant')
user_input=st.text_input('Your Input:')

if user_input:
    try:
        st.session_state['messages'].append({'role': 'user', 'content': user_input})

        stock_ticker = extract_ticker(user_input)
        bot_response = ""
        if any(keyword in user_input.upper() for keyword in ["RSI", "RELATIVE STRENGTH INDEX"]):
            bot_response = f"The current RSI of {stock_ticker} is: **{calculate_rsi(stock_ticker)}**"

        elif "MACD" in user_input.upper():
            bot_response = f"The MACD values for {stock_ticker} are: **{calculate_macd(stock_ticker)}**"

        elif "SMA" in user_input.upper():
            bot_response = f"The 50-day SMA of {stock_ticker} is: **{calculate_sma(stock_ticker, 50)}**"

        elif "EMA" in user_input.upper():
            bot_response = f"The 50-day EMA of {stock_ticker} is: **{calculate_ema(stock_ticker, 50)}**"

        elif any(keyword in user_input.upper() for keyword in ["stock price", "current price"]):
            bot_response = f"The latest stock price of {stock_ticker} is: **${get_stock_price(stock_ticker)}**"

        elif any(keyword in user_input.upper() for keyword in ["predict","future price", "forecast"]):
            bot_response = predict_stock_price(stock_ticker, model)

        else:
            model = genai.GenerativeModel("gemini-2.0-pro-exp")
            response = model.generate_content([msg['content'] for msg in st.session_state['messages']])
            bot_response = response.text.strip()

        st.session_state['messages'].append({'role': 'assistant', 'content': bot_response})
        st.write(bot_response)

    except Exception as e:
        st.error(f"Error: {e}")

st.subheader('📉 Stock Price with Moving Averages')
fig1, ax1 = plt.subplots(figsize=(8, 6))
ax1.plot(data['Close'], label='Stock Price', color='g')
for label, ma in ma_values.items():
    ax1.plot(ma, label=label)
ax1.legend()
st.pyplot(fig1)

st.subheader('📊 Relative Strength Index (RSI)')
fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2.plot(data['RSI'], label='RSI', color='orange')
ax2.axhline(70, linestyle='--', color='r', label='Overbought (70)')
ax2.axhline(30, linestyle='--', color='g', label='Oversold (30)')
ax2.legend()
st.pyplot(fig2)

st.subheader('📈 MACD Indicator')
fig3, ax3 = plt.subplots(figsize=(8, 6))
ax3.plot(data['MACD'], label='MACD', color='blue')
ax3.plot(data['Signal Line'], label='Signal Line', color='red')
ax3.legend()
st.pyplot(fig3)