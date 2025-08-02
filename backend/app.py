import os
import pandas as pd
import requests
from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import io
import base64
import re
from datetime import datetime, timedelta
import yfinance as yf
import pandas_datareader as pdr
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import random
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

# API Keys - Ücretsiz planlar için
FINNHUB_API_KEY = "d25p1d9r01qhge4dmh5gd25p1d9r01qhge4dmh60"

# Finnhub API anahtarları
FINNHUB_API_KEYS = [
    "d25p1d9r01qhge4dmh5gd25p1d9r01qhge4dmh60",  # 🔑 API ANAHTARINIZ
    "d25p181r01qhge4dmgbgd25p181r01qhge4dmgc0",  # 🔑 YEDEK API ANAHTARINIZ
    "d25o23pr01qhge4di1egd25o23pr01qhge4di1f0",  # Yedek 1
    "d253na9r01qns40d15hgd253na9r01qns40d15i0"   # Yedek 2
]

# Alpha Vantage API Key (Ücretsiz: 500 istek/gün)
ALPHA_VANTAGE_API_KEY = "YOUR_ALPHA_VANTAGE_API_KEY"  # 🔑 ALPHA VANTAGE API ANAHTARINIZI BURAYA YAZIN

# Genişletilmiş BIST hisseleri listesi (BIST30 + BIST50 + popüler hisseler)
BIST_STOCKS = [
    # BIST30
    'AKBNK', 'ARCLK', 'ASELS', 'BIMAS', 'EKGYO', 'ENKAI', 'EUPWR', 'FROTO', 
    'GARAN', 'GUBRF', 'HEKTS', 'ISCTR', 'KCHOL', 'KRDMD', 'KOZAA', 'KOZAL', 
    'MGROS', 'PGSUS', 'SAHOL', 'SASA', 'SISE', 'TCELL', 'THYAO', 'TKFEN', 
    'TOASO', 'TUPRS', 'VAKBN', 'YKBNK', 'CCOLA', 'EREGL', 'SODA', 'PETKM',
    
    # BIST50 ek hisseler
    'AHLAT', 'AKSA', 'ALARK', 'ALBRK', 'ALGYO', 'ANACM', 'BASCM', 'BERA', 
    'BRISA', 'BRYAT', 'CEMAS', 'CEMTS', 'CIMSA', 'DOHOL', 'EGEEN', 'ENJSA', 
    'FMIZP', 'GESAN', 'GLYHO', 'HALKB', 'HATEK', 'INDES', 'IPEKE', 'KAREL', 
    'KARSN', 'KERVN', 'KERVT', 'KONTR', 'KONYA', 'LOGO', 'MACKO', 'NETAS', 
    'NTHOL', 'ODAS', 'OTKAR', 'OYAKC', 'PENTA', 'POLHO', 'PRKAB', 'PRKME', 
    'QUAGR', 'SAFKN', 'SELEC', 'SELGD', 'SMRTG', 'SNGYO', 'SOKM', 'TAVHL', 
    'TKNSA', 'TRGYO', 'TSKB', 'TTKOM', 'TTRAK', 'ULKER', 'VESBE', 'VESTL', 
    'YATAS', 'YUNSA', 'ZRGYO'
]

load_dotenv("api.env")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# LLM Model - Ücretsiz yerel model
llm_model = None
llm_tokenizer = None

def initialize_llm():
    """LLM modelini başlat"""
    global llm_model, llm_tokenizer
    try:
        print("LLM modeli yükleniyor...")
        # Küçük ve hızlı bir model kullan
        model_name = "microsoft/DialoGPT-small"
        llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
        llm_model = AutoModelForCausalLM.from_pretrained(model_name)
        print("✅ LLM modeli başarıyla yüklendi!")
        return True
    except Exception as e:
        print(f"❌ LLM modeli yüklenemedi: {e}")
        return False

def ask_llm(question, max_length=100):
    """LLM'e soru sor"""
    global llm_model, llm_tokenizer
    try:
        if llm_model is None or llm_tokenizer is None:
            return "LLM modeli henüz yüklenmedi."
        
        # Soruyu tokenize et
        inputs = llm_tokenizer.encode(question, return_tensors="pt")
        
        # Yanıt üret
        with torch.no_grad():
            outputs = llm_model.generate(
                inputs, 
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=llm_tokenizer.eos_token_id
            )
        
        # Yanıtı decode et
        response = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    except Exception as e:
        print(f"LLM hatası: {e}")
        return "LLM yanıt veremedi."

# LLM'i başlat
initialize_llm()

def get_finnhub_quote(symbol):
    """Gerçek zamanlı hisse verisi al - Finnhub, Alpha Vantage ve Yahoo Finance ile"""
    
    # Önce Finnhub'ı dene
    for api_key in FINNHUB_API_KEYS:
        try:
            # BIST hisseleri için doğru format: BIST:CCOLA
            finnhub_symbol = f"BIST:{symbol}"
            url = f"https://finnhub.io/api/v1/quote?symbol={finnhub_symbol}&token={api_key}"
            response = requests.get(url, timeout=10)
            
            print(f"Finnhub API call for {finnhub_symbol} with key {api_key[:10]}...: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                if data.get('c', 0) > 0:
                    print(f"✅ Finnhub data for {symbol}: {data}")
                    return data
            elif response.status_code == 429:  # Rate limit
                print(f"Rate limit for key {api_key[:10]}..., trying next key")
                continue
            else:
                print(f"Finnhub API error for {symbol}: {response.status_code}")
                continue
        except Exception as e:
            print(f"Finnhub API exception for {symbol} with key {api_key[:10]}...: {e}")
            continue
    
    # Finnhub başarısız olursa Yahoo Finance'i dene
    print(f"Finnhub failed for {symbol}, trying Yahoo Finance...")
    yf_data = get_yfinance_quote(symbol)
    if yf_data and yf_data.get('c', 0) > 0:
        print(f"✅ Yahoo Finance data for {symbol}: {yf_data}")
        return yf_data
    
    # Yahoo Finance başarısız olursa Alpha Vantage'ı dene
    print(f"Yahoo Finance failed for {symbol}, trying Alpha Vantage...")
    alpha_data = get_alpha_vantage_quote(symbol)
    if alpha_data and alpha_data.get('c', 0) > 0:
        print(f"✅ Alpha Vantage data for {symbol}: {alpha_data}")
        return alpha_data
    
    print(f"All APIs failed for {symbol}")
    return None

def get_yfinance_quote(symbol):
    """Yahoo Finance'den gerçek zamanlı hisse verisi al"""
    try:
        # BIST hisseleri için format: THYAO.IS
        yf_symbol = f"{symbol}.IS"
        ticker = yf.Ticker(yf_symbol)
        info = ticker.info
        
        # Gerçek zamanlı veri al
        hist = ticker.history(period="1d")
        if not hist.empty:
            current_price = hist['Close'].iloc[-1]
            open_price = hist['Open'].iloc[0]
            high_price = hist['High'].max()
            low_price = hist['Low'].min()
            volume = hist['Volume'].sum()
            
            change = current_price - open_price
            change_percent = (change / open_price) * 100 if open_price > 0 else 0
            
            return {
                'c': current_price,  # Current price
                'd': change,  # Change
                'dp': change_percent,  # Change percent
                'h': high_price,  # High
                'l': low_price,  # Low
                'v': volume  # Volume
            }
        return None
    except Exception as e:
        print(f"Yahoo Finance API exception for {symbol}: {e}")
        return None

def get_yfinance_chart(symbol, days=30):
    """Yahoo Finance'den hisse grafiği verisi al"""
    try:
        yf_symbol = f"{symbol}.IS"
        ticker = yf.Ticker(yf_symbol)
        
        # Son N günün verilerini al
        hist = ticker.history(period=f"{days}d")
        
        if not hist.empty:
            dates = [d.timestamp() for d in hist.index]
            prices = hist['Close'].tolist()
            
            return {
                's': 'ok',
                't': dates,
                'c': prices
            }
        return None
    except Exception as e:
        print(f"Yahoo Finance chart API exception for {symbol}: {e}")
        return None

def get_forecast_prophet(symbol, days=30):
    """Prophet ile hisse fiyat tahmini"""
    try:
        yf_symbol = f"{symbol}.IS"
        ticker = yf.Ticker(yf_symbol)
        
        # Son 1 yıllık veri al
        hist = ticker.history(period="1y")
        
        print(f"Prophet için {symbol} verisi: {len(hist)} satır")
        
        if hist.empty:
            print(f"{symbol} için veri boş!")
            return None
        
        if len(hist) < 30:  # En az 30 gün veri gerekli
            print(f"{symbol} için yeterli veri yok: {len(hist)} satır")
            return None
        
        # Prophet için veri hazırla
        df = hist.reset_index()
        df = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
        
        # Timezone'u kaldır (Prophet timezone desteklemiyor)
        df['ds'] = df['ds'].dt.tz_localize(None)
        
        # NaN değerleri temizle
        df = df.dropna()
        
        print(f"Prophet modeli eğitiliyor... Veri şekli: {df.shape}")
        print(f"Veri örneği: {df.head()}")
        
        # Prophet modeli oluştur (daha basit ayarlar)
        model = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=0.01
        )
        model.fit(df)
        
        # Gelecek günler için tahmin
        future = model.make_future_dataframe(periods=days)
        forecast = model.predict(future)
        
        # Son N günün tahminini al
        predictions = forecast.tail(days)
        
        print(f"Tahmin tamamlandı! {len(predictions)} gün tahmin")
        print(f"İlk tahmin: {predictions['yhat'].iloc[0]:.2f}")
        
        return {
            'dates': [d.timestamp() for d in predictions['ds']],
            'predictions': predictions['yhat'].tolist(),
            'lower': predictions['yhat_lower'].tolist(),
            'upper': predictions['yhat_upper'].tolist()
        }
    except Exception as e:
        print(f"Prophet forecast exception for {symbol}: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_forecast_arima(symbol, days=30):
    """ARIMA ile hisse fiyat tahmini"""
    try:
        yf_symbol = f"{symbol}.IS"
        ticker = yf.Ticker(yf_symbol)
        
        # Son 1 yıllık veri al
        hist = ticker.history(period="1y")
        
        if hist.empty:
            return None
        
        # Fiyat verilerini al
        prices = hist['Close'].values
        
        # ARIMA modeli (p=1, d=1, q=1)
        model = ARIMA(prices, order=(1, 1, 1))
        model_fit = model.fit()
        
        # Gelecek günler için tahmin
        forecast = model_fit.forecast(steps=days)
        
        # Tarihleri oluştur
        last_date = hist.index[-1]
        dates = [last_date + pd.Timedelta(days=i+1) for i in range(days)]
        
        return {
            'dates': [d.timestamp() for d in dates],
            'predictions': forecast.tolist()
        }
    except Exception as e:
        print(f"ARIMA forecast exception for {symbol}: {e}")
        return None

def get_forecast_lstm(symbol, days=30):
    """LSTM ile hisse fiyat tahmini"""
    try:
        yf_symbol = f"{symbol}.IS"
        ticker = yf.Ticker(yf_symbol)
        
        # Son 1 yıllık veri al
        hist = ticker.history(period="1y")
        
        if hist.empty:
            return None
        
        # Veriyi normalize et
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(hist['Close'].values.reshape(-1, 1))
        
        # LSTM için veri hazırla (son 60 gün)
        X, y = [], []
        for i in range(60, len(scaled_data)):
            X.append(scaled_data[i-60:i, 0])
            y.append(scaled_data[i, 0])
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        # LSTM modeli
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(60, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Modeli eğit (hızlı eğitim için epochs=10)
        model.fit(X, y, epochs=10, batch_size=32, verbose=0)
        
        # Gelecek günler için tahmin
        last_60_days = scaled_data[-60:]
        predictions = []
        
        for _ in range(days):
            X_test = last_60_days.reshape(1, 60, 1)
            pred = model.predict(X_test, verbose=0)
            predictions.append(pred[0, 0])
            last_60_days = np.append(last_60_days[1:], pred[0, 0])
        
        # Tahminleri denormalize et
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        
        # Tarihleri oluştur
        last_date = hist.index[-1]
        dates = [last_date + pd.Timedelta(days=i+1) for i in range(days)]
        
        return {
            'dates': [d.timestamp() for d in dates],
            'predictions': predictions.flatten().tolist()
        }
    except Exception as e:
        print(f"LSTM forecast exception for {symbol}: {e}")
        return None

def get_available_stocks(language="tr"):
    """Mevcut hisseler listesini döndür"""
    if language == "tr":
        stocks_list = "\n".join([f"• {stock}" for stock in BIST_STOCKS])
        return f"📋 MEVCUT HİSSELER (Finnhub API ile gerçek zamanlı veri):\n\n{stocks_list}\n\n💡 Örnek kullanım:\n• 'CCOLA güncel fiyat'\n• 'THYAO grafik'\n• 'GARAN haber'"
    else:
        stocks_list = "\n".join([f"• {stock}" for stock in BIST_STOCKS])
        return f"📋 AVAILABLE STOCKS (Real-time data via Finnhub API):\n\n{stocks_list}\n\n💡 Example usage:\n• 'CCOLA current price'\n• 'THYAO chart'\n• 'GARAN news'"

@app.post("/ask")
async def ask_question(question: str = Form(...), language: str = Form("tr")):
    print(f"Received request - question: '{question}', language: '{language}'")
    try:
        question_lower = question.lower()
        print(f"Processing question: '{question_lower}'")
        
        # "Hangi şirketler mevcut" sorusu
        if any(word in question_lower for word in ['hangi şirket', 'mevcut', 'available', 'companies', 'stocks']):
            print("Detected 'available stocks' question")
            result = {"answer": get_available_stocks(language), "chart": None}
            print(f"Returning: {result}")
            return result
        
        # Hisse kodu var mı? (ör: CCOLA, BIMAS, THYAO)
        hisse = None
        for code in BIST_STOCKS:
            if code.lower() in question_lower:
                hisse = code
                break
        
        # Forecasting sorusu mu? (ör: CCOLA tahmin, CCOLA forecast, CCOLA gelecek)
        if hisse and any(word in question_lower for word in ['tahmin', 'forecast', 'gelecek', 'future', 'prediction']):
            print(f"Getting forecast for {hisse}")
            try:
                # Hangi forecasting yöntemi kullanılacak?
                forecast_method = 'prophet'  # Varsayılan
                if 'arima' in question_lower:
                    forecast_method = 'arima'
                elif 'lstm' in question_lower or 'neural' in question_lower:
                    forecast_method = 'lstm'
                elif 'prophet' in question_lower:
                    forecast_method = 'prophet'
                
                # Forecasting yap
                if forecast_method == 'prophet':
                    forecast_data = get_forecast_prophet(hisse, days=30)
                elif forecast_method == 'arima':
                    forecast_data = get_forecast_arima(hisse, days=30)
                elif forecast_method == 'lstm':
                    forecast_data = get_forecast_lstm(hisse, days=30)
                else:
                    forecast_data = get_forecast_prophet(hisse, days=30)
                
                if forecast_data:
                    # Grafik çiz
                    dates = [datetime.fromtimestamp(ts) for ts in forecast_data['dates']]
                    predictions = forecast_data['predictions']
                    
                    plt.figure(figsize=(12, 6))
                    plt.plot(dates, predictions, linewidth=2, color='green', label='Tahmin')
                    
                    # Prophet için güven aralığı
                    if 'lower' in forecast_data and 'upper' in forecast_data:
                        plt.fill_between(dates, forecast_data['lower'], forecast_data['upper'], 
                                       alpha=0.3, color='green', label='Güven Aralığı')
                    
                    plt.title(f"{hisse} 30 Günlük Fiyat Tahmini ({forecast_method.upper()})", 
                             fontsize=14, fontweight='bold')
                    plt.xlabel('Tarih', fontsize=12)
                    plt.ylabel('Tahmin Fiyatı (TL)', fontsize=12)
                    plt.grid(True, alpha=0.3)
                    plt.legend()
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                    plt.close()
                    buf.seek(0)
                    chart_b64 = base64.b64encode(buf.read()).decode('utf-8')
                    
                    # Son 5 tahmin değeri
                    last_5_predictions = predictions[-5:]
                    last_5_dates = [d.strftime('%d.%m') for d in dates[-5:]]
                    
                    if language == 'tr':
                        answer = f"🔮 {hisse} 30 GÜNLÜK TAHMİN ({forecast_method.upper()}):\n\n"
                        answer += "📊 Son 5 gün tahmini:\n"
                        for i, (date, pred) in enumerate(zip(last_5_dates, last_5_predictions), 1):
                            answer += f"   {date}: {pred:.2f} TL\n"
                        answer += f"\n💡 Tahmin yöntemi: {forecast_method.upper()}\n"
                        answer += "⚠️ Bu tahminler sadece referans amaçlıdır!"
                    else:
                        answer = f"🔮 {hisse} 30-DAY FORECAST ({forecast_method.upper()}):\n\n"
                        answer += "📊 Last 5 days prediction:\n"
                        for i, (date, pred) in enumerate(zip(last_5_dates, last_5_predictions), 1):
                            answer += f"   {date}: {pred:.2f} TL\n"
                        answer += f"\n💡 Forecast method: {forecast_method.upper()}\n"
                        answer += "⚠️ These predictions are for reference only!"
                    
                    return {"answer": answer, "chart": chart_b64}
                else:
                    if language == 'tr':
                        answer = f"❌ {hisse} için tahmin yapılamadı. Yeterli veri yok."
                    else:
                        answer = f"❌ Could not forecast {hisse}. Insufficient data."
                    return {"answer": answer, "chart": None}
            except Exception as e:
                print(f"Error processing forecast request: {e}")
                if language == 'tr':
                    return {"answer": f"❌ {hisse} için tahmin oluşturulamadı.", "chart": None}
                else:
                    return {"answer": f"❌ Could not create forecast for {hisse}.", "chart": None}
        
        # Grafik sorusu mu? (ör: CCOLA grafik, CCOLA chart)
        if hisse and any(word in question_lower for word in ['grafik', 'chart', 'görsel']):
            print(f"Getting chart for {hisse}")
            try:
                # Zaman aralığını belirle
                days = 30  # Varsayılan
                if '1 ay' in question_lower or '1ay' in question_lower:
                    days = 30
                elif '3 ay' in question_lower or '3ay' in question_lower:
                    days = 90
                elif '6 ay' in question_lower or '6ay' in question_lower:
                    days = 180
                elif '1 yıl' in question_lower or '1yıl' in question_lower or '1 yil' in question_lower:
                    days = 365
                
                # Yahoo Finance'den grafik verisi al
                chart_data = get_yfinance_chart(hisse, days=days)
                if chart_data and chart_data.get('s') == 'ok':
                    dates = [datetime.fromtimestamp(ts) for ts in chart_data['t']]
                    prices = chart_data['c']
                    
                    # Grafik çiz
                    plt.figure(figsize=(12, 6))
                    plt.plot(dates, prices, linewidth=2, color='blue', label='Fiyat')
                    
                    plt.title(f"{hisse} Son {days} Günlük Fiyat Grafiği", 
                             fontsize=14, fontweight='bold')
                    plt.xlabel('Tarih', fontsize=12)
                    plt.ylabel('Fiyat (TL)', fontsize=12)
                    plt.grid(True, alpha=0.3)
                    plt.legend()
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                    plt.close()
                    buf.seek(0)
                    chart_b64 = base64.b64encode(buf.read()).decode('utf-8')
                    
                    # Son 5 fiyat değeri
                    last_5_prices = prices[-5:]
                    last_5_dates = [d.strftime('%d.%m') for d in dates[-5:]]
                    
                    if language == 'tr':
                        answer = f"📊 {hisse} SON {days} GÜNLÜK FİYAT GRAFİĞİ:\n\n"
                        answer += "💰 Son 5 gün fiyatı:\n"
                        for i, (date, price) in enumerate(zip(last_5_dates, last_5_prices), 1):
                            answer += f"   {date}: {price:.2f} TL\n"
                        answer += f"\n📈 Grafik: Son {days} günlük fiyat hareketi"
                    else:
                        answer = f"📊 {hisse} LAST {days} DAYS PRICE CHART:\n\n"
                        answer += "💰 Last 5 days price:\n"
                        for i, (date, price) in enumerate(zip(last_5_dates, last_5_prices), 1):
                            answer += f"   {date}: {price:.2f} TL\n"
                        answer += f"\n📈 Chart: Last {days} days price movement"
                    
                    return {"answer": answer, "chart": chart_b64}
                else:
                    if language == 'tr':
                        answer = f"❌ {hisse} için grafik verisi bulunamadı."
                    else:
                        answer = f"❌ Chart data not found for {hisse}."
                    return {"answer": answer, "chart": None}
            except Exception as e:
                print(f"Error processing chart request: {e}")
                if language == 'tr':
                    return {"answer": f"❌ {hisse} için grafik oluşturulamadı.", "chart": None}
                else:
                    return {"answer": f"❌ Could not create chart for {hisse}.", "chart": None}
        
        # Yatırım tavsiyesi sorusu mu? (ör: 1000 TL ne alayım, portföy önerisi)
        if any(word in question_lower for word in ['tavsiye', 'öneri', 'ne alayım', 'portföy', 'yatırım', 'advice', 'recommendation', 'portfolio']):
            print("Getting investment advice")
            try:
                # Miktar belirleme (regex ile)
                amount = 1000  # Varsayılan
                match = re.search(r'(\d{3,6})\s*tl', question_lower)
                if match:
                    amount = int(match.group(1))
                else:
                    # 'bin', '5bin', '10bin' gibi ifadeleri de yakala
                    if '10bin' in question_lower or '10 bin' in question_lower:
                        amount = 10000
                    elif '5bin' in question_lower or '5 bin' in question_lower:
                        amount = 5000
                    elif 'bin' in question_lower:
                        amount = 1000
                
                # Risk profili belirleme
                risk_profile = 'orta'  # Varsayılan
                if any(word in question_lower for word in ['düşük', 'güvenli', 'low', 'safe']):
                    risk_profile = 'düşük'
                elif any(word in question_lower for word in ['yüksek', 'agresif', 'high', 'aggressive']):
                    risk_profile = 'yüksek'
                
                # Hisse önerileri - Dinamik seçim
                
                # Farklı sektörlerden hisseler
                bank_stocks = [('GARAN', 'Garanti Bankası'), ('AKBNK', 'Akbank'), ('ISCTR', 'İş Bankası'), ('YKBNK', 'Yapı Kredi')]
                tech_stocks = [('ARCLK', 'Arçelik'), ('BIMAS', 'BİM'), ('MGROS', 'Migros'), ('SAHOL', 'Sabancı Holding')]
                energy_stocks = [('TUPRS', 'Tüpraş'), ('AKSA', 'Aksa'), ('ENJSA', 'Enerjisa'), ('EGEEN', 'Ege Enerji')]
                consumer_stocks = [('CCOLA', 'Coca Cola'), ('ULKER', 'Ülker'), ('SASA', 'Sasa'), ('PETKM', 'Petkim')]
                industrial_stocks = [('EREGL', 'Ereğli Demir Çelik'), ('KRDMD', 'Kardemir'), ('CIMSA', 'Çimsa'), ('ASELS', 'Aselsan')]
                transport_stocks = [('THYAO', 'Türk Hava Yolları'), ('PGSUS', 'Pegasus'), ('DOAS', 'Doğuş Otomotiv'), ('FROTO', 'Ford Otosan')]
                
                # Miktara göre portföy büyüklüğü
                if amount <= 2000:
                    portfolio_size = 3  # Küçük portföy
                elif amount <= 5000:
                    portfolio_size = 4  # Orta portföy
                else:
                    portfolio_size = 5  # Büyük portföy
                
                # Risk profiline göre sektör seçimi
                if risk_profile == 'düşük':
                    # Güvenli sektörler: Banka, Tüketici, Enerji
                    sectors = [bank_stocks, consumer_stocks, energy_stocks]
                    sector_weights = [0.4, 0.35, 0.25]
                elif risk_profile == 'yüksek':
                    # Yüksek risk sektörler: Teknoloji, Endüstri, Ulaşım
                    sectors = [tech_stocks, industrial_stocks, transport_stocks]
                    sector_weights = [0.4, 0.35, 0.25]
                else:  # orta risk
                    # Dengeli sektörler
                    sectors = [bank_stocks, tech_stocks, consumer_stocks, energy_stocks]
                    sector_weights = [0.3, 0.3, 0.2, 0.2]
                
                # Her sektörden rastgele hisse seç
                recommendations = []
                used_sectors = set()
                
                for i in range(portfolio_size):
                    # Henüz kullanılmamış sektörlerden seç
                    available_sectors = [s for j, s in enumerate(sectors) if j not in used_sectors]
                    if not available_sectors:
                        # Tüm sektörler kullanıldıysa tekrar kullan
                        available_sectors = sectors
                        used_sectors.clear()
                    
                    # Rastgele sektör seç
                    sector = random.choice(available_sectors)
                    used_sectors.add(sectors.index(sector))
                    
                    # O sektörden rastgele hisse seç
                    stock = random.choice(sector)
                    
                    # Ağırlık hesapla
                    if i < len(sector_weights):
                        weight = sector_weights[i]
                    else:
                        weight = 1.0 / portfolio_size
                    
                    recommendations.append((stock[0], stock[1], weight))
                
                # Ağırlıkları normalize et
                total_weight = sum(rec[2] for rec in recommendations)
                recommendations = [(rec[0], rec[1], rec[2] / total_weight) for rec in recommendations]
                
                # LLM ile kişiselleştirilmiş tavsiye al
                llm_advice = ""
                try:
                    llm_prompt = f"Ben {amount} TL ile yatırım yapmak istiyorum. Risk profili: {risk_profile}. Önerilen hisseler: {[rec[0] for rec in recommendations]}. Bu portföy hakkında kısa bir yorum yap."
                    llm_advice = ask_llm(llm_prompt, max_length=150)
                    if llm_advice and llm_advice != "LLM modeli henüz yüklenmedi." and llm_advice != "LLM yanıt veremedi.":
                        llm_advice = f"\n🤖 LLM Analizi: {llm_advice}"
                except:
                    llm_advice = ""
                
                # Önerileri formatla
                if language == 'tr':
                    answer = f"💼 {amount:,} TL İÇİN YATIRIM TAVSİYESİ:\n\n"
                    answer += f"📊 Risk Profili: {risk_profile.upper()}\n\n"
                    answer += "🎯 Önerilen Portföy:\n"
                    
                    for symbol, name, ratio in recommendations:
                        investment = amount * ratio
                        shares = int(investment / 100)  # Yaklaşık hisse sayısı
                        answer += f"   • {symbol} ({name}): {investment:,.0f} TL ({shares} hisse)\n"
                    
                    answer += f"\n💰 Toplam Yatırım: {amount:,} TL{llm_advice}\n"
                    answer += "⚠️ Bu tavsiyeler sadece referans amaçlıdır!\n"
                    answer += "🕐 Borsa saati: 10:00-18:00"
                else:
                    answer = f"💼 INVESTMENT ADVICE FOR {amount:,} TL:\n\n"
                    answer += f"📊 Risk Profile: {risk_profile.upper()}\n\n"
                    answer += "🎯 Recommended Portfolio:\n"
                    
                    for symbol, name, ratio in recommendations:
                        investment = amount * ratio
                        shares = int(investment / 100)
                        answer += f"   • {symbol} ({name}): {investment:,.0f} TL ({shares} shares)\n"
                    
                    answer += f"\n💰 Total Investment: {amount:,} TL{llm_advice}\n"
                    answer += "⚠️ These recommendations are for reference only!\n"
                    answer += "🕐 Market hours: 10:00-18:00"
                
                return {"answer": answer, "chart": None}
            except Exception as e:
                print(f"Error processing investment advice: {e}")
                if language == 'tr':
                    return {"answer": "❌ Yatırım tavsiyesi oluşturulamadı.", "chart": None}
                else:
                    return {"answer": "❌ Could not create investment advice.", "chart": None}
        
        # Güncel fiyat sorusu
        if hisse and any(word in question_lower for word in ['güncel', 'current', 'fiyat', 'price', 'son', 'last', 'anlık']):
            print(f"Getting current price for {hisse}")
            
            # Yahoo Finance'den gerçek zamanlı veri al
            yf_data = get_yfinance_quote(hisse)
            if yf_data and yf_data.get('c', 0) > 0:
                current_price = yf_data['c']
                change = yf_data.get('d', 0)
                change_percent = yf_data.get('dp', 0)
                high = yf_data.get('h', 0)
                low = yf_data.get('l', 0)
                volume = yf_data.get('v', 0)
                
                if language == 'tr':
                    answer = f"🎯 {hisse} GÜNCEL FİYAT BİLGİLERİ:\n\n"
                    answer += f"💰 Anlık Fiyat: {current_price:.2f} TL\n"
                    answer += f"📈 Günlük Değişim: {change:+.2f} TL ({change_percent:+.2f}%)\n"
                    answer += f"📊 Günlük Yüksek: {high:.2f} TL\n"
                    answer += f"📉 Günlük Düşük: {low:.2f} TL\n"
                    answer += f"📈 İşlem Hacmi: {volume:,} adet\n\n"
                    answer += "🕐 *Gerçek zamanlı veri (Yahoo Finance)*"
                else:
                    answer = f"🎯 {hisse} CURRENT PRICE INFO:\n\n"
                    answer += f"💰 Current Price: {current_price:.2f} TL\n"
                    answer += f"📈 Daily Change: {change:+.2f} TL ({change_percent:+.2f}%)\n"
                    answer += f"📊 Daily High: {high:.2f} TL\n"
                    answer += f"📉 Daily Low: {low:.2f} TL\n"
                    answer += f"📈 Volume: {volume:,} shares\n\n"
                    answer += "🕐 *Real-time data (Yahoo Finance)*"
                
                return {"answer": answer, "chart": None}
            else:
                if language == 'tr':
                    answer = f"❌ {hisse} için gerçek zamanlı veri bulunamadı."
                else:
                    answer = f"❌ Real-time data not found for {hisse}."
                return {"answer": answer, "chart": None}
        
        # Diğer sorular için yardım
        if language == 'tr':
            answer = "🤖 FINBOT Size Nasıl Yardımcı Olabilir?\n\n"
            answer += "📈 Güncel fiyat: 'ARCLK güncel fiyat'\n"
            answer += "📊 Grafik: 'ARCLK grafik' veya 'ARCLK 3 ay grafik'\n"
            answer += "🔮 Tahmin: 'ARCLK tahmin' veya 'ARCLK forecast'\n"
            answer += "💼 Yatırım tavsiyesi: '1000 TL ne alayım' veya 'portföy önerisi'\n"
            answer += "📋 Hisse listesi: 'Hangi şirketler mevcut'\n\n"
            answer += "💡 Grafik süreleri: 1 ay, 3 ay, 6 ay, 1 yıl\n"
            answer += "💡 Tahmin yöntemleri: Prophet, ARIMA, LSTM\n"
            answer += "💡 Risk profilleri: Düşük, Orta, Yüksek"
        else:
            answer = "🤖 How can FINBOT help you?\n\n"
            answer += "📈 Current price: 'ARCLK current price'\n"
            answer += "📊 Chart: 'ARCLK chart' or 'ARCLK 3 months chart'\n"
            answer += "🔮 Forecast: 'ARCLK forecast' or 'ARCLK prediction'\n"
            answer += "💼 Investment advice: 'What should I buy with 1000 TL' or 'portfolio recommendation'\n"
            answer += "📋 Stock list: 'Which companies are available'\n\n"
            answer += "💡 Chart periods: 1 month, 3 months, 6 months, 1 year\n"
            answer += "💡 Forecast methods: Prophet, ARIMA, LSTM\n"
            answer += "💡 Risk profiles: Low, Medium, High"
        
        return {"answer": answer, "chart": None}
        
    except Exception as e:
        print(f"Error in ask_question: {e}")
        if language == 'tr':
            return {"error": f"❌ Bir hata oluştu: {str(e)}"}
        else:
            return {"error": f"❌ An error occurred: {str(e)}"}

@app.get("/")
def root():
    return {"message": "FINBOT backend is running with forecasting capabilities."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 