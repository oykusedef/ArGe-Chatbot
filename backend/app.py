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
from newsapi import NewsApiClient
import json
from typing import List, Dict, Tuple
import http.client
import urllib.parse

# API Keys - Ücretsiz planlar için
FINNHUB_API_KEY = "d25p1d9r01qhge4dmh5gd25p1d9r01qhge4dmh60"

# Finnhub API anahtarları
FINNHUB_API_KEYS = [
    "d25p1d9r01qhge4dmh5gd25p1d9r01qhge4dmh60",  # 🔑 API ANAHTARINIZ
    "d25p181r01qhge4dmgbgd25p181r01qhge4dmgc0",  # 🔑 YEDEK API ANAHTARINIZ
    "d25o23pr01qhge4di1egd25o23pr01qhge4di1f0",  # Yedek 1
    "d253na9r01qns40d15hgd253na9r01qns40d15i0"   # Yedek 2
]


# News API Key (Ücretsiz: 100 istek/gün)
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "4VPDfIzFhUWoRyzmwtWso4rREi9fsIw18CSOcsx9")  # 🔑 NEWS API ANAHTARINIZI BURAYA YAZIN

# Türkçe sentiment analizi için anahtar kelimeler
TURKISH_POSITIVE_WORDS = [
    # Finansal pozitif kelimeler
    'artış', 'yükseliş', 'kazanç', 'kâr', 'olumlu', 'iyi', 'güzel', 'başarılı',
    'büyüme', 'gelişme', 'ilerleme', 'yükselme', 'artma', 'çıkış', 'yükseliş',
    'olumlu', 'mükemmel', 'harika', 'süper', 'güçlü', 'sağlam', 'stabil',
    'güvenilir', 'kaliteli', 'profesyonel', 'yenilikçi', 'modern', 'teknolojik',
    'rekabetçi', 'dinamik', 'esnek', 'sürdürülebilir', 'verimli', 'etkili',
    'stratejik', 'vizyoner', 'lider', 'pazar lideri', 'sektör lideri',
    'yüksek performans', 'güçlü büyüme', 'olumlu trend', 'iyi sonuç',
    'başarılı proje', 'yeni ürün', 'inovasyon', 'teknoloji', 'dijitalleşme',
    'sürdürülebilir büyüme', 'finansal güç', 'nakit akışı', 'temettü',
    'yatırım', 'genişleme', 'pazar payı', 'müşteri memnuniyeti', 'kalite',
    'sertifika', 'ödül', 'başarı', 'hedef', 'plan', 'strateji', 'vizyon'
]

TURKISH_NEGATIVE_WORDS = [
    # Finansal negatif kelimeler
    'düşüş', 'kayıp', 'zarar', 'olumsuz', 'kötü', 'kriz', 'problem', 'sorun',
    'düşme', 'azalma', 'kaybetme', 'başarısızlık', 'başarısız', 'zayıf', 'kırılgan',
    'riskli', 'belirsiz', 'kararsız', 'durgun', 'yavaş', 'zayıf', 'kötüleşme',
    'düşüş', 'çöküş', 'iflas', 'borç', 'kayıp', 'zarar', 'olumsuz', 'negatif',
    'düşük performans', 'zayıf büyüme', 'olumsuz trend', 'kötü sonuç',
    'başarısız proje', 'güvenlik açığı', 'veri sızıntısı', 'hack', 'siber saldırı',
    'rekabet baskısı', 'pazar kaybı', 'müşteri kaybı', 'şikayet', 'dava',
    'ceza', 'yaptırım', 'denetim', 'uyarı', 'kınama', 'soruşturma', 'araştırma',
    'şüphe', 'güvensizlik', 'belirsizlik', 'risk', 'tehlike', 'tehdit', 'korku',
    'endişe', 'kaygı', 'stres', 'baskı', 'zorluk', 'engel', 'obstacle', 'barrier'
]

# Duygu kategorileri için kelimeler
EMOTION_CATEGORIES = {
    'güven': ['güven', 'güvenilir', 'güvenli', 'sağlam', 'stabil', 'sürdürülebilir', 'kaliteli'],
    'korku': ['korku', 'endişe', 'kaygı', 'tehlike', 'risk', 'tehdit', 'belirsizlik'],
    'umut': ['umut', 'gelecek', 'potansiyel', 'fırsat', 'vizyon', 'hedef', 'plan'],
    'hayal kırıklığı': ['hayal kırıklığı', 'düş kırıklığı', 'başarısız', 'kötü', 'olumsuz'],
    'coşku': ['coşku', 'heyecan', 'harika', 'mükemmel', 'süper', 'inanılmaz'],
    'öfke': ['öfke', 'kızgın', 'sinir', 'şikayet', 'dava', 'ceza', 'yaptırım']
}

def analyze_turkish_sentiment_detailed(text: str) -> Dict:
    """Türkçe metin için detaylı sentiment analizi"""
    text_lower = text.lower()
    
    # Pozitif ve negatif kelime sayısı
    positive_count = sum(1 for word in TURKISH_POSITIVE_WORDS if word in text_lower)
    negative_count = sum(1 for word in TURKISH_NEGATIVE_WORDS if word in text_lower)
    
    # Toplam kelime sayısı
    total_words = len(text.split())
    
    if total_words == 0:
        return {
            'sentiment_score': 0.0,
            'sentiment_label': 'Nötr',
            'confidence': 0.0,
            'emotions': {},
            'key_phrases': [],
            'word_count': 0
        }
    
    # Sentiment skoru (-1 ile 1 arası)
    sentiment_score = (positive_count - negative_count) / total_words
    sentiment_score = max(-1.0, min(1.0, sentiment_score))
    
    # Güven skoru (kelime sayısına göre)
    confidence = min(1.0, (positive_count + negative_count) / max(1, total_words * 0.1))
    
    # Duygu analizi
    emotions = {}
    for emotion, words in EMOTION_CATEGORIES.items():
        emotion_count = sum(1 for word in words if word in text_lower)
        if emotion_count > 0:
            emotions[emotion] = emotion_count
    
    # Anahtar kelimeleri çıkar
    key_phrases = []
    for word in TURKISH_POSITIVE_WORDS + TURKISH_NEGATIVE_WORDS:
        if word in text_lower:
            key_phrases.append(word)
    
    # Sentiment etiketi
    if sentiment_score > 0.1:
        sentiment_label = 'Olumlu'
    elif sentiment_score < -0.1:
        sentiment_label = 'Olumsuz'
    else:
        sentiment_label = 'Nötr'
    
    return {
        'sentiment_score': sentiment_score,
        'sentiment_label': sentiment_label,
        'confidence': confidence,
        'emotions': emotions,
        'key_phrases': key_phrases[:5],  # İlk 5 anahtar kelime
        'word_count': total_words,
        'positive_words': positive_count,
        'negative_words': negative_count
    }

def analyze_turkish_sentiment(text: str) -> float:
    """Türkçe metin için sentiment analizi (geriye uyumluluk için)"""
    result = analyze_turkish_sentiment_detailed(text)
    return result['sentiment_score']

def get_news_sentiment(company_name: str, stock_code: str) -> Dict:
    """Şirket için haber sentiment analizi"""
    try:
        # Önce The News API'den Türkçe haberleri al
        turkish_news = get_turkish_stock_news_by_company(company_name, stock_code)
        
        all_news = []
        
        # The News API'den gelen haberleri işle
        if turkish_news['success'] and turkish_news['news']:
            for article in turkish_news['news']:
                # The News API formatını News API formatına çevir
                formatted_article = {
                    'title': article.get('title', ''),
                    'description': article.get('description', ''),
                    'source': {'name': article.get('source', 'Bilinmeyen')},
                    'publishedAt': article.get('published_at', ''),
                    'url': article.get('url', '')
                }
                all_news.append(formatted_article)
            print(f"Found {len(turkish_news['news'])} Turkish articles from The News API")
        
        # Eğer The News API'den yeterli haber yoksa, News API'yi de dene
        if len(all_news) < 5:
            newsapi = NewsApiClient(api_key=NEWS_API_KEY)
            
            # Daha geniş arama terimleri oluştur
            search_terms = [company_name, stock_code]
            
            # Şirket adına göre ek terimler ekle
            if stock_code == 'AKBNK':
                search_terms.extend(['Akbank', 'akbank', 'AKBANK', 'Türkiye bankacılık', 'banka', 'finans'])
            elif stock_code == 'GARAN':
                search_terms.extend(['Garanti', 'garanti', 'GARANTI', 'Türkiye bankacılık', 'banka', 'finans'])
            elif stock_code == 'ISCTR':
                search_terms.extend(['İş Bankası', 'İşbank', 'isbank', 'ISBANK', 'Türkiye bankacılık', 'banka', 'finans'])
            elif stock_code == 'YKBNK':
                search_terms.extend(['Yapı Kredi', 'Yapıkredi', 'yapikredi', 'YAPIKREDI', 'Türkiye bankacılık', 'banka', 'finans'])
            elif stock_code == 'VAKBN':
                search_terms.extend(['Vakıfbank', 'vakifbank', 'VAKIFBANK', 'Türkiye bankacılık', 'banka', 'finans'])
            elif stock_code == 'THYAO':
                search_terms.extend(['Türk Hava Yolları', 'THY', 'thy', 'havacılık', 'uçak', 'havayolu'])
            elif stock_code == 'TCELL':
                search_terms.extend(['Turkcell', 'turkcell', 'TURKCELL', 'telekomünikasyon', 'mobil', 'iletişim'])
            elif stock_code == 'TUPRS':
                search_terms.extend(['Tüpraş', 'tupras', 'TUPRAS', 'petrol', 'rafineri', 'enerji'])
            elif stock_code == 'ASELS':
                search_terms.extend(['Aselsan', 'aselsan', 'ASELSAN', 'savunma', 'elektronik', 'teknoloji'])
            elif stock_code == 'EREGL':
                search_terms.extend(['Ereğli', 'eregli', 'EREGLI', 'demir çelik', 'çelik', 'metal'])
            elif stock_code == 'KCHOL':
                search_terms.extend(['Koç Holding', 'Koç', 'koc', 'KOC', 'holding', 'sanayi'])
            elif stock_code == 'SAHOL':
                search_terms.extend(['Sabancı Holding', 'Sabancı', 'sabanci', 'SABANCI', 'holding', 'sanayi'])
            elif stock_code == 'FROTO':
                search_terms.extend(['Ford Otosan', 'Ford', 'ford', 'FORD', 'otomotiv', 'araç'])
            elif stock_code == 'TOASO':
                search_terms.extend(['Toyota Otosan', 'Toyota', 'toyota', 'TOYOTA', 'otomotiv', 'araç'])
            elif stock_code == 'BIMAS':
                search_terms.extend(['BİM', 'bim', 'BIM', 'market', 'perakende', 'gıda'])
            elif stock_code == 'MGROS':
                search_terms.extend(['Migros', 'migros', 'MIGROS', 'market', 'perakende', 'gıda'])
            elif stock_code == 'SASA':
                search_terms.extend(['Sasa Polyurethan', 'Sasa Polyurethan A.Ş.', 'SASA Polyurethan', 'Sasa kimya', 'Sasa plastik', 'Sasa polietilen'])
            elif stock_code == 'SISE':
                search_terms.extend(['Şişe Cam', 'Şişe', 'sise', 'SISE', 'cam', 'cam ürünleri'])
            elif stock_code == 'CCOLA':
                search_terms.extend(['Coca Cola', 'Coca-Cola', 'coca cola', 'COCA COLA', 'içecek', 'meşrubat'])
            elif stock_code == 'PGSUS':
                search_terms.extend(['Pegasus', 'pegasus', 'PEGASUS', 'havacılık', 'uçak', 'havayolu'])
            else:
                # Genel terimler ekle
                search_terms.extend([company_name.lower(), company_name.upper(), company_name.title()])
            
            for term in search_terms:
                try:
                    # Son 7 günün haberlerini al
                    news = newsapi.get_everything(
                        q=term,
                        language='tr',
                        sort_by='publishedAt',
                        from_param=(datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
                        page_size=20
                    )
                    
                    if news['status'] == 'ok' and news['articles']:
                        # Haberleri filtrele - başlıkta şirket adı geçenleri öncelikle al
                        filtered_articles = []
                        for article in news['articles']:
                            title = article.get('title', '').lower()
                            description = article.get('description', '').lower()
                            
                            # Şirket adı veya hisse kodu başlıkta geçiyorsa öncelikli
                            if (company_name.lower() in title or 
                                stock_code.lower() in title or
                                company_name.lower() in description or 
                                stock_code.lower() in description):
                                filtered_articles.append(article)
                        
                        if filtered_articles:
                            all_news.extend(filtered_articles)
                            print(f"Found {len(filtered_articles)} relevant articles for term: {term}")
                        else:
                            # Eğer filtrelenmiş haber yoksa, tüm haberleri al ama log'la
                            all_news.extend(news['articles'])
                            print(f"Found {len(news['articles'])} articles for term: {term} (no exact match)")
                except Exception as e:
                    print(f"News API error for {term}: {e}")
                    continue
        
        if not all_news:
            # Eğer Türkçe haber bulunamazsa İngilizce dene
            for term in search_terms[:5]:  # İlk 5 terimi dene
                try:
                    news = newsapi.get_everything(
                        q=term,
                        language='en',
                        sort_by='publishedAt',
                        from_param=(datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
                        page_size=20
                    )
                    
                    if news['status'] == 'ok' and news['articles']:
                        all_news.extend(news['articles'])
                        print(f"Found {len(news['articles'])} English articles for term: {term}")
                except Exception as e:
                    print(f"News API error for {term} (EN): {e}")
                    continue
        
        if not all_news:
            return {
                'sentiment_score': 0.0,
                'sentiment_label': 'Nötr',
                'confidence': 0.0,
                'news_count': 0,
                'positive_news': 0,
                'negative_news': 0,
                'neutral_news': 0,
                'top_emotions': [],
                'top_key_phrases': [],
                'recent_news': [],
                'error': 'Haber bulunamadı'
            }
        
        # Haberleri son kez filtrele - tamamen alakasız olanları çıkar
        filtered_all_news = []
        for article in all_news:
            title = article.get('title', '').lower()
            description = article.get('description', '').lower()
            
            # Şirket adı veya hisse kodu geçmiyorsa ve tamamen alakasızsa çıkar
            if (company_name.lower() not in title and 
                stock_code.lower() not in title and
                company_name.lower() not in description and 
                stock_code.lower() not in description):
                # Eğer NVIDIA, Apple, Microsoft gibi tamamen farklı şirketler geçiyorsa çıkar
                irrelevant_keywords = ['nvidia', 'apple', 'microsoft', 'google', 'amazon', 'tesla', 'meta', 'netflix']
                if any(keyword in title or keyword in description for keyword in irrelevant_keywords):
                    continue
            
            filtered_all_news.append(article)
        
        print(f"Filtered {len(all_news)} articles down to {len(filtered_all_news)} relevant articles")
        
        if not filtered_all_news:
            return {
                'sentiment_score': 0.0,
                'sentiment_label': 'Nötr',
                'confidence': 0.0,
                'news_count': 0,
                'positive_news': 0,
                'negative_news': 0,
                'neutral_news': 0,
                'top_emotions': [],
                'top_key_phrases': [],
                'recent_news': [],
                'error': 'İlgili haber bulunamadı'
            }
        
        # Detaylı sentiment analizi
        detailed_sentiments = []
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        recent_news = []
        all_emotions = {}
        all_key_phrases = []
        total_confidence = 0.0
        
        for article in filtered_all_news[:10]:  # Son 10 haberi analiz et
            title = article.get('title', '')
            description = article.get('description', '')
            content = f"{title} {description}"
            
            # Detaylı Türkçe sentiment analizi
            detailed_sentiment = analyze_turkish_sentiment_detailed(content)
            detailed_sentiments.append(detailed_sentiment)
            
            # Kategorize et
            if detailed_sentiment['sentiment_score'] > 0.1:
                positive_count += 1
            elif detailed_sentiment['sentiment_score'] < -0.1:
                negative_count += 1
            else:
                neutral_count += 1
            
            # Duyguları topla
            for emotion, count in detailed_sentiment['emotions'].items():
                all_emotions[emotion] = all_emotions.get(emotion, 0) + count
            
            # Anahtar kelimeleri topla
            all_key_phrases.extend(detailed_sentiment['key_phrases'])
            total_confidence += detailed_sentiment['confidence']
            
            # Son haberler listesi
            recent_news.append({
                'title': title,
                'source': article.get('source', {}).get('name', 'Bilinmeyen'),
                'published_at': article.get('publishedAt', ''),
                'sentiment': detailed_sentiment['sentiment_score'],
                'sentiment_label': detailed_sentiment['sentiment_label'],
                'confidence': detailed_sentiment['confidence'],
                'emotions': detailed_sentiment['emotions'],
                'key_phrases': detailed_sentiment['key_phrases'],
                'url': article.get('url', '')
            })
        
        # Ortalama sentiment skoru
        avg_sentiment = sum(s['sentiment_score'] for s in detailed_sentiments) / len(detailed_sentiments) if detailed_sentiments else 0.0
        avg_confidence = total_confidence / len(detailed_sentiments) if detailed_sentiments else 0.0
        
        # Sentiment etiketi
        if avg_sentiment > 0.1:
            sentiment_label = 'Olumlu'
        elif avg_sentiment < -0.1:
            sentiment_label = 'Olumsuz'
        else:
            sentiment_label = 'Nötr'
        
        # En yaygın duyguları sırala
        top_emotions = sorted(all_emotions.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # En yaygın anahtar kelimeleri sırala
        from collections import Counter
        key_phrase_counts = Counter(all_key_phrases)
        top_key_phrases = key_phrase_counts.most_common(5)
        
        return {
            'sentiment_score': avg_sentiment,
            'sentiment_label': sentiment_label,
            'confidence': avg_confidence,
            'news_count': len(filtered_all_news),
            'positive_news': positive_count,
            'negative_news': negative_count,
            'neutral_news': neutral_count,
            'top_emotions': top_emotions,
            'top_key_phrases': top_key_phrases,
            'recent_news': recent_news[:10],  # Son 10 haber
            'error': None
        }
        
    except Exception as e:
        print(f"Error in get_news_sentiment: {e}")
        return {
            'sentiment_score': 0.0,
            'sentiment_label': 'Nötr',
            'confidence': 0.0,
            'news_count': 0,
            'positive_news': 0,
            'negative_news': 0,
            'neutral_news': 0,
            'top_emotions': [],
            'top_key_phrases': [],
            'recent_news': [],
            'error': f'Haber analizi hatası: {str(e)}'
        }

def get_turkish_stock_news():
    """The News API'den Türkçe borsa haberlerini al"""
    try:
        conn = http.client.HTTPSConnection('api.thenewsapi.com')
        
        params = urllib.parse.urlencode({
            'api_token': '4VPDfIzFhUWoRyzmwtWso4rREi9fsIw18CSOcsx9',  # thenewsapi.com anahtarı
            'categories': 'business,tech',
            'limit': 3,  # Ücretsiz plan limiti
            'language': 'tr',        # Türkçe haberleri filtrelemek için
            'locale': 'tr',          # Türkiye kaynaklı haberler için
            'search': 'borsa'        # Türk hisse/borsa araması
        })
        
        conn.request('GET', '/v1/news/all?{}'.format(params))
        res = conn.getresponse()
        data = res.read()
        
        # JSON verisini parse et
        news_data = json.loads(data.decode('utf-8'))
        
        # API yanıtını kontrol et (status yerine data varlığını kontrol et)
        if news_data.get('data') and len(news_data.get('data', [])) > 0:
            return {
                'success': True,
                'news': news_data.get('data', []),
                'total': len(news_data.get('data', []))
            }
        else:
            return {
                'success': False,
                'error': 'Haber bulunamadı',
                'news': [],
                'total': 0
            }
            
    except Exception as e:
        print(f"Türk borsa haberleri alınırken hata: {e}")
        return {
            'success': False,
            'error': str(e),
            'news': [],
            'total': 0
        }

def get_turkish_stock_news_by_company(company_name: str, stock_code: str):
    """Belirli bir şirket için Türkçe borsa haberlerini al"""
    try:
        conn = http.client.HTTPSConnection('api.thenewsapi.com')
        
        # Şirket adı ve hisse kodu ile arama
        search_terms = [company_name, stock_code]
        
        all_news = []
        
        for term in search_terms:
            # SASA için daha spesifik arama
            if stock_code == 'SASA':
                search_query = f'"{term}"'  # Tırnak içinde arama yap
            else:
                search_query = term
                
            params = urllib.parse.urlencode({
                'api_token': '4VPDfIzFhUWoRyzmwtWso4rREi9fsIw18CSOcsx9',
                'categories': 'business,tech',
                'limit': 3,  # Ücretsiz plan limiti
                'language': 'tr',
                'locale': 'tr',
                'search': search_query
            })
            
            conn.request('GET', '/v1/news/all?{}'.format(params))
            res = conn.getresponse()
            data = res.read()
            
            news_data = json.loads(data.decode('utf-8'))
            
            # API yanıtını kontrol et
            if news_data.get('data') and len(news_data.get('data', [])) > 0:
                all_news.extend(news_data.get('data', []))
        
        return {
            'success': len(all_news) > 0,
            'news': all_news,
            'total': len(all_news)
        }
        
    except Exception as e:
        print(f"Şirket haberleri alınırken hata: {e}")
        return {
            'success': False,
            'error': str(e),
            'news': [],
            'total': 0
        }

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
    'YATAS', 'YUNSA', 'ZRGYO','FROTO'
]

# Şirket isimlerini hisse kodlarına eşleyen sözlük
COMPANY_TO_CODE = {
    # BIST30 şirketleri
    'akbank': 'AKBNK', 'arcelik': 'ARCLK', 'aselsan': 'ASELS', 'bim': 'BIMAS',
    'ekonomi': 'EKGYO', 'enka': 'ENKAI', 'eupwr': 'EUPWR', 'ford otosan': 'FROTO',
    'garanti': 'GARAN', 'gubre fabrikalari': 'GUBRF', 'hektas': 'HEKTS', 'isbank': 'ISCTR',
    'koç holding': 'KCHOL', 'kardemir': 'KRDMD', 'koza altin': 'KOZAA', 'koza anadolu': 'KOZAL',
    'migros': 'MGROS', 'pegasus': 'PGSUS', 'sabanci holding': 'SAHOL', 'sasa': 'SASA',
    'sise cam': 'SISE', 'turkcell': 'TCELL', 'turk hava yollari': 'THYAO', 'turk telekom': 'TKFEN',
    'toyota otosan': 'TOASO', 'tüpras': 'TUPRS', 'vakifbank': 'VAKBN', 'yapi kredi': 'YKBNK',
    'coca cola': 'CCOLA', 'eregli demir celik': 'EREGL', 'soda sanayi': 'SODA', 'petkim': 'PETKM',
    
    # BIST50 ek şirketleri
    'ahlat': 'AHLAT', 'aksa': 'AKSA', 'alarko': 'ALARK', 'albayrak': 'ALBRK',
    'alarko gayrimenkul': 'ALGYO', 'anadolu cam': 'ANACM', 'bascm': 'BASCM', 'bera': 'BERA',
    'brisa': 'BRISA', 'bryat': 'BRYAT', 'cemas': 'CEMAS', 'cemts': 'CEMTS',
    'cimsa': 'CIMSA', 'dogus holding': 'DOHOL', 'ege enerji': 'EGEEN', 'enerjisa': 'ENJSA',
    'fmizp': 'FMIZP', 'gesan': 'GESAN', 'glyho': 'GLYHO', 'halkbank': 'HALKB',
    'hateks': 'HATEK', 'indes': 'INDES', 'ipek': 'IPEKE', 'karel': 'KAREL',
    'karsan': 'KARSN', 'kervn': 'KERVN', 'kervansaray': 'KERVT', 'kontr': 'KONTR',
    'konya': 'KONYA', 'logo': 'LOGO', 'macko': 'MACKO', 'netas': 'NETAS',
    'nthol': 'NTHOL', 'odas': 'ODAS', 'otokar': 'OTKAR', 'oyak': 'OYAKC',
    'penta': 'PENTA', 'polisan': 'POLHO', 'prkab': 'PRKAB', 'prkme': 'PRKME',
    'quagr': 'QUAGR', 'safkn': 'SAFKN', 'selec': 'SELEC', 'selgd': 'SELGD',
    'smrtg': 'SMRTG', 'sngyo': 'SNGYO', 'sokm': 'SOKM', 'tav': 'TAVHL',
    'tknsa': 'TKNSA', 'trgyo': 'TRGYO', 'tskb': 'TSKB', 'ttkom': 'TTKOM',
    'ttrak': 'TTRAK', 'ülker': 'ULKER', 'vesbe': 'VESBE', 'vestel': 'VESTL',
    'yatas': 'YATAS', 'yunsa': 'YUNSA', 'zrgyo': 'ZRGYO', 'frodo': 'FRODO',
    
    # Alternatif yazımlar
    'coca-cola': 'CCOLA', 'coca cola icecek': 'CCOLA', 'koç': 'KCHOL', 'koç grubu': 'KCHOL',
    'turk hava yollari': 'THYAO', 'thy': 'THYAO', 'turk hava': 'THYAO',
    'turk telekomünikasyon': 'TKFEN', 'turk telekom': 'TKFEN',
    'toyota otosan': 'TOASO', 'toyota': 'TOASO',
    'tüpras': 'TUPRS', 'tupras': 'TUPRS', 'türkiye petrol rafinerileri': 'TUPRS',
    'garanti bankasi': 'GARAN', 'garanti': 'GARAN',
    'is bankasi': 'ISCTR', 'isbank': 'ISCTR',
    'yapi kredi bankasi': 'YKBNK', 'yapi kredi': 'YKBNK', 'yapikredi': 'YKBNK',
    'vakifbank': 'VAKBN', 'vakif': 'VAKBN',
    'sabanci': 'SAHOL', 'sabanci holding': 'SAHOL',
    'arcelik': 'ARCLK', 'arcelik a.ş.': 'ARCLK',
    'bim': 'BIMAS', 'bim birlesik magazalar': 'BIMAS',
    'migros': 'MGROS', 'migros ticaret': 'MGROS',
    'sasa': 'SASA', 'sasa polyurethan': 'SASA',
    'turkcell': 'TCELL', 'turkcell iletisim': 'TCELL',
    'sise cam': 'SISE', 'sise': 'SISE',
    'eregli': 'EREGL', 'eregli demir': 'EREGL', 'eregli demir çelik': 'EREGL',
    'petkim': 'PETKM', 'petkim petrokimya': 'PETKM',
    'soda sanayi': 'SODA', 'soda': 'SODA',
    'ülker': 'ULKER', 'ulker': 'ULKER',
    'vestel': 'VESTL', 'vestel elektronik': 'VESTL',
    'pegasus': 'PGSUS', 'pegasus havacilik': 'PGSUS',
    'ford otosan': 'FROTO', 'ford': 'FROTO',
    'akbank': 'AKBNK', 'akbank t.a.ş.': 'AKBNK',
    'aselsan': 'ASELS', 'aselsan elektronik': 'ASELS',
    'enka': 'ENKAI', 'enka insaat': 'ENKAI',
    'ekonomi': 'EKGYO', 'ekonomi bankasi': 'EKGYO',
    'hektas': 'HEKTS', 'hektas insaat': 'HEKTS',
    'kardemir': 'KRDMD', 'kardemir demir celik': 'KRDMD',
    'koza altin': 'KOZAA', 'koza': 'KOZAA',
    'koza anadolu': 'KOZAL', 'koza anadolu metal': 'KOZAL'
}

load_dotenv("api.env")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ENV değişkenlerini yükle
load_dotenv("api.env")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def ask_groq(prompt: str) -> str:
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GROQ_API_KEY}"
    }
    payload = {
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "messages": [
            {
                "role": "system",
                "content": "Sen profesyonel bir finans danışmanısın. Türkiye borsasındaki hisse senetleri hakkında doğru, daha özet ve yatırımcı dostu cevaplar ver."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        return f"API Error: {response.status_code} - {response.text}"

# LLM ile şirket ismini hisse koduna çeviren fonksiyon
def get_stock_code_from_llm(company_name):
    prompt = (
        f"Kullanıcı '{company_name}' şirketinin Borsa İstanbul'daki hisse kodunu soruyor. "
        "Sadece hisse kodunu, başka hiçbir şey yazmadan, büyük harflerle döndür."
    )
    code = ask_groq(prompt)
    code = code.strip().upper()
    if code in BIST_STOCKS:
        return code
    return None

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
        
        # Türk borsa haberleri sorusu
        if any(word in question_lower for word in ['türk borsa haberleri', 'güncel haberler', 'borsa haberleri', 'turkish news', 'market news']):
            print("Detected 'Turkish stock news' question")
            try:
                news_data = get_turkish_stock_news()
                if news_data['success'] and news_data['news']:
                    if language == 'tr':
                        answer = f"📰 GÜNCEL TÜRK BORSA HABERLERİ ({news_data['total']} haber):\n\n"
                    else:
                        answer = f"📰 CURRENT TURKISH MARKET NEWS ({news_data['total']} articles):\n\n"
                    
                    for i, article in enumerate(news_data['news'][:5], 1):
                        answer += f"{i}. {article.get('title', 'Başlık yok')}\n"
                        answer += f"   📰 {article.get('source', 'Bilinmeyen')} | {article.get('published_at', 'Tarih bilgisi yok')[:10]}\n"
                        if article.get('description'):
                            answer += f"   📝 {article['description'][:100]}...\n"
                        answer += "\n"
                else:
                    if language == 'tr':
                        answer = "❌ Haber bulunamadı. Lütfen daha sonra tekrar deneyin."
                    else:
                        answer = "❌ No news found. Please try again later."
                
                return {"answer": answer, "chart": None}
            except Exception as e:
                print(f"Error getting Turkish news: {e}")
                if language == 'tr':
                    return {"answer": "❌ Haber alınırken bir hata oluştu.", "chart": None}
                else:
                    return {"answer": "❌ An error occurred while fetching news.", "chart": None}
        
        # Hisse kodu var mı? (ör: CCOLA, BIMAS, THYAO)
        hisse = None
        
        # 1. Önce doğrudan hisse kodu ara
        for code in BIST_STOCKS:
            if code.lower() in question_lower:
                hisse = code
                break
        
        # 2. Eğer hisse kodu bulunamazsa, şirket ismi sözlüğünde ara
        if not hisse:
            # En uzun eşleşmeyi bul (daha spesifik eşleşme için)
            best_match = None
            best_match_length = 0
            
            for company_name, stock_code in COMPANY_TO_CODE.items():
                if company_name in question_lower:
                    if len(company_name) > best_match_length:
                        best_match = stock_code
                        best_match_length = len(company_name)
            
            if best_match:
                hisse = best_match
                print(f"Found best company match (length {best_match_length}) -> '{best_match}'")
        
        # 3. Eğer hala bulunamazsa, LLM ile bulmayı dene
        if not hisse:
            llm_code = get_stock_code_from_llm(question)
            if llm_code:
                hisse = llm_code
        
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
        if hisse and any(word in question_lower for word in ['grafik','çiz','grafiği','çizdir','grafiğini','chart', 'görsel']):
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
                    llm_advice = ask_groq(llm_prompt)
                    if llm_advice and not llm_advice.startswith("API Error"):
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
        
        # Sosyal medya sentiment analizi sorusu mu? (ör: ASELS sosyal medya, ASELS haber analizi)
        if hisse and any(word in question_lower for word in ['sosyal medya', 'haber', 'sentiment', 'analiz', 'hava', 'genel hava', 'medya', 'news', 'social media','medya analizi','haber analizi','haberler','sentiment analizi']):
            print(f"Getting social media sentiment analysis for {hisse}")
            try:
                # Şirket adını bul
                company_name = hisse
                for name, code in COMPANY_TO_CODE.items():
                    if code == hisse:
                        company_name = name
                        break
                
                # Sentiment analizi yap
                sentiment_data = get_news_sentiment(company_name, hisse)
                
                if sentiment_data['error']:
                    if language == 'tr':
                        answer = f"❌ {hisse} için haber analizi yapılamadı: {sentiment_data['error']}"
                    else:
                        answer = f"❌ Could not analyze news for {hisse}: {sentiment_data['error']}"
                    return {"answer": answer, "chart": None}
                
                # Trend analizi yap
                trend_data = analyze_sentiment_trend(sentiment_data)
                
                # Sektör analizi yap
                sector_data = analyze_sector_sentiment(sentiment_data, hisse)
                
                # Sentiment özeti oluştur
                summary = get_sentiment_summary(sentiment_data, trend_data)
                
                # Sentiment skoruna göre emoji ve renk
                sentiment_score = sentiment_data['sentiment_score']
                if sentiment_score > 0.1:
                    sentiment_emoji = "🟢"
                    sentiment_color = "Olumlu"
                elif sentiment_score < -0.1:
                    sentiment_emoji = "🔴"
                    sentiment_color = "Olumsuz"
                else:
                    sentiment_emoji = "🟡"
                    sentiment_color = "Nötr"
                
                if language == 'tr':
                    answer = f"📰 {hisse} SOSYAL MEDYA SENTIMENT ANALİZİ:\n\n"
                    answer += f"{sentiment_emoji} Genel Hava: {sentiment_color}\n"
                    answer += f"📊 Sentiment Skoru: {sentiment_score:.3f}\n"
                    #answer += f"🎯 Güven Skoru: {sentiment_data['confidence']:.2f}\n"
                    answer += f"📈 Toplam Haber: {sentiment_data['news_count']} adet\n"
                    #answer += f"✅ Olumlu Haber: {sentiment_data['positive_news']} adet\n"
                    #answer += f"❌ Olumsuz Haber: {sentiment_data['negative_news']} adet\n"
                    answer += f"⚪ Nötr Haber: {sentiment_data['neutral_news']} adet\n\n"
                    
                    # Trend analizi
                    trend_emoji = "📈" if trend_data['trend'] == 'Yükseliş' else "📉" if trend_data['trend'] == 'Düşüş' else "📊"
                    answer += f"{trend_emoji} Trend Analizi: {trend_data['trend']}\n"
                    #answer += f"📊 Trend Skoru: {trend_data['trend_score']:.3f}\n"
                    answer += f"📝 Trend Açıklaması: {trend_data['trend_description']}\n\n"
                    
                    # Özet
                    answer += f"💡 Analiz Özeti:\n{summary}\n\n"
                    
                    # Sektör analizi
                    sector_emoji = "🏭" if sector_data['sector'] in ['Demir-Çelik', 'Kimya', 'Cam'] else \
                                  "🏦" if sector_data['sector'] == 'Bankacılık' else \
                                  "✈️" if sector_data['sector'] == 'Havacılık' else \
                                  "📱" if sector_data['sector'] == 'Telekomünikasyon' else \
                                  "⚡" if sector_data['sector'] == 'Enerji' else \
                                  "🛡️" if sector_data['sector'] == 'Savunma' else \
                                  "🏢" if sector_data['sector'] == 'Holding' else \
                                  "🚗" if sector_data['sector'] == 'Otomotiv' else \
                                  "🛒" if sector_data['sector'] == 'Perakende' else \
                                  "🥤" if sector_data['sector'] == 'İçecek' else \
                                  "🍽️" if sector_data['sector'] == 'Gıda' else "📊"
                    
                    answer += f"{sector_emoji} Sektör Analizi: {sector_data['sector']}\n"
                    #answer += f"📊 Sektör Sentiment: {sector_data['sector_sentiment']:.3f}\n"
                    #answer += f"🎯 Sektör Uygunluğu: {sector_data['sector_relevance']:.2f}\n"
                    
                    if sector_data['sector_keywords_found']:
                        answer += f"🔑 Sektör Anahtar Kelimeleri: {', '.join(sector_data['sector_keywords_found'][:3])}\n"
                    answer += "\n"
                    

                    
                    # Anahtar kelimeler
                    if sentiment_data['top_key_phrases']:
                        answer += "🔑 Anahtar Kelimeler:\n"
                        for phrase, count in sentiment_data['top_key_phrases']:
                            answer += f"   • {phrase}: {count} kez\n"
                        answer += "\n"
                    
                    if sentiment_data['recent_news']:
                        answer += "📰 Son Haberler:\n"
                        for i, news in enumerate(sentiment_data['recent_news'], 1):
                            answer += f"   {i}. {news['title'][:60]}...\n"
                            answer += f"      📰 {news['source']} | {news['published_at'][:10]}\n"
                            if news.get('url'):
                                answer += f"      🔗 Haber Linki: {news['url']}\n"
                    
                    answer += "\n💡 Bu analiz son 7 günün haberlerine dayanmaktadır."
                else:
                    answer = f"📰 {hisse} SOCIAL MEDIA SENTIMENT ANALYSIS:\n\n"
                    answer += f"{sentiment_emoji} General Sentiment: {sentiment_color}\n"
                    answer += f"📊 Sentiment Score: {sentiment_score:.3f}\n"
                    answer += f"🎯 Confidence Score: {sentiment_data['confidence']:.2f}\n"
                    answer += f"📈 Total News: {sentiment_data['news_count']} articles\n"
                    #answer += f"✅ Positive News: {sentiment_data['positive_news']} articles\n"
                    #answer += f"❌ Negative News: {sentiment_data['negative_news']} articles\n"
                    answer += f"⚪ Neutral News: {sentiment_data['neutral_news']} articles\n\n"
                    
                    # Trend analysis
                    trend_emoji = "📈" if trend_data['trend'] == 'Yükseliş' else "📉" if trend_data['trend'] == 'Düşüş' else "📊"
                    answer += f"{trend_emoji} Trend Analysis: {trend_data['trend']}\n"
                    #answer += f"📊 Trend Score: {trend_data['trend_score']:.3f}\n"
                    answer += f"📝 Trend Description: {trend_data['trend_description']}\n\n"
                    
                    # Summary
                    answer += f"💡 Analysis Summary:\n{summary}\n\n"
                    
                    # Sector analysis
                    sector_emoji = "🏭" if sector_data['sector'] in ['Demir-Çelik', 'Kimya', 'Cam'] else \
                                  "🏦" if sector_data['sector'] == 'Bankacılık' else \
                                  "✈️" if sector_data['sector'] == 'Havacılık' else \
                                  "📱" if sector_data['sector'] == 'Telekomünikasyon' else \
                                  "⚡" if sector_data['sector'] == 'Enerji' else \
                                  "🛡️" if sector_data['sector'] == 'Savunma' else \
                                  "🏢" if sector_data['sector'] == 'Holding' else \
                                  "🚗" if sector_data['sector'] == 'Otomotiv' else \
                                  "🛒" if sector_data['sector'] == 'Perakende' else \
                                  "🥤" if sector_data['sector'] == 'İçecek' else \
                                  "🍽️" if sector_data['sector'] == 'Gıda' else "📊"
                    
                    answer += f"{sector_emoji} Sector Analysis: {sector_data['sector']}\n"
                    #answer += f"📊 Sector Sentiment: {sector_data['sector_sentiment']:.3f}\n"
                    #answer += f"🎯 Sector Relevance: {sector_data['sector_relevance']:.2f}\n"
                    
                    if sector_data['sector_keywords_found']:
                        answer += f"🔑 Sector Keywords: {', '.join(sector_data['sector_keywords_found'][:3])}\n"
                    answer += "\n"
                    
                    # Emotion analysis

                    
                    # Key phrases
                    if sentiment_data['top_key_phrases']:
                        answer += "🔑 Key Phrases:\n"
                        for phrase, count in sentiment_data['top_key_phrases']:
                            answer += f"   • {phrase}: {count} times\n"
                        answer += "\n"
                    
                    if sentiment_data['recent_news']:
                        answer += "📰 Recent News:\n"
                        for i, news in enumerate(sentiment_data['recent_news'], 1):
                            answer += f"   {i}. {news['title'][:60]}...\n"
                            answer += f"      📰 {news['source']} | {news['published_at'][:10]}\n"
                            if news.get('url'):
                                answer += f"      🔗 Haber Linki: {news['url']}\n"
                    
                    answer += "\n💡 This analysis is based on news from the last 7 days."
                
                return {"answer": answer, "chart": None}
            except Exception as e:
                print(f"Error processing sentiment analysis: {e}")
                if language == 'tr':
                    return {"answer": f"❌ {hisse} için sentiment analizi yapılamadı.", "chart": None}
                else:
                    return {"answer": f"❌ Could not perform sentiment analysis for {hisse}.", "chart": None}
        
        # Eğer hiçbir anahtar kelimeye uymuyorsa, LLM'e sor
        print("No specific command detected, using LLM for general questions")
        try:
            llm_response = ask_groq(question)
            return {"answer": llm_response, "chart": None}
        except Exception as e:
            print(f"Error getting LLM response: {e}")
            # LLM hatası durumunda yardım mesajı döndür
            if language == 'tr':
                answer = "🤖 FINBOT Size Nasıl Yardımcı Olabilir?\n\n"
                answer += "📈 Güncel fiyat: 'ARCLK güncel fiyat'\n"
                answer += "📊 Grafik: 'ARCLK grafik' veya 'ARCLK 3 ay grafik'\n"
                answer += "🔮 Tahmin: 'ARCLK tahmin' veya 'ARCLK forecast'\n"
                answer += "📰 Sentiment: 'ASELS medya analizi' veya 'ASELS haber analizi'\n"
                answer += "💼 Yatırım tavsiyesi: '1000 TL ne alayım' veya 'portföy önerisi'\n"
                answer += "📋 Hisse listesi: 'Hangi şirketler mevcut'\n\n"
                answer += "💡 Grafik süreleri: 1 ay, 3 ay, 6 ay, 1 yıl\n"
                answer += "💡 Tahmin yöntemleri: Prophet, ARIMA, LSTM\n"
                answer += "💡 Risk profilleri: Düşük, Orta, Yüksek\n"
                answer += "💡 Sentiment analizi: Son 7 günün haberleri"
            else:
                answer = "🤖 How can FINBOT help you?\n\n"
                answer += "📈 Current price: 'ARCLK current price'\n"
                answer += "📊 Chart: 'ARCLK chart' or 'ARCLK 3 months chart'\n"
                answer += "🔮 Forecast: 'ARCLK forecast' or 'ARCLK prediction'\n"
                answer += "📰 Sentiment: 'ASELS social media' or 'ASELS news analysis'\n"
                answer += "💼 Investment advice: 'What should I buy with 1000 TL' or 'portfolio recommendation'\n"
                answer += "📋 Stock list: 'Which companies are available'\n\n"
                answer += "💡 Chart periods: 1 month, 3 months, 6 months, 1 year\n"
                answer += "💡 Forecast methods: Prophet, ARIMA, LSTM\n"
                answer += "💡 Risk profiles: Low, Medium, High\n"
                answer += "💡 Sentiment analysis: News from last 7 days"
        
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

@app.get("/turkish-news")
def get_turkish_news():
    """Genel Türk borsa haberlerini al"""
    try:
        news_data = get_turkish_stock_news()
        return news_data
    except Exception as e:
        return {"success": False, "error": str(e), "news": [], "total": 0}

@app.get("/company-news/{stock_code}")
def get_company_news(stock_code: str):
    """Belirli bir şirket için Türkçe haberleri al"""
    try:
        # Şirket adını bul
        company_name = stock_code
        for name, code in COMPANY_TO_CODE.items():
            if code == stock_code:
                company_name = name
                break
        
        news_data = get_turkish_stock_news_by_company(company_name, stock_code)
        return news_data
    except Exception as e:
        return {"success": False, "error": str(e), "news": [], "total": 0}



def analyze_sentiment_trend(sentiment_data: Dict) -> Dict:
    """Sentiment trend analizi"""
    if not sentiment_data['recent_news']:
        return {'trend': 'Belirsiz', 'trend_score': 0.0, 'trend_description': 'Yeterli veri yok'}
    
    # Haberleri tarihe göre sırala
    sorted_news = sorted(sentiment_data['recent_news'], 
                        key=lambda x: x['published_at'], reverse=True)
    
    if len(sorted_news) < 2:
        return {'trend': 'Belirsiz', 'trend_score': 0.0, 'trend_description': 'Yeterli veri yok'}
    
    # Son 3 haber ile önceki 3 haberin ortalamasını karşılaştır
    recent_sentiments = [news['sentiment'] for news in sorted_news[:3]]
    older_sentiments = [news['sentiment'] for news in sorted_news[3:6]] if len(sorted_news) >= 6 else []
    
    recent_avg = sum(recent_sentiments) / len(recent_sentiments)
    older_avg = sum(older_sentiments) / len(older_sentiments) if older_sentiments else recent_avg
    
    trend_score = recent_avg - older_avg
    
    if trend_score > 0.1:
        trend = 'Yükseliş'
        trend_description = 'Sentiment pozitif yönde gelişiyor'
    elif trend_score < -0.1:
        trend = 'Düşüş'
        trend_description = 'Sentiment negatif yönde gelişiyor'
    else:
        trend = 'Stabil'
        trend_description = 'Sentiment stabil seyrediyor'
    
    return {
        'trend': trend,
        'trend_score': trend_score,
        'trend_description': trend_description,
        'recent_avg': recent_avg,
        'older_avg': older_avg
    }

def get_sentiment_summary(sentiment_data: Dict, trend_data: Dict) -> str:
    """Sentiment özeti oluştur"""
    summary = ""
    
    # Genel durum
    if sentiment_data['sentiment_score'] > 0.2:
        summary += "📈 Genel olarak çok olumlu bir hava var. "
    elif sentiment_data['sentiment_score'] > 0.05:
        summary += "📊 Genel olarak olumlu bir hava var. "
    elif sentiment_data['sentiment_score'] < -0.2:
        summary += "📉 Genel olarak çok olumsuz bir hava var. "
    elif sentiment_data['sentiment_score'] < -0.05:
        summary += "📊 Genel olarak olumsuz bir hava var. "
    else:
        summary += "📊 Genel olarak nötr bir hava var. "
    
    # Trend
    if trend_data['trend'] == 'Yükseliş':
        summary += "Trend pozitif yönde gelişiyor. "
    elif trend_data['trend'] == 'Düşüş':
        summary += "Trend negatif yönde gelişiyor. "
    else:
        summary += "Trend stabil seyrediyor. "
    
    # Güven
    if sentiment_data['confidence'] > 0.7:
        summary += "Analiz sonuçları yüksek güvenilirlikte. "
    elif sentiment_data['confidence'] > 0.4:
        summary += "Analiz sonuçları orta güvenilirlikte. "
    else:
        summary += "Analiz sonuçları düşük güvenilirlikte. "
    
    # Ana duygular
    if sentiment_data['top_emotions']:
        top_emotion = sentiment_data['top_emotions'][0]
        summary += f"En yaygın duygu: {top_emotion[0]} ({top_emotion[1]} kez). "
    
    return summary

# Sektör tanımları
SECTOR_DEFINITIONS = {
    'AKBNK': 'Bankacılık', 'GARAN': 'Bankacılık', 'ISCTR': 'Bankacılık', 'YKBNK': 'Bankacılık', 'VAKBN': 'Bankacılık',
    'THYAO': 'Havacılık', 'PGSUS': 'Havacılık',
    'TCELL': 'Telekomünikasyon',
    'TUPRS': 'Enerji', 'ENJSA': 'Enerji', 'ENKAI': 'Enerji',
    'ASELS': 'Savunma', 'ASELSAN': 'Savunma',
    'EREGL': 'Demir-Çelik', 'KRDMD': 'Demir-Çelik',
    'KCHOL': 'Holding', 'SAHOL': 'Holding',
    'FROTO': 'Otomotiv', 'TOASO': 'Otomotiv',
    'BIMAS': 'Perakende', 'MGROS': 'Perakende',
    'SASA': 'Kimya', 'SISE': 'Cam',
    'CCOLA': 'İçecek',
    'SOKM': 'Gıda', 'ULKER': 'Gıda'
}

# Sektör bazlı anahtar kelimeler
SECTOR_KEYWORDS = {
    'Bankacılık': ['kredi', 'mevduat', 'faiz', 'banka', 'finans', 'kredi kartı', 'mortgage', 'leasing'],
    'Havacılık': ['uçuş', 'uçak', 'havayolu', 'terminal', 'bagaj', 'bilet', 'rota', 'pilot'],
    'Telekomünikasyon': ['mobil', 'internet', '5g', 'telefon', 'operatör', 'tarife', 'veri', 'şebeke'],
    'Enerji': ['petrol', 'rafineri', 'elektrik', 'doğalgaz', 'enerji', 'yakıt', 'boru hattı'],
    'Savunma': ['savunma', 'silah', 'radar', 'elektronik', 'askeri', 'teknoloji', 'proje'],
    'Demir-Çelik': ['çelik', 'demir', 'metal', 'üretim', 'fabrika', 'hammadde', 'hurda'],
    'Holding': ['holding', 'şirket', 'yatırım', 'portföy', 'diversifikasyon', 'strateji'],
    'Otomotiv': ['araç', 'otomobil', 'fabrika', 'üretim', 'satış', 'model', 'motor'],
    'Perakende': ['market', 'mağaza', 'satış', 'ürün', 'fiyat', 'kampanya', 'müşteri'],
    'Kimya': ['kimya', 'polietilen', 'plastik', 'petrokimya', 'hammadde', 'üretim'],
    'Cam': ['cam', 'şişe', 'ambalaj', 'üretim', 'geri dönüşüm'],
    'İçecek': ['içecek', 'meşrubat', 'şişe', 'kutu', 'üretim', 'dağıtım']
}

def get_company_sector(stock_code: str) -> str:
    """Şirketin sektörünü döndür"""
    return SECTOR_DEFINITIONS.get(stock_code, 'Genel')

def analyze_sector_sentiment(sentiment_data: Dict, stock_code: str) -> Dict:
    """Sektör bazlı sentiment analizi"""
    sector = get_company_sector(stock_code)
    sector_keywords = SECTOR_KEYWORDS.get(sector, [])
    
    if not sentiment_data['recent_news']:
        return {
            'sector': sector,
            'sector_sentiment': 0.0,
            'sector_keywords_found': [],
            'sector_relevance': 0.0
        }
    
    sector_sentiments = []
    sector_keywords_found = []
    total_relevance = 0.0
    
    for news in sentiment_data['recent_news']:
        title = news.get('title', '')
        content = title.lower()
        
        # Sektör anahtar kelimelerini ara
        found_keywords = [keyword for keyword in sector_keywords if keyword.lower() in content]
        if found_keywords:
            sector_keywords_found.extend(found_keywords)
            sector_sentiments.append(news['sentiment'])
            total_relevance += 1
    
    sector_sentiment = sum(sector_sentiments) / len(sector_sentiments) if sector_sentiments else 0.0
    sector_relevance = total_relevance / len(sentiment_data['recent_news']) if sentiment_data['recent_news'] else 0.0
    
    return {
        'sector': sector,
        'sector_sentiment': sector_sentiment,
        'sector_keywords_found': list(set(sector_keywords_found)),
        'sector_relevance': sector_relevance
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 