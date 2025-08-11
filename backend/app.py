def remove_non_latin(text):
    # Sadece Latin harfleri, rakamlar, Türkçe karakterler ve temel noktalama işaretlerini bırak
    import re
    # Türkçe karakterler dahil Latin harfler
    allowed = r"[^a-zA-Z0-9çÇğĞıİöÖşŞüÜ.,:;!?()\[\]{}<>@#$%^&*\-_=+/'\"\s]"
    return re.sub(allowed, '', text)
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
                search_terms.extend(['İş Bankası','iş bankası' 'İşbank', 'isbank', 'ISBANK', 'Türkiye bankacılık', 'banka', 'finans'])
            elif stock_code == 'YKBNK':
                search_terms.extend(['Yapı Kredi', 'Yapıkredi', 'yapikredi', 'YAPIKREDI', 'Türkiye bankacılık', 'banka', 'finans'])
            elif stock_code == 'VAKBN':
                search_terms.extend(['Vakıfbank','vakıfbank' 'vakifbank', 'VAKIFBANK', 'Türkiye bankacılık', 'banka', 'finans'])
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
    """Prophet ile hisse fiyat tahmini (sadece 1 gün sonrası, tatil/haftasonu kontrolü ile)"""
    try:
        import pandas as pd
        yf_symbol = f"{symbol}.IS"
        ticker = yf.Ticker(yf_symbol)
        # Try to get last 1 month of data, fallback to 10 days if needed
        hist = ticker.history(period="1mo")
        if hist.empty or len(hist) < 10:
            hist = ticker.history(period="10d")
        if hist.empty or len(hist) < 5:
            print(f"Not enough data for {symbol}, got {len(hist)} rows.")
            # Try to extrapolate from whatever is available
            if not hist.empty:
                df = hist.reset_index()
                df = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
                df['ds'] = df['ds'].dt.tz_localize(None)
                df = df.dropna()
                model = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False, changepoint_prior_scale=0.01)
                model.fit(df)
                last_date = df['ds'].max()
                future_dates = []
                d = last_date
                while len(future_dates) < 5:
                    d += pd.Timedelta(days=1)
                    if d.weekday() < 5:
                        future_dates.append(d)
                future = pd.DataFrame({'ds': future_dates})
                forecast = model.predict(future)
                return {
                    'dates': [d.timestamp() for d in df['ds']] + [d.timestamp() for d in future_dates],
                    'actuals': df['y'].tolist(),
                    'predictions': forecast['yhat'].tolist(),
                    'pred_dates': [d.timestamp() for d in future_dates],
                    'lower': forecast['yhat_lower'].tolist(),
                    'upper': forecast['yhat_upper'].tolist()
                }
            else:
                return None
        # Use available data (>=5 rows)
        df = hist.reset_index()
        df = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
        df['ds'] = df['ds'].dt.tz_localize(None)
        df = df.dropna()
        model = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False, changepoint_prior_scale=0.01)
        model.fit(df)
        # Son 5 iş gününü al
        last5_hist = hist[hist.index.dayofweek < 5].tail(5)
        # Sonraki 5 iş günü için tahmin
        last_date = df['ds'].max()
        future_dates = []
        d = last_date
        while len(future_dates) < 5:
            d += pd.Timedelta(days=1)
            if d.weekday() < 5:
                future_dates.append(d)
        future = pd.DataFrame({'ds': future_dates})
        forecast = model.predict(future)
        return {
            'dates': [d.timestamp() for d in last5_hist.index] + [d.timestamp() for d in future_dates],
            'actuals': last5_hist['Close'].tolist(),
            'predictions': forecast['yhat'].tolist(),
            'pred_dates': [d.timestamp() for d in future_dates],
            'lower': forecast['yhat_lower'].tolist(),
            'upper': forecast['yhat_upper'].tolist()
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
        return f"📋 MEVCUT HİSSELER ( gerçek zamanlı veri):\n\n{stocks_list}\n\n💡 Örnek kullanım:\n• 'CCOLA güncel fiyat'\n• 'THYAO grafik'\n• 'GARAN haber'"
    else:
        stocks_list = "\n".join([f"• {stock}" for stock in BIST_STOCKS])
        return f"📋 AVAILABLE STOCKS (Real-time data):\n\n{stocks_list}\n\n💡 Example usage:\n• 'CCOLA current price'\n• 'THYAO chart'\n• 'GARAN news'"
# Türkiye resmi tatilleri - 2025
TURKEY_HOLIDAYS_2025 = {
    datetime(2025, 1, 1),   # Yeni Yıl
    datetime(2025, 3, 29),  # Ramazan Bayramı Arefesi
    datetime(2025, 3, 30),
    datetime(2025, 3, 31),
    datetime(2025, 4, 1),
    datetime(2025, 4, 23),  # Ulusal Egemenlik ve Çocuk Bayramı
    datetime(2025, 5, 1),   # Emek ve Dayanışma Günü
    datetime(2025, 5, 19),  # Atatürk'ü Anma, Gençlik ve Spor Bayramı
    datetime(2025, 6, 5),   # Kurban Bayramı Arefesi
    datetime(2025, 6, 6),
    datetime(2025, 6, 7),
    datetime(2025, 6, 8),
    datetime(2025, 6, 9),
    datetime(2025, 7, 15),  # Demokrasi ve Milli Birlik Günü
    datetime(2025, 8, 30),  # Zafer Bayramı
    datetime(2025, 10, 28), # Cumhuriyet Bayramı Arefesi
    datetime(2025, 10, 29)  # Cumhuriyet Bayramı
}

def get_next_trading_day(date):
    """Hafta sonlarını ve Türkiye 2025 resmi tatillerini atlayarak bir sonraki işlem gününü döndürür."""
    next_day = date + timedelta(days=1)
    while (
        next_day.weekday() >= 5  # Cumartesi/Pazar
        or next_day in TURKEY_HOLIDAYS_2025
    ):
        next_day += timedelta(days=1)
    return next_day

@app.post("/ask")
async def ask_question(question: str = Form(...), language: str = Form("tr")):
    import unicodedata

    def normalize_text(text):
        text = text.lower()
        text = unicodedata.normalize('NFD', text)
        text = ''.join([c for c in text if unicodedata.category(c) != 'Mn'])
        return text

    print(f"Received request - question: '{question}', language: '{language}'")
    question_lower = normalize_text(question)
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
                answer = "❌ Haber bulunamadı. Lütfen daha sonra tekrar deneyin." if language == 'tr' else "❌ No news found. Please try again later."
            return {"answer": answer, "chart": None}
        except Exception as e:
            print(f"Error getting Turkish news: {e}")
            error_msg = "❌ Haber alınırken bir hata oluştu." if language == 'tr' else "❌ An error occurred while fetching news."
            return {"answer": error_msg, "chart": None}

    # Hisse kodu tespiti
    hisse_list = []
    for code in BIST_STOCKS:
        if code.lower() in question_lower:
            hisse_list.append(code)
    for company_name, stock_code in COMPANY_TO_CODE.items():
        if company_name in question_lower and stock_code not in hisse_list:
            hisse_list.append(stock_code)
    if not hisse_list:
        llm_code = get_stock_code_from_llm(question)
        if llm_code:
            hisse_list.append(llm_code)
    hisse_list = hisse_list[:2]
    hisse = hisse_list[0] if hisse_list else None

    # Forecasting sorusu mu?
    if hisse and any(word in question_lower for word in ['tahmin', 'forecast', 'gelecek', 'future', 'prediction']):
        print(f"Getting forecast for {hisse}")
        try:
            forecast_method = 'prophet'
            if 'arima' in question_lower:
                forecast_method = 'arima'
            elif 'lstm' in question_lower or 'neural' in question_lower:
                forecast_method = 'lstm'
            elif 'prophet' in question_lower:
                forecast_method = 'prophet'

            # Tahmin verisini al
            if forecast_method == 'prophet':
                forecast_data = get_forecast_prophet(hisse, days=5)
            elif forecast_method == 'arima':
                forecast_data = get_forecast_arima(hisse, days=5)
            elif forecast_method == 'lstm':
                forecast_data = get_forecast_lstm(hisse, days=5)
            else:
                forecast_data = get_forecast_prophet(hisse, days=5)

            if forecast_data and forecast_data.get('predictions'):
                dates = [datetime.fromtimestamp(ts) for ts in forecast_data['dates']]
                actuals = forecast_data.get('actuals', [])
                preds = forecast_data['predictions']
                lowers = forecast_data.get('lower', [])
                uppers = forecast_data.get('upper', [])

                plt.figure(figsize=(10, 6))
                # Son 5 gün mavi çizgi
                if actuals:
                    plt.plot(dates[:len(actuals)], actuals, marker='o', color='blue', label='Son Günler Fiyatı')
                # Tahmin edilen günler kırmızı çizgi
                plt.plot(dates[len(actuals):], preds, marker='o', color='red', label='Tahmin (Sonraki 5 İş Günü)')
                # Prophet güven aralığı
                if lowers and uppers:
                    plt.fill_between(dates[len(actuals):], lowers, uppers, alpha=0.2, color='red', label='Güven Aralığı')

                plt.title(f"{hisse} Son Günler ve 5 İş Günü Tahmini ({forecast_method.upper()})", fontsize=13, fontweight='bold')
                plt.xlabel('Tarih', fontsize=11)
                plt.ylabel('Fiyat (TL)', fontsize=11)
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.tight_layout()

                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=200, bbox_inches='tight')
                plt.close()
                buf.seek(0)
                chart_b64 = base64.b64encode(buf.read()).decode('utf-8')

                answer = (
                    f"🔮 {hisse} 5 İŞ GÜNÜ TAHMİNİ ({forecast_method.upper()}):\n\n" +
                    "\n".join([f"📊 {dates[len(actuals)+i].strftime('%d.%m.%Y')}: {preds[i]:.2f} TL" for i in range(len(preds))]) +
                    "\n⚠️ Bu tahminler sadece referans amaçlıdır!"
                    if language == 'tr'
                    else f"🔮 {hisse} 5 BUSINESS DAY FORECAST ({forecast_method.upper()}):\n\n" +
                    "\n".join([f"📊 {dates[len(actuals)+i].strftime('%d.%m.%Y')}: {preds[i]:.2f} TL" for i in range(len(preds))]) +
                    "\n⚠️ These predictions are for reference only!"
                )
                return {"answer": answer, "chart": chart_b64}
            else:
                # If there is any actual data, show it as a chart
                if forecast_data and forecast_data.get('actuals'):
                    dates = [datetime.fromtimestamp(ts) for ts in forecast_data['dates']]
                    actuals = forecast_data.get('actuals', [])
                    plt.figure(figsize=(8, 5))
                    plt.plot(dates[:len(actuals)], actuals, marker='o', color='blue', label='Son Günler Fiyatı')
                    plt.title(f"{hisse} Son Günler Fiyatı", fontsize=13, fontweight='bold')
                    plt.xlabel('Tarih', fontsize=11)
                    plt.ylabel('Fiyat (TL)', fontsize=11)
                    plt.grid(True, alpha=0.3)
                    plt.legend()
                    plt.tight_layout()
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', dpi=200, bbox_inches='tight')
                    plt.close()
                    buf.seek(0)
                    chart_b64 = base64.b64encode(buf.read()).decode('utf-8')
                    answer = f"❌ {hisse} için yeterli tahmin verisi yok, sadece son günler gösteriliyor." if language == 'tr' else f"❌ Not enough forecast data for {hisse}, showing only recent days."
                    return {"answer": answer, "chart": chart_b64}
                else:
                    answer = f"❌ {hisse} için tahmin yapılamadı. Hiç veri yok." if language == 'tr' else f"❌ Could not forecast {hisse}. No data available."
                    return {"answer": answer, "chart": None}

        except Exception as e:
            print(f"Error processing forecast request: {e}")
            err_msg = f"❌ {hisse} için tahmin oluşturulamadı." if language == 'tr' else f"❌ Could not create forecast for {hisse}."
            return {"answer": err_msg, "chart": None}
    try:
        import unicodedata
        def normalize_text(text):
            # Lowercase and remove accents for robust matching
            text = text.lower()
            text = unicodedata.normalize('NFD', text)
            text = ''.join([c for c in text if unicodedata.category(c) != 'Mn'])
            return text

        question_lower = normalize_text(question)
        print(f"Processing question: '{question_lower}'")
        
        # "Hangi şirketler mevcut" sorusu
        if any(word in question_lower for word in ['hangi şirket', 'mevcut', 'available', 'companies', 'stocks']):
            print("Detected 'available stocks' question")
            result = {"answer": get_available_stocks(language), "chart": None}
            print(f"Returning: {result}")
            return result

        if hisse and any(word in question_lower for word in ['otomatik strateji', 'bugünkü strateji', 'al/tut/sat önerisi', 'yatırım stratejisi','analist agent','analiz ajanı','al/tut/sat kararı',' al tut sat kararı',' al tut sat','al sat tut','al/tut/sat önerisi']):
            try:
                result = generate_auto_strategy(hisse, company_name=None, use_llm=True)
                answer = (
                    f"📈 {hisse} OTOMATİK STRATEJİ ÖNERİSİ:\n"
                    f"💰 Fiyat: {result['current_price']} TL (%{result['change_pct']:.2f})\n"
                    f"📊 RSI: {result['rsi']}, MACD Hist: {result['macd_hist']}\n"
                    f"📰 Sentiment: {result['news_sentiment_label']} \n"
                    f"🎯 Karar: {result['decision']}\n"
                    f"💡 Sebep: {result['rationale']}\n"
                )
                if result.get('llm_summary'):
                    answer += f"\n🤖 Açıklama: {result['llm_summary']}"
                return {"answer": answer, "chart": None}
            except Exception as e:
                print(f"Auto strategy error: {e}")
                return {"answer": "❌ Otomatik strateji hesaplanamadı.", "chart": None}


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
        
        # Hisse kodu veya kodları var mı? (ör: CCOLA, BIMAS, THYAO veya karşılaştırma)
        hisse_list = []
        # 1. Doğrudan hisse kodlarını ara (birden fazla olabilir)
        for code in BIST_STOCKS:
            if code.lower() in question_lower:
                hisse_list.append(code)
        # 2. Şirket ismi sözlüğünde ara (birden fazla olabilir)
        for company_name, stock_code in COMPANY_TO_CODE.items():
            if company_name in question_lower and stock_code not in hisse_list:
                hisse_list.append(stock_code)
        # 3. LLM ile bulmayı dene (tekli fallback)
        if not hisse_list:
            llm_code = get_stock_code_from_llm(question)
            if llm_code:
                hisse_list.append(llm_code)
        # Sadece ilk iki hisseyi al (karşılaştırma için)
        hisse_list = hisse_list[:2]
        hisse = hisse_list[0] if hisse_list else None
        
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
                    
                    plt.title(f"{hisse} Fiyat Tahmini ({forecast_method.upper()})", 
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
                        answer = f"🔮 {hisse} GÜNLÜK TAHMİN ({forecast_method.upper()}):\n\n"
                        answer += "📊 Borsada işlem görecek bir sonraki günün tahmini:\n"
                        for i, (date, pred) in enumerate(zip(last_5_dates, last_5_predictions), 1):
                            answer += f"   {date}: {pred:.2f} TL\n"
                        #answer += f"\n💡 Tahmin yöntemi: {forecast_method.upper()}\n"
                        answer += "⚠️ Bu tahminler sadece referans amaçlıdır!"
                    else:
                        answer = f"🔮 {hisse} DAY FORECAST ({forecast_method.upper()}):\n\n"
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
        
        # Grafik veya karşılaştırma sorusu mu? (ör: CCOLA grafik, CCOLA vs BIMAS grafik, CCOLA ile BIMAS karşılaştır)
        chart_keywords = [
            'grafik','çiz','grafiği','çizdir','grafiğini','chart', 'görsel',
            'karşılaştır', 'karşılaştırma', 'vs', 'ile'
        ]
        # Ayrıca büyük harfli ve normalize edilmiş varyantları da ekle
        chart_keywords += [k.upper() for k in chart_keywords]
        chart_keywords = list(set([normalize_text(k) for k in chart_keywords]))
        if hisse_list and any(word in question_lower for word in chart_keywords):
            print(f"Getting chart for {hisse_list}")
            try:
                days = 30  # Varsayılan
                if '1 ay' in question_lower or '1ay' in question_lower:
                    days = 30
                elif '3 ay' in question_lower or '3ay' in question_lower:
                    days = 90
                elif '6 ay' in question_lower or '6ay' in question_lower:
                    days = 180
                elif '1 yıl' in question_lower or '1yıl' in question_lower or '1 yil' in question_lower:
                    days = 365

                plt.figure(figsize=(12, 6))
                chart_found = False
                chart_labels = []
                for idx, hisse_kodu in enumerate(hisse_list):
                    chart_data = get_yfinance_chart(hisse_kodu, days=days)
                    if chart_data and chart_data.get('s') == 'ok':
                        dates = [datetime.fromtimestamp(ts) for ts in chart_data['t']]
                        prices = chart_data['c']
                        color = ['blue', 'red', 'green', 'orange', 'purple'][idx % 5]
                        plt.plot(dates, prices, linewidth=2, color=color, label=hisse_kodu)
                        chart_found = True
                        chart_labels.append(hisse_kodu)
                if chart_found:
                    plt.title(f"{' vs '.join(chart_labels)} Son {days} Günlük Fiyat Karşılaştırması", fontsize=14, fontweight='bold')
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
                    # Kullanıcıya örnek mesajlar ekle
                    example_hisse = chart_labels[0] if chart_labels else "hisse"
                    if language == 'tr':
                        answer = f"📊 {' ve '.join(chart_labels)} SON {days} GÜNLÜK FİYAT KARŞILAŞTIRMASI:\n\n"
                        answer += f"Bu grafikte {', '.join(chart_labels)} hisselerinin fiyat hareketleri karşılaştırılmıştır.\n\n"
                        answer += f"💡 Farklı dönemler için örnekler:\n"
                        answer += f"• {example_hisse} ile 1 aylık grafik\n"
                        answer += f"• {example_hisse} ile 3 aylık grafik\n"
                        answer += f"• {example_hisse} ile 6 aylık grafik\n"
                        answer += f"• {example_hisse} ile 1 yıllık grafik\n"
                        answer += f"\nBaşka bir hisse ile karşılaştırmak için: {example_hisse} ve BIMAS grafik\n"
                    else:
                        answer = f"📊 {' and '.join(chart_labels)} LAST {days} DAYS PRICE COMPARISON:\n\n"
                        answer += f"This chart compares the price movements of {', '.join(chart_labels)} stocks.\n\n"
                        answer += f"💡 For different periods, try:\n"
                        answer += f"• {example_hisse} 1 month chart\n"
                        answer += f"• {example_hisse} 3 months chart\n"
                        answer += f"• {example_hisse} 6 months chart\n"
                        answer += f"• {example_hisse} 1 year chart\n"
                        answer += f"\nTo compare with another stock: {example_hisse} and BIMAS chart\n"
                    return {"answer": answer, "chart": chart_b64}
                else:
                    if language == 'tr':
                        answer = f"❌ Grafik verisi bulunamadı."
                    else:
                        answer = f"❌ Chart data not found."
                    return {"answer": answer, "chart": None}
            except Exception as e:
                print(f"Error processing chart request: {e}")
                if language == 'tr':
                    return {"answer": f"❌ Grafik oluşturulamadı.", "chart": None}
                else:
                    return {"answer": f"❌ Could not create chart.", "chart": None}
        
        # Yatırım tavsiyesi sorusu mu? (ör: 1000 TL ne alayım, portföy önerisi)
        # Genişletilmiş anahtar kelime listesi, büyük harfli ve normalleştirilmiş varyantlar dahil
        portfolio_keywords = [
            'tavsiye', 'öneri', 'ne alayım', 'hangi hisseleri', 'portföy', 'yatırım',
            'advice', 'recommendation', 'portfolio',
            'ne alabilirim', 'ne alinir', 'ne alinabilir', 'hangi hisseleri alabilirim',
            'tl var ne alayim', 'tl var ne alabilirim', 'tl var hangi hisseleri alabilirim',
            'tl var hangi hisseleri alinir', 'tl var hangi hisseleri alinabilir',
            'tl var portföy', 'tl var portfoy', 'tl var portföy önerisi', 'tl var portfoy onerisi',
            'tl var', 'ne yapayim', 'hangi hisseler', 'hangi hisse', 'hangi hisseyi','öner',"tl ile nasıl yatırım yapabilirim","tl ile ne alabilirim"
        ]
        # Ayrıca, anahtar kelimelerin büyük harfli varyantlarını da ekle
        portfolio_keywords += [k.upper() for k in portfolio_keywords]
        # Normalize edilmiş anahtar kelimelerle karşılaştır
        portfolio_keywords = list(set([normalize_text(k) for k in portfolio_keywords]))
        if any(word in question_lower for word in portfolio_keywords):
            print("Getting investment advice")
            try:
                # Miktar belirleme (gelişmiş regex ve Türkçe/İngilizce yazımlar)
                amount = 1000  # Varsayılan
                # 1. Noktalı, virgüllü, boşluklu rakamlar: 1.000, 1,000, 1000, 10 000, 5 000
                match = re.search(r'(\d{1,3}(?:[.,\s]\d{3})+|\d{3,6})\s*tl', question_lower)
                if match:
                    raw = match.group(1)
                    raw = raw.replace('.', '').replace(',', '').replace(' ', '')
                    amount = int(raw)
                else:
                    # 2. 'bin', 'milyon' gibi Türkçe ifadeler
                    bin_match = re.search(r'(\d*)\s*bin\s*tl', question_lower)
                    if bin_match:
                        num = bin_match.group(1)
                        if num.strip() == '' or num.strip() == '1':
                            amount = 1000
                        else:
                            amount = int(num) * 1000
                    else:
                        # Sadece 'bin tl' geçiyorsa
                        if re.search(r'\bbin\s*tl\b', question_lower):
                            amount = 1000
                        # 'milyon' desteği (isteğe bağlı)
                        milyon_match = re.search(r'(\d*)\s*milyon\s*tl', question_lower)
                        if milyon_match:
                            num = milyon_match.group(1)
                            if num.strip() == '' or num.strip() == '1':
                                amount = 1_000_000
                            else:
                                amount = int(num) * 1_000_000
                
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

                    # O sektörden daha önce seçilmemiş hisse seç
                    already_selected = set([rec[0] for rec in recommendations])
                    available_stocks = [s for s in sector if s[0] not in already_selected]
                    if not available_stocks:
                        # Eğer o sektördeki tüm hisseler seçildiyse, sektördeki tüm hisselerden seç (tekrar olmaması için diğer sektörlere geçilecek)
                        available_stocks = [s for s in sector if s[0] not in already_selected]
                    if not available_stocks:
                        # Hala yoksa, tüm sektörlerdeki hisselerden seç (son çare, ama tekrar olmaması için)
                        all_stocks = [s for sec in sectors for s in sec if s[0] not in already_selected]
                        if not all_stocks:
                            break  # Tüm hisseler seçildi, çık
                        stock = random.choice(all_stocks)
                    else:
                        stock = random.choice(available_stocks)

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
                # LLM analizini, portföyde gerçekten önerilen hisselerle yap
                try:
                    # final_portfolio henüz oluşmadıysa recommendations ile devam et, ama mümkünse final_portfolio'yu kullan
                    temp_portfolio_syms = None
                    if 'final_portfolio' in locals() and final_portfolio:
                        temp_portfolio_syms = [rec[0] for rec in final_portfolio]
                    else:
                        temp_portfolio_syms = [rec[0] for rec in recommendations]
                    llm_prompt = f"Ben {amount} TL ile yatırım yapmak istiyorum. Risk profili: {risk_profile}. Önerilen hisseler: {temp_portfolio_syms}. Bu portföy hakkında kısa bir yorum yap."
                    llm_advice = ask_groq(llm_prompt)
                    if llm_advice and not llm_advice.startswith("API Error"):
                        llm_advice = f"\n🤖 Analiz: {llm_advice}"
                except:
                    llm_advice = ""
                
                # Önerileri formatla
                # Daha mantıklı ve dengeli dağılım algoritması (uygun hisse yoksa yerine başka hisse öner)
                price_cache = {}
                for symbol, name, ratio in recommendations:
                    yf_data = get_yfinance_quote(symbol)
                    if yf_data and yf_data.get('c', 0) > 0:
                        price_cache[symbol] = yf_data['c']
                    else:
                        price_cache[symbol] = 100  # fallback if price not found

                portfolio_size = len(recommendations)
                if portfolio_size == 0:
                    if language == 'tr':
                        return {"answer": "❗ Portföy oluşturulamadı.", "chart": None}
                    else:
                        return {"answer": "❗ Portfolio could not be created.", "chart": None}

                per_stock = amount // portfolio_size
                kalan = amount - (per_stock * portfolio_size)
                # Hisseleri fiyatına göre sırala (en ucuzdan pahalıya)
                sorted_recs = sorted(recommendations, key=lambda x: price_cache.get(x[0], 100))
                all_candidates = sorted(set([(s, n, price_cache[s]) for s, n, _ in recommendations]), key=lambda x: x[2])
                final_portfolio = []
                used_symbols = set()
                kalan_tutar = kalan
                for idx, (symbol, name, _) in enumerate(sorted_recs):
                    price = price_cache.get(symbol, 100)
                    invest = per_stock + (1 if idx < kalan_tutar else 0)
                    shares = int(invest // price)
                    used = shares * price
                    if shares < 1:
                        found = False
                        # Önce all_candidates listesinden dene
                        for cand_symbol, cand_name, cand_price in all_candidates:
                            if cand_symbol not in used_symbols and cand_price <= invest:
                                cand_shares = int(invest // cand_price)
                                cand_used = cand_shares * cand_price
                                if cand_shares > 0:
                                    final_portfolio.append((cand_symbol, cand_name, cand_used, cand_shares, cand_price, invest))
                                    used_symbols.add(cand_symbol)
                                    found = True
                                    break
                        # Eğer hala bulunamazsa, BIST_STOCKS listesinden dene
                        if not found:
                            for alt_symbol in BIST_STOCKS:
                                if alt_symbol not in used_symbols:
                                    # Fiyatı çek
                                    alt_price = price_cache.get(alt_symbol)
                                    if alt_price is None:
                                        yf_data = get_yfinance_quote(alt_symbol)
                                        if yf_data and yf_data.get('c', 0) > 0:
                                            alt_price = yf_data['c']
                                            price_cache[alt_symbol] = alt_price
                                        else:
                                            continue
                                    if alt_price <= invest:
                                        alt_shares = int(invest // alt_price)
                                        alt_used = alt_shares * alt_price
                                        if alt_shares > 0:
                                            # Hisse adı bul
                                            alt_name = alt_symbol
                                            for s, n, _ in recommendations:
                                                if s == alt_symbol:
                                                    alt_name = n
                                                    break
                                            final_portfolio.append((alt_symbol, alt_name, alt_used, alt_shares, alt_price, invest))
                                            used_symbols.add(alt_symbol)
                                            found = True
                                            break
                        # Hala bulunamazsa, slot boş kalsın (uyarı verilecek)
                        if not found:
                            final_portfolio.append((symbol, name, invest, 0, price, invest))
                            used_symbols.add(symbol)
                    else:
                        final_portfolio.append((symbol, name, used, shares, price, invest))
                        used_symbols.add(symbol)

                # Kalan tutarlarla tekrar en ucuzdan başlayarak hisse alınabiliyorsa ekle
                kalan_artan = amount - sum(x[2] for x in final_portfolio)
                idx = 0
                while kalan_artan > 0 and idx < len(final_portfolio):
                    symbol, name, used, shares, price, invest = final_portfolio[idx]
                    if price <= kalan_artan:
                        ek_hisse = int(kalan_artan // price)
                        if ek_hisse > 0:
                            final_portfolio[idx] = (symbol, name, used + ek_hisse * price, shares + ek_hisse, price, invest)
                            kalan_artan -= ek_hisse * price
                    idx += 1

                # LLM analizini, final_portfolio oluştuktan sonra ve sadece portföydeki hisselerle yap
                llm_advice = ""
                try:
                    temp_portfolio_syms = [rec[0] for rec in final_portfolio]
                    llm_prompt = f"Ben {amount} TL ile yatırım yapmak istiyorum. Risk profili: {risk_profile}. Önerilen hisseler: {temp_portfolio_syms}. Bu portföy hakkında kısa bir yorum yap."
                    llm_advice_raw = ask_groq(llm_prompt)
                    # Latin olmayan karakterleri temizle
                    if llm_advice_raw:
                        llm_advice_clean = remove_non_latin(llm_advice_raw)
                        if not llm_advice_clean.startswith("API Error"):
                            llm_advice = f"\n🤖 Analiz: {llm_advice_clean}"
                except:
                    llm_advice = ""

                # Eğer hiç hisse alınamıyorsa uyarı ver
                if all(shares == 0 for _, _, _, shares, _, _ in final_portfolio):
                    if language == 'tr':
                        answer = f"💼 {amount:,} TL İÇİN YATIRIM TAVSİYESİ:\n\n"
                        answer += f"📊 Risk Profili: {risk_profile.upper()}\n\n"
                        answer += "❗ Bu tutarla portföy oluşturulamıyor. Lütfen daha yüksek bir tutar girin.\n"
                        return {"answer": answer, "chart": None}
                    else:
                        answer = f"💼 INVESTMENT ADVICE FOR {amount:,} TL:\n\n"
                        answer += f"📊 Risk Profile: {risk_profile.upper()}\n\n"
                        answer += "❗ Cannot create a portfolio with this amount. Please enter a higher amount.\n"
                        return {"answer": answer, "chart": None}

                if language == 'tr':
                    answer = f"💼 {amount:,} TL İÇİN YATIRIM TAVSİYESİ:\n\n"
                    answer += f"📊 Risk Profili: {risk_profile.upper()}\n\n"
                    answer += "🎯 Önerilen Portföy:\n"
                    for symbol, name, used, shares, price, invest in final_portfolio:
                        if shares > 0:
                            answer += f"   • {symbol} ({name}): {used:,.0f} TL ({shares} hisse, 1 hisse ≈ {price:.2f} TL)\n"
                        else:
                            answer += f"   • {symbol} ({name}): {invest:,.0f} TL (Miktar yetersiz, 1 hisse alınamaz)\n"
                    answer += f"\n💰 Toplam Yatırım: {sum(x[2] for x in final_portfolio):,.0f} TL{llm_advice}\n"
                    answer += "⚠️ Bu tavsiyeler sadece referans amaçlıdır!\n"
                    example_amount = f"{amount:,}".replace(",", ".")
                    answer += f"\n⚠️ Risk profilinize göre yüksek ve düşük riskli opsiyonları da gösterebiliriz. \"{example_amount} tl var düşük riskle hangi hisseleri alabilirim?\" \n\"{example_amount} tl var yüksek riskli portföy oluşturur musun?\" gibi yazabilirsiniz.\n"
                else:
                    answer = f"💼 INVESTMENT ADVICE FOR {amount:,} TL:\n\n"
                    answer += f"📊 Risk Profile: {risk_profile.upper()}\n\n"
                    answer += "🎯 Recommended Portfolio:\n"
                    for symbol, name, used, shares, price, invest in final_portfolio:
                        if shares > 0:
                            answer += f"   • {symbol} ({name}): {used:,.0f} TL ({shares} shares, 1 share ≈ {price:.2f} TL)\n"
                        else:
                            answer += f"   • {symbol} ({name}): {invest:,.0f} TL (Insufficient for 1 share)\n"
                    answer += f"\n💰 Total Investment: {sum(x[2] for x in final_portfolio):,.0f} TL{llm_advice}\n"
                    answer += "⚠️ These recommendations are for reference only!\n"
                    answer += "🕐 Market hours: 10:00-18:00\n"
                    answer += f"\nWe can show you high and low risk options according to your risk profile. For example: 'What can I buy with {amount:,} TL with low risk?'\n"
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
                    answer += "🕐 *Gerçek zamanlı veri "
                else:
                    answer = f"🎯 {hisse} CURRENT PRICE INFO:\n\n"
                    answer += f"💰 Current Price: {current_price:.2f} TL\n"
                    answer += f"📈 Daily Change: {change:+.2f} TL ({change_percent:+.2f}%)\n"
                    answer += f"📊 Daily High: {high:.2f} TL\n"
                    answer += f"📉 Daily Low: {low:.2f} TL\n"
                    answer += f"📈 Volume: {volume:,} shares\n\n"
                    answer += "🕐 *Real-time data"
                
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
                    answer = f"📰 {hisse} HABER SENTIMENT ANALİZİ:\n\n"
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
                    
                    answer += "\n💡 Bu analiz son haberlere dayanmaktadır."
                else:
                    answer = f"📰 {hisse} NEWS SENTIMENT ANALYSIS:\n\n"
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
                    
                    answer += "\n💡 This analysis is based on news from the last days."
                
                return {"answer": answer, "chart": None}
            except Exception as e:
                print(f"Error processing sentiment analysis: {e}")
                if language == 'tr':
                    return {"answer": f"❌ {hisse} için sentiment analizi yapılamadı.", "chart": None}
                else:
                    return {"answer": f"❌ Could not perform sentiment analysis for {hisse}.", "chart": None}
        

        # Eğer hiçbir anahtar kelimeye uymuyorsa, önce finans/borsa ile ilgili olup olmadığını kontrol et
        finance_keywords = [
           'borsa', 'hisse', 'yatırım', 'finans', 'şirket', 'portföy', 'endeks', 'dolar', 'altın', 'kripto', 'bitcoin',
            'usd', 'eur', 'euro', 'doviz', 'döviz', 'faiz', 'tahvil', 'fon', 'viop', 'vadeli', 'borsada', 'borsacı',
            'strateji', 'parite', 'usdtry', 'eurtry', 'usd/tl', 'eur/tl', 'trader', 'trading', 'analiz', 
            'teknik analiz', 'temel analiz', 'grafik', 'fiyat', 'haber', 'borsa haberi', 'borsa analizi',
            'yatırımcı', 'yatırım tavsiyesi', 'sermaye', 'kar', 'zarar', 'temettü', 'bedelsiz', 'hisse senedi',
            'bilanço', 'gelir tablosu', 'finansal rapor', 'piyasa değeri', 'arz', 'talep', 'kapanış', 'açılış',
            'alım satım', 'işlem hacmi', 'emir', 'destek', 'direnç', 'stop loss', 'kaldıraç', 'marjin', 'volatilite',
            'borsa istanbul', 'bist', 'bist100', 'bist30', 'endeks fonu', 'yatırım fonu', 'borsa fonu', 'etf',
        ]
        finance_keywords += [k.upper() for k in finance_keywords]
        finance_keywords = list(set([normalize_text(k) for k in finance_keywords]))
        if not any(word in question_lower for word in finance_keywords):
            # Finans/borsa ile alakalı değilse profesyonel cevap ver
            if language == 'tr':
                answer = "❗ Bu asistan sadece finans, borsa ve yatırım ile ilgili soruları yanıtlar. Diğer konular için lütfen finbotdestek@gmail.com adresine yazabilirsiniz."
            else:
                answer = "❗ This assistant only answers questions about finance, stocks, and investment. For other topics, please contact finbotdestek@gmail.com."
            return {"answer": answer, "chart": None}

        # Eğer finans/borsa ile ilgiliyse, LLM'e sor
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
                answer += "💡 Sentiment analizi: Son haberler\n"
                answer += "🤖 Otomatik Strateji: 'ARCLK bugünkü strateji' veya 'ASELS al/tut/sat'\n"
                
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
                answer += "💡 Sentiment analysis: News from last days"
        
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

# Auto Strategy Agent 
# Usage: copy the functions and endpoint into app.py (same level as other endpoints)
# This file assumes `app`, `get_finnhub_quote`, `get_yfinance_quote`, `get_turkish_stock_news_by_company`,
# and `ask_groq` already exist in app.py . It uses yfinance for historical data.



def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
    """Calculates the Relative Strength Index (RSI) for a price series.
    Returns the last RSI value (float)."""
    if prices is None or len(prices) < period + 1:
        return None
    delta = prices.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)

    # Use Wilder smoothing
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()

    rs = roll_up / roll_down
    rsi = 100 - (100 / (1 + rs))

    # Return last RSI value as float
    try:
        last_rsi = float(rsi.iloc[-1])
    except Exception:
        last_rsi = None
    return last_rsi


def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> dict:
    """Calculates MACD, signal and histogram. Returns last values as dict.
    Keys: macd, signal, hist (floats)"""
    if prices is None or len(prices) < slow + signal:
        return {'macd': None, 'signal': None, 'hist': None}

    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - signal_line

    try:
        return {
            'macd': float(macd_line.iloc[-1]),
            'signal': float(signal_line.iloc[-1]),
            'hist': float(macd_hist.iloc[-1])
        }
    except Exception:
        return {'macd': None, 'signal': None, 'hist': None}


def get_historical_close_prices(symbol: str, days: int = 120) -> pd.Series:
    """Fetch historical close prices for BIST ticker (symbol like 'THYAO' or 'CCOLA').
    Returns a pandas Series indexed by datetime with Close prices or raises an error."""
    try:
        yf_symbol = f"{symbol}.IS"
        ticker = yf.Ticker(yf_symbol)
        # Use at least `days` calendar days; if market closed some days will be missing
        hist = ticker.history(period=f"{days}d")
        if hist.empty:
            return pd.Series(dtype=float)
        return hist['Close'].dropna()
    except Exception as e:
        print(f"get_historical_close_prices error for {symbol}: {e}")
        return pd.Series(dtype=float)


def generate_auto_strategy(symbol: str, company_name: str = None, days: int = 120, use_llm: bool = True) -> dict:
    """Generates an automatic AL/TUT/SAT recommendation for given BIST symbol.

    Returns a dict containing: symbol, price, price_change_pct, rsi, macd, news_sentiment,
    decision (AL/TUT/SAT), rationale (text), details (raw values).
    """
    # 1) Current quote
    quote = get_finnhub_quote(symbol)
    if not quote:
        raise ValueError(f"Gerçek zamanlı fiyat alınamadı: {symbol}")

    current_price = float(quote.get('c', 0) or 0)
    change = float(quote.get('d', 0) or 0)
    change_pct = float(quote.get('dp', 0) or 0)

    # 2) Historical prices -> RSI & MACD
    closes = get_historical_close_prices(symbol, days=days)
    rsi = calculate_rsi(closes, period=14)
    macd_data = calculate_macd(closes)

    # 3) News sentiment
    sentiment_info = {'sentiment_score': 0.0, 'sentiment_label': 'Nötr', 'confidence': 0.0}
    try:
        # company_name fallback to symbol if not provided
        comp = company_name if company_name else symbol
        sentiment_info = get_news_sentiment(comp, symbol)
    except Exception as e:
        print(f"Haber sentiment alınamadı: {e}")

    sentiment_score = float(sentiment_info.get('sentiment_score', 0.0) or 0.0)
    sentiment_label = sentiment_info.get('sentiment_label', 'Nötr')
    sentiment_conf = float(sentiment_info.get('confidence', 0.0) or 0.0)

    # 4) Simple rule-based decision
    # Prioritize extreme RSI and sentiment signals, then MACD histogram direction, then price momentum
    decision = 'TUT'
    reasons = []

    # RSI rules
    if rsi is not None:
        if rsi < 30:
            reasons.append(f"RSI düşük ({rsi:.1f}) → potansiyel aşırı satım")
        elif rsi > 70:
            reasons.append(f"RSI yüksek ({rsi:.1f}) → potansiyel aşırı alım")

    # Sentiment rules
    if sentiment_score > 0.15:
        reasons.append(f"Haberler olumlu (score={sentiment_score:.4f})")
    elif sentiment_score < -0.15:
        reasons.append(f"Haberler olumsuz (score={sentiment_score:.4f})")

    # MACD momentum
    macd_hist = macd_data.get('hist')
    if macd_hist is not None:
        if macd_hist > 0:
            reasons.append(f"MACD histogram pozitif ({macd_hist:.4f}) → yükseliş momentumu")
        elif macd_hist < 0:
            reasons.append(f"MACD histogram negatif ({macd_hist:.4f}) → düşüş momentumu")

    # Decision combining rules (simple priority logic)
    # 1) Strong buy signals
    if (rsi is not None and rsi < 30 and sentiment_score > 0.1) or (macd_hist is not None and macd_hist > 0 and sentiment_score > 0.2):
        decision = 'AL'
    # 2) Strong sell signals
    elif (rsi is not None and rsi > 70 and sentiment_score < -0.1) or (macd_hist is not None and macd_hist < 0 and sentiment_score < -0.2):
        decision = 'SAT'
    else:
        # Fallback heuristics
        if rsi is not None and rsi < 35 and sentiment_score >= -0.05:
            decision = 'AL'
        elif rsi is not None and rsi > 65 and sentiment_score <= 0.05:
            decision = 'SAT'
        else:
            decision = 'TUT'

    # 5) Build rationale text (optionally refine with LLM)
    rationale = ' / '.join(reasons) if reasons else 'Belirgin teknik veya haber sinyali yok.'

    # Optionally ask LLM for a short summary (use_llm toggles this)
    llm_summary = None
    if use_llm:
        try:

                    # 🔹 Önce formatlı stringleri hazırla
            rsi_str = f"{rsi:.2f}" if rsi is not None else "N/A"
            macd_hist_str = f"{macd_hist:.4f}" if macd_hist is not None else "N/A"

            prompt = f"""
            Sen bir finans analisti olarak davran.  
            Aşağıdaki teknik veriler ve otomatik sistemin verdiği kararı dikkate alarak, yatırımcıya anlaşılır ve net bir şekilde bu kararı destekleyen nedenleri 3-4 cümleyle açıkla.  
    

    Hisse: {symbol}
    Son fiyat: {current_price:.2f} TL (günlük değişim %{change_pct:.2f})
    RSI: {rsi_str}
    MACD Histogram: {macd_hist_str}
    Haber Sentiment Skoru: ({sentiment_label})
    Karar: {decision}
    
    """
            llm_text = ask_groq(prompt)
            if isinstance(llm_text, str) and llm_text.strip():
                llm_summary = llm_text.strip()
        except Exception as e:
            print(f"LLM özetleme başarısız: {e}")

    result = {
        'symbol': symbol,
        'current_price': current_price,
        'change': change,
        'change_pct': change_pct,
        'rsi': None if rsi is None else round(float(rsi), 2),
        'macd': None if macd_data.get('macd') is None else round(float(macd_data.get('macd')), 6),
        'macd_signal': None if macd_data.get('signal') is None else round(float(macd_data.get('signal')), 6),
        'macd_hist': None if macd_data.get('hist') is None else round(float(macd_data.get('hist')), 6),
        'news_sentiment_score': round(sentiment_score, 4),
        'news_sentiment_label': sentiment_label,
        'news_sentiment_confidence': round(sentiment_conf, 3),
        'decision': decision,
        'rationale': rationale,
        'llm_summary': llm_summary,
        'details': {
            'reasons': reasons,
            'recent_news_sample': sentiment_info.get('recent_news', [])[:3] if isinstance(sentiment_info, dict) else [],
        }
    }

    return result


# FastAPI endpoint to expose the auto strategy
@app.get('/auto-strategy')
async def auto_strategy(symbol: str, company_name: str = None, use_llm: bool = True):
    """GET /auto-strategy?symbol=CCOLA&company_name=Coca%20Cola&use_llm=true

    Returns JSON with the auto strategy decision and details.
    """
    symbol = symbol.strip().upper()
    if not symbol:
        raise HTTPException(status_code=400, detail='symbol query param is required')

    try:
        result = generate_auto_strategy(symbol, company_name=company_name, use_llm=use_llm)
        return result
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        print(f"auto_strategy endpoint error: {e}")
        raise HTTPException(status_code=500, detail='Internal server error while generating strategy')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)