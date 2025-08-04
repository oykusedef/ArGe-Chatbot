#!/usr/bin/env python3
"""
Türk borsa haberleri API test dosyası
Bu dosya The News API entegrasyonunu test etmek için kullanılır.
"""

import http.client
import urllib.parse
import json

def test_turkish_stock_news():
    """The News API'den Türkçe borsa haberlerini test et"""
    try:
        print("🔍 Türk borsa haberleri test ediliyor...")
        
        conn = http.client.HTTPSConnection('api.thenewsapi.com')
        
        params = urllib.parse.urlencode({
            'api_token': 'PPIqVOjLOi5W7Jsk3cXjUHmMJwRFDA9DLXlYhejX',
            'categories': 'business,tech',
            'limit': 3,  # Ücretsiz plan limiti
            'language': 'tr',
            'locale': 'tr',
            'search': 'borsa'
        })
        
        print(f"📡 API isteği gönderiliyor: /v1/news/all?{params}")
        
        conn.request('GET', '/v1/news/all?{}'.format(params))
        res = conn.getresponse()
        data = res.read()
        
        print(f"📊 API yanıt kodu: {res.status}")
        
        # JSON verisini parse et
        news_data = json.loads(data.decode('utf-8'))
        
        print(f"📰 API yanıt durumu: {news_data.get('status', 'Bilinmeyen')}")
        
        if news_data.get('data') and len(news_data.get('data', [])) > 0:
            articles = news_data.get('data', [])
            print(f"✅ Başarılı! {len(articles)} haber bulundu.\n")
            
            print("📋 BULUNAN HABERLER:")
            print("=" * 50)
            
            for i, article in enumerate(articles[:5], 1):  # İlk 5 haberi göster
                print(f"\n{i}. {article.get('title', 'Başlık yok')}")
                print(f"   📰 Kaynak: {article.get('source', 'Bilinmeyen')}")
                print(f"   📅 Tarih: {article.get('published_at', 'Tarih bilgisi yok')}")
                if article.get('description'):
                    print(f"   📝 Açıklama: {article['description'][:100]}...")
                print(f"   🔗 URL: {article.get('url', 'URL yok')}")
            
            return True
        else:
            print(f"❌ API yanıtı başarısız: {news_data}")
            return False
            
    except Exception as e:
        print(f"❌ Hata oluştu: {e}")
        return False

def test_company_specific_news():
    """Belirli şirket için haber test et"""
    try:
        print("\n🔍 Şirket özel haberleri test ediliyor (THYAO)...")
        
        conn = http.client.HTTPSConnection('api.thenewsapi.com')
        
        params = urllib.parse.urlencode({
            'api_token': 'PPIqVOjLOi5W7Jsk3cXjUHmMJwRFDA9DLXlYhejX',
            'categories': 'business,tech',
            'limit': 3,  # Ücretsiz plan limiti
            'language': 'tr',
            'locale': 'tr',
            'search': 'THYAO'
        })
        
        conn.request('GET', '/v1/news/all?{}'.format(params))
        res = conn.getresponse()
        data = res.read()
        
        news_data = json.loads(data.decode('utf-8'))
        
        if news_data.get('data') and len(news_data.get('data', [])) > 0:
            articles = news_data.get('data', [])
            print(f"✅ THYAO için {len(articles)} haber bulundu.\n")
            
            for i, article in enumerate(articles, 1):
                print(f"{i}. {article.get('title', 'Başlık yok')}")
                print(f"   📰 {article.get('source', 'Bilinmeyen')}")
            
            return True
        else:
            print(f"❌ THYAO haberleri bulunamadı: {news_data}")
            return False
            
    except Exception as e:
        print(f"❌ Şirket haberleri test hatası: {e}")
        return False

if __name__ == "__main__":
    print("🚀 TÜRK BORSA HABERLERİ API TESTİ")
    print("=" * 50)
    
    # Genel borsa haberleri testi
    success1 = test_turkish_stock_news()
    
    # Şirket özel haberleri testi
    success2 = test_company_specific_news()
    
    print("\n" + "=" * 50)
    if success1 and success2:
        print("✅ TÜM TESTLER BAŞARILI!")
        print("🎉 API entegrasyonu çalışıyor.")
    else:
        print("❌ BAZI TESTLER BAŞARISIZ!")
        print("🔧 API anahtarını veya bağlantıyı kontrol edin.")
    
    print("\n💡 Kullanım örnekleri:")
    print("• 'Türk borsa haberleri'")
    print("• 'Güncel haberler'")
    print("• 'THYAO haber analizi'")
    print("• 'ASELS sosyal medya'") 