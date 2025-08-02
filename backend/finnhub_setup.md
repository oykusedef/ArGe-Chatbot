# Finnhub API Kurulum Rehberi

## 🔑 API Anahtarı Alma

1. **Finnhub'a Kayıt Olun**: https://finnhub.io/
2. **Ücretsiz Plan Seçin**: Günlük 60 API çağrısı
3. **API Anahtarınızı Alın**: Dashboard'dan kopyalayın

## 📝 API Anahtarını Güncelleme

`backend/app.py` dosyasında şu satırı güncelleyin:

```python
FINNHUB_API_KEYS = [
    "YOUR_API_KEY_HERE",  # Kendi API anahtarınızı buraya yazın
    "cn8v9v9r01qns40d15hgd253na9r01qns40d15i0",  # Yedek 1
    "cn8v9v9r01qns40d15hgd253na9r01qns40d15i1"   # Yedek 2
]
```

## 🧪 API Test Etme

```bash
# Backend klasöründe
python app.py

# Başka terminal'de
curl http://localhost:8000/test-finnhub
```

## 📊 Mevcut Hisse Kodları

- AKBNK, ASELS, BIMAS, CCOLA, EKGYO, ENKAI, EUPWR, FROTO
- GARAN, GUBRF, HEKTS, ISCTR, KCHOL, KRDMD, KOZAA, KOZAL
- MGROS, PGSUS, SAHOL, SASA, SISE, TCELL, THYAO, TKFEN
- TOASO, TUPRS, VAKBN, YKBNK, EREGL, SODA, PETKM

## ⚠️ Önemli Notlar

- **ARCLK**: Bu hisse kodu BIST'te mevcut değil
- **Borsa Saatleri**: 09:00-18:00 (Türkiye saati)
- **API Limitleri**: Ücretsiz plan günlük 60 çağrı
- **Hisse Formatı**: BIST:CCOLA şeklinde kullanılır

## 🚀 Test Komutları

```bash
# CCOLA güncel fiyat
curl -X POST "http://localhost:8000/ask" -d "question=CCOLA güncel fiyat"

# Mevcut hisseler
curl -X POST "http://localhost:8000/ask" -d "question=Hangi şirketler mevcut"

# THYAO grafik
curl -X POST "http://localhost:8000/ask" -d "question=THYAO grafik"
``` 