# 🧪 Test Komutları Rehberi

## 🔑 API Anahtarı Testi

```bash
# Backend'i başlat
python app.py

# API testi
curl http://localhost:8000/test-finnhub
```

## 📊 Hisse Testleri

```bash
# CCOLA güncel fiyat
Invoke-WebRequest -Uri "http://localhost:8000/ask" -Method POST -Body "question=CCOLA güncel fiyat" -ContentType "application/x-www-form-urlencoded"

# Mevcut hisseler
Invoke-WebRequest -Uri "http://localhost:8000/ask" -Method POST -Body "question=Hangi şirketler mevcut" -ContentType "application/x-www-form-urlencoded"

# CCOLA grafik (arşiv)
Invoke-WebRequest -Uri "http://localhost:8000/ask" -Method POST -Body "question=CCOLA için 2023 grafiği" -ContentType "application/x-www-form-urlencoded"
```

## 🔧 API Anahtarı Güncelleme

1. **Yeni API Anahtarı Alın**:
   - https://finnhub.io/ adresine gidin
   - "Get free API key" butonuna tıklayın
   - E-posta ile kayıt olun
   - API anahtarınızı kopyalayın

2. **Dosyayı Güncelleyin**:
   ```python
   # backend/app.py dosyasında
   FINNHUB_API_KEYS = [
       "YOUR_NEW_API_KEY_HERE",  # 🔑 YENİ API ANAHTARINIZI BURAYA YAZIN
       # ... diğer anahtarlar
   ]
   ```

3. **Test Edin**:
   ```bash
   # Backend'i yeniden başlatın
   python app.py
   
   # Test edin
   curl http://localhost:8000/test-finnhub
   ```

## 📋 Beklenen Sonuçlar

### ✅ Başarılı API Testi:
```json
{
  "success": true,
  "message": "Finnhub API test completed: Key 1 working with BIST:CCOLA"
}
```

### ❌ Başarısız API Testi:
```json
{
  "success": false,
  "message": "Finnhub API test completed: All keys failed"
}
```

## 🚨 Sorun Giderme

- **"All keys failed"**: Yeni API anahtarı gerekli
- **"Python bulunamad"**: Virtual environment aktifleştirin
- **"Connection refused"**: Backend çalışmıyor, `python app.py` ile başlatın 