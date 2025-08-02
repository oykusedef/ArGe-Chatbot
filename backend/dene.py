import requests
url = 'https://api.yapikredi.com.tr/api/stockmarket/v1/bistIndices'
try:
    r = requests.get(url, timeout=10)
    print(r.status_code)
    print(r.text)
except Exception as e:
    print("Hata:", e)