import os
import pandas as pd
import requests
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langdetect import detect
import openai
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import io
import base64
import re

load_dotenv("api.env")

openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# Huggingface Mistral API ile yanıt üretme fonksiyonu
def ask_mistral(prompt: str) -> str:
    url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    payload = {"inputs": prompt}
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        result = response.json()
        if isinstance(result, list) and len(result) > 0 and 'generated_text' in result[0]:
            return result[0]['generated_text']
        elif isinstance(result, dict) and 'generated_text' in result:
            return result['generated_text']
        else:
            return str(result)
    else:
        return f"API Error: {response.status_code} - {response.text}"

# Archive klasöründen hisse verisi okuma fonksiyonu
def read_stock_csv(stock_code: str) -> pd.DataFrame:
    # Debug için çalışma dizinini yazdır
    print(f"Çalışma dizini: {os.getcwd()}")
    
    # Proje kök dizinini bul (backend klasörünün bir üstü)
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(backend_dir)
    file_path = os.path.join(project_root, 'archive', f'{stock_code}_MG.csv')
    
    print(f"Backend dizini: {backend_dir}")
    print(f"Proje kökü: {project_root}")
    print(f"Aranan dosya yolu: {file_path}")
    print(f"Dosya mevcut mu: {os.path.exists(file_path)}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} bulunamadı.")
    # CSV başlıksız, uygun isimleri ver
    columns = ["HISSE", "TARIH", "ACILIS", "YUKSEK", "DUSUK", "KAPANIS", "HACIM", "BOS"]
    df = pd.read_csv(file_path, header=None, names=columns)
    return df

@app.post("/ask")
async def ask_question(question: str = Form(...), stock_code: str = Form(...)):
    try:
        df = read_stock_csv(stock_code)
        chart_b64 = None
        # Grafik isteği var mı kontrol et
        year_match = re.search(r"(\d{4})[\s\S]*grafik|grafik[\s\S]*(\d{4})", question, re.IGNORECASE)
        if year_match:
            # Yıl bazlı grafik
            year = year_match.group(1) or year_match.group(2)
            year = int(year)
            if 'TARIH' in df.columns:
                df['TARIH'] = pd.to_datetime(df['TARIH'], errors='coerce')
                year_df = df[df['TARIH'].dt.year == year]
            else:
                year_df = df
            if not year_df.empty:
                plt.figure(figsize=(8, 4))
                plt.plot(year_df['TARIH'], year_df['KAPANIS'], label='Kapanış')
                plt.title(f"{stock_code} {year} Yılı Kapanış Fiyatları")
                plt.xlabel('Tarih')
                plt.ylabel('Kapanış')
                plt.legend()
                plt.tight_layout()
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                plt.close()
                buf.seek(0)
                chart_b64 = base64.b64encode(buf.read()).decode('utf-8')
        # Son 5 gün analizi
        last5 = df.tail(5)
        summary = last5.to_string(index=False)
        prompt = f"Kullanıcı sorusu: {question}\nSon 5 günün verisi:\n{summary}\nCevabını kısa ve net ver."
        answer = ask_mistral(prompt)
        return {"answer": answer, "chart": chart_b64}
    except Exception as e:
        return {"error": str(e)}

class ChatRequest(BaseModel):
    message: str
    language: str = None  # Optional, can be auto-detected

class ChatResponse(BaseModel):
    response: str
    language: str

# Simple prompt templates for support and recommendation
PROMPT_TEMPLATES = {
    'en': "You are a helpful assistant for a stock trading and product manufacturing platform. Answer customer support questions and recommend products as needed.\nUser: {message}\nAssistant:",
    'tr': "Bir stok ticareti ve ürün üretim platformu için yardımcı bir asistansın. Müşteri destek sorularını yanıtla ve gerektiğinde ürün öner.\nKullanıcı: {message}\nAsistan:"
}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(chat: ChatRequest):
    # Detect language if not provided
    if chat.language:
        lang = chat.language.lower()
    else:
        try:
            lang = detect(chat.message)
        except Exception:
            lang = 'en'
        if lang not in ['en', 'tr']:
            lang = 'en'
    prompt = PROMPT_TEMPLATES[lang].format(message=chat.message)
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": prompt}],
        max_tokens=256,
        temperature=0.7
    )
    answer = response.choices[0].message.content.strip()
    return ChatResponse(response=answer, language=lang)

@app.get("/")
def root():
    return {"message": "Chatbot backend is running."} 