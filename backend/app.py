import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langdetect import detect
import openai
from dotenv import load_dotenv

load_dotenv()

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