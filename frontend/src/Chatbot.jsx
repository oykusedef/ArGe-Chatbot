import React, { useState, useRef, useEffect } from 'react';

// URL'leri tıklanabilir link haline getiren fonksiyon
const formatMessageWithLinks = (text) => {
  if (!text) return text;
  
  // URL'leri tespit edip JSX elementlerine dönüştür
  const urlRegex = /(https?:\/\/[^\s]+)/g;
  const parts = text.split(urlRegex);
  
  return parts.map((part, index) => {
    if (urlRegex.test(part)) {
      return (
        <a 
          key={index}
          href={part} 
          target="_blank" 
          rel="noopener noreferrer"
          style={{ color: '#1976d2', textDecoration: 'underline' }}
        >
          {part}
        </a>
      );
    }
    return part;
  });
};

// Yardımcı: Haber mesajını parse eden fonksiyon
function parseNewsList(text) {
  // Her haber satırı '1.', '2.', ... ile başlar
  const newsRegex = /\n(\d+)\. /g;
  const parts = text.split(newsRegex);
  // İlk parça başlık, sonra [num, haber, num, haber, ...]
  let news = [];
  for (let i = 2; i < parts.length; i += 2) {
    news.push(parts[i]);
  }
  return news;
}

const Chatbot = () => {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState([
    { sender: 'bot', text: `Merhaba! Ben FINBOT.

🤖 FINBOT Size Nasıl Yardımcı Olabilir? Hangi şirketler mevcut diye sorabilirsiniz.

📈 Güncel fiyat: 'ARCLK güncel fiyat'
📊 Grafik: 'ARCLK grafik' veya 'ARCLK 3 ay grafik'
🔮 Tahmin: 'ARCLK tahmin' veya 'ARCLK forecast'
📰 Sentiment: 'ASELS medya analizi' veya 'ASELS haber analizi'
💼 Yatırım tavsiyesi: '1000 TL ne alayım' veya 'portföy önerisi'
📋 Hisse listesi: 'Hangi şirketler mevcut'
💡 Grafik süreleri: 1 ay, 3 ay, 6 ay, 1 yıl
💡 Tahmin yöntemleri: Prophet, ARIMA, LSTM
💡 Risk profilleri: Düşük, Orta, Yüksek
📰 Sentiment analizi: Son 7 günün haberleri` }
  ]);
  const [loading, setLoading] = useState(false);
  const [language, setLanguage] = useState('tr');

  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMessage = { sender: 'user', text: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      // Türk borsa haberleri için özel kontrol
      const inputLower = input.toLowerCase();
      if (inputLower.includes('türk borsa haberleri') || 
          inputLower.includes('güncel haberler') || 
          inputLower.includes('borsa haberleri') ||
          inputLower.includes('turkish news') ||
          inputLower.includes('market news')) {
        
        const response = await fetch('http://localhost:8000/turkish-news');
        const data = await response.json();
        
        if (data.success && data.news.length > 0) {
          let newsText = language === 'tr' ? 
            `📰 GÜNCEL TÜRK BORSA HABERLERİ (${data.total} haber):\n\n` :
            `📰 CURRENT TURKISH MARKET NEWS (${data.total} articles):\n\n`;
          
          data.news.slice(0, 5).forEach((article, index) => {
            newsText += `${index + 1}. ${article.title}\n`;
            newsText += `   📰 ${article.source} | ${article.published_at?.slice(0, 10) || 'Tarih bilgisi yok'}\n`;
            if (article.description) {
              newsText += `   📝 ${article.description.slice(0, 100)}...\n`;
            }
            newsText += '\n';
          });
          
          setMessages(prev => [...prev, { 
            sender: 'bot', 
            text: newsText, 
            chart: null 
          }]);
        } else {
          setMessages(prev => [...prev, { 
            sender: 'bot', 
            text: language === 'tr' ? 
              '❌ Haber bulunamadı. Lütfen daha sonra tekrar deneyin.' :
              '❌ No news found. Please try again later.', 
            chart: null 
          }]);
        }
      } else {
        // Normal soru-cevap işlemi
        const formData = new FormData();
        formData.append('question', input);
        formData.append('language', language);

        const response = await fetch('http://localhost:8000/ask', {
          method: 'POST',
          body: formData,
        });

        if (!response.ok) {
          const errorText = await response.text();
          console.error('Response error:', errorText);
          throw new Error(`HTTP ${response.status}: ${errorText}`);
        }

        const data = await response.json();
        
        setMessages(prev => [...prev, { 
          sender: 'bot', 
          text: data.answer || data.error, 
          chart: data.chart || null 
        }]);
      }
    } catch (error) {
      setMessages(prev => [...prev, { 
        sender: 'bot', 
        text: `Bir hata oluştu: ${error.message}`, 
        chart: null 
      }]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // Bot mesajı haber listesi ise özel render
  function renderBotMessage(msg, index) {
    // Haber mesajı mı?
    if (msg.text && /\n1\. /.test(msg.text) && /HABERLERİ|NEWS/.test(msg.text)) {
      const newsList = parseNewsList(msg.text);
      return (
        <div>
          <div>{msg.text.split('\n1. ')[0]}</div>
          <ul style={{paddingLeft: 18}}>
            {newsList.map((n, i) => <li key={i} style={{marginBottom: 8}}>{formatMessageWithLinks(n)}</li>)}
          </ul>
        </div>
      );
    }
    // Normal bot mesajı
    return formatMessageWithLinks(msg.text);
  }

  return (
    <div style={styles.container}>
      <div style={styles.header}>FINBOT</div>
      <div style={styles.langSelect}>
        <label htmlFor="lang">Language: </label>
        <select
          id="lang"
          name="language"
          value={language}
          onChange={(e) => setLanguage(e.target.value)}
          style={styles.select}
        >
          <option value="tr">Türkçe</option>
          <option value="en">English</option>
        </select>
      </div>
      <div style={styles.chatArea}>
        {messages.map((msg, index) => (
          <div key={index} style={msg.sender === 'user' ? styles.userMessage : styles.botMessage}>
            <div style={styles.messageText}>
              {msg.sender === 'bot' ? renderBotMessage(msg, index) : msg.text}
            </div>
            {msg.chart && (
              <img 
                src={`data:image/png;base64,${msg.chart}`} 
                alt="Chart" 
                style={styles.chartImage} 
              />
            )}
          </div>
        ))}
        {loading && (
          <div style={styles.botMessage}>
            <div style={styles.messageText}>Düşünüyorum...</div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>
      <div style={styles.inputArea}>
        <input
          type="text"
          id="messageInput"
          name="message"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Mesajınızı yazın..."
          style={styles.input}
          disabled={loading}
        />
        <button 
          onClick={handleSend} 
          style={styles.sendButton}
          disabled={loading || !input.trim()}
        >
          Gönder
        </button>
      </div>
    </div>
  );
};

const styles = {
  container: {
    maxWidth: 600,
    margin: '40px auto',
    background: '#fff',
    borderRadius: 12,
    boxShadow: '0 2px 16px rgba(0,0,0,0.12)',
    display: 'flex',
    flexDirection: 'column',
    fontFamily: 'Segoe UI, sans-serif',
    minHeight: 600,
    overflow: 'hidden',
  },
  header: {
    background: '#1976d2',
    color: '#fff',
    padding: '16px',
    fontSize: 22,
    fontWeight: 600,
    textAlign: 'center',
  },
  langSelect: {
    padding: '8px 16px',
    background: '#f5f5f5',
    borderBottom: '1px solid #eee',
    display: 'flex',
    gap: 10,
    alignItems: 'center',
  },
  select: {
    padding: '8px 12px',
    borderRadius: 8,
    border: '1px solid #ccc',
    fontSize: 15,
    cursor: 'pointer',
  },
  chatArea: {
    flex: 1,
    padding: 16,
    overflowY: 'auto',
    maxHeight: 400,
    background: '#f9f9f9',
    display: 'flex',
    flexDirection: 'column',
    gap: 8,
  },
  userMessage: {
    alignSelf: 'flex-end',
    background: '#1976d2',
    color: '#fff',
    padding: '8px 14px',
    borderRadius: '16px 16px 0 16px',
    maxWidth: '80%',
    fontSize: 15,
    wordWrap: 'break-word',
  },
  botMessage: {
    alignSelf: 'flex-start',
    background: '#e3eafc',
    color: '#222',
    padding: '8px 14px',
    borderRadius: '16px 16px 16px 0',
    maxWidth: '80%',
    fontSize: 15,
    wordWrap: 'break-word',
  },
  messageText: {
    marginBottom: '5px',
    whiteSpace: 'pre-wrap',
  },
  chartImage: {
    maxWidth: '100%',
    height: 'auto',
    borderRadius: '5px',
    marginTop: '5px',
  },
  inputArea: {
    display: 'flex',
    borderTop: '1px solid #eee',
    padding: 12,
    background: '#fafafa',
  },
  input: {
    flex: 1,
    padding: 10,
    borderRadius: 8,
    border: '1px solid #ccc',
    fontSize: 15,
    marginRight: 8,
    outline: 'none',
  },
  sendButton: {
    padding: '0 18px',
    borderRadius: 8,
    border: 'none',
    background: '#1976d2',
    color: '#fff',
    fontWeight: 600,
    fontSize: 15,
    cursor: 'pointer',
  },
};

export default Chatbot; 