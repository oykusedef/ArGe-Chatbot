import React, { useState, useRef, useEffect } from 'react';
// import { Chart } from 'react-chartjs-2';

const Chatbot = () => {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState([
    { sender: 'bot', text: 'Merhaba! Ben FINBOT. Size nasıl yardımcı olabilirim? Hangi şirketler mevcut diye sorabilirsiniz.' }
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
      const formData = new FormData();
      formData.append('question', input);
      formData.append('language', language);

      console.log('Sending data:', { question: input, language: language });

      const response = await fetch('http://localhost:8000/ask', {
        method: 'POST',
        body: formData,
      });

      console.log('Response status:', response.status);
      console.log('Response headers:', response.headers);

      if (!response.ok) {
        const errorText = await response.text();
        console.error('Response error:', errorText);
        throw new Error(`HTTP ${response.status}: ${errorText}`);
      }

      const data = await response.json();
      console.log('Response data:', data);
      
      setMessages(prev => [...prev, { 
        sender: 'bot', 
        text: data.answer || data.error, 
        chart: data.chart || null 
      }]);
    } catch (error) {
      console.error('Fetch error:', error);
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
            <div style={styles.messageText}>{msg.text}</div>
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