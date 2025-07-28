import React, { useState, useRef, useEffect } from 'react';

const LANGUAGES = [
  { code: 'en', label: 'English' },
  { code: 'tr', label: 'Türkçe' }
];

const Chatbot = () => {
  const [messages, setMessages] = useState([
    { sender: 'bot', text: 'Hello! How can I help you? (Merhaba! Size nasıl yardımcı olabilirim?)' }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [language, setLanguage] = useState('en');
  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;
    const userMsg = { sender: 'user', text: input };
    setMessages((msgs) => [...msgs, userMsg]);
    setInput('');
    setLoading(true);
    try {
      const res = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: input, language })
      });
      const data = await res.json();
      setMessages((msgs) => [...msgs, { sender: 'bot', text: data.response }]);
    } catch (err) {
      setMessages((msgs) => [...msgs, { sender: 'bot', text: 'Error: Could not reach server.' }]);
    }
    setLoading(false);
  };

  const handleLanguageChange = (e) => {
    setLanguage(e.target.value);
    setMessages([
      { sender: 'bot', text: e.target.value === 'en' ? 'Hello! How can I help you?' : 'Merhaba! Size nasıl yardımcı olabilirim?' }
    ]);
  };

  return (
    <div style={styles.container}>
      <div style={styles.header}>ArGe Chatbot</div>
      <div style={styles.langSelect}>
        <label htmlFor="lang">Language: </label>
        <select id="lang" value={language} onChange={handleLanguageChange}>
          {LANGUAGES.map((lang) => (
            <option key={lang.code} value={lang.code}>{lang.label}</option>
          ))}
        </select>
      </div>
      <div style={styles.chatArea}>
        {messages.map((msg, i) => (
          <div key={i} style={msg.sender === 'user' ? styles.userMsg : styles.botMsg}>
            {msg.text}
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>
      <form style={styles.inputArea} onSubmit={handleSend}>
        <input
          style={styles.input}
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={language === 'en' ? 'Type your message...' : 'Mesajınızı yazın...'}
          disabled={loading}
        />
        <button style={styles.button} type="submit" disabled={loading || !input.trim()}>
          {loading ? '...' : (language === 'en' ? 'Send' : 'Gönder')}
        </button>
      </form>
    </div>
  );
};

const styles = {
  container: {
    maxWidth: 400,
    margin: '40px auto',
    background: '#fff',
    borderRadius: 12,
    boxShadow: '0 2px 16px rgba(0,0,0,0.12)',
    display: 'flex',
    flexDirection: 'column',
    fontFamily: 'Segoe UI, sans-serif',
    minHeight: 500,
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
  },
  chatArea: {
    flex: 1,
    padding: 16,
    overflowY: 'auto',
    background: '#f9f9f9',
    display: 'flex',
    flexDirection: 'column',
    gap: 8,
  },
  userMsg: {
    alignSelf: 'flex-end',
    background: '#1976d2',
    color: '#fff',
    padding: '8px 14px',
    borderRadius: '16px 16px 0 16px',
    maxWidth: '80%',
    fontSize: 15,
  },
  botMsg: {
    alignSelf: 'flex-start',
    background: '#e3eafc',
    color: '#222',
    padding: '8px 14px',
    borderRadius: '16px 16px 16px 0',
    maxWidth: '80%',
    fontSize: 15,
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
  },
  button: {
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