import { useState } from 'react';
import { sendMessage as apiSendMessage } from '../services/api';

export const useChat = () => {
  const [messages, setMessages] = useState([]);

  const sendMessage = async (content) => {
    try {
      const userMessage = { role: 'user', content }; // Using 'user' role consistently
      setMessages(prev => [...prev, userMessage]);

      const allMessages = [...messages, userMessage];
      const data = await apiSendMessage(allMessages);
      
      setMessages(prev => [...prev, { role: 'assistant', content: data.response }]);
      return data;
    } catch (error) {
      console.error('Chat error:', error);
      setMessages(prev => [...prev, {
        role: 'system',
        content: 'Sorry, there was an error processing your message.',
      }]);
    }
  };

  return {
    messages,
    sendMessage,
    setMessages,
  };
};
