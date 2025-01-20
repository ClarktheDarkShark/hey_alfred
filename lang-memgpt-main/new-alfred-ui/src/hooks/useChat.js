import { useState } from 'react';
import { sendMessage as apiSendMessage } from '../services/api';

export const useChat = () => {
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);

  const sendMessage = async (content) => {
    try {
      const userMessage = { role: 'user', content };
      const allMessages = [...messages, userMessage];
      
      // Update messages with user input immediately
      setMessages(allMessages);
      setIsLoading(true);
      
      // Call the backend
      const data = await apiSendMessage(allMessages);
      // The front-end expects data.response
      setMessages(prev => [...prev, { role: 'assistant', content: data.response }]);
      return data;
    } catch (error) {
      console.error('Chat error:', error);
      setMessages(prev => [...prev, {
        role: 'system',
        content: 'Sorry, there was an error processing your message.',
      }]);
      throw error;
    } finally {
      setIsLoading(false);
    }
  };

  return {
    messages,
    sendMessage,
    setMessages,
    isLoading,
  };
};
