// useChat.js
import { useState, useRef } from 'react';
import { sendMessage as apiSendMessage } from '../services/api';

export const useChat = () => {
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const messagesRef = useRef([]); // Store the latest messages

  // Helper to update both ref and state
  const updateMessages = (newMessages) => {
    messagesRef.current = newMessages;
    setMessages(newMessages);
  };

  const sendMessage = async (content) => {
    try {
      // Create user message and update messages state
      const userMessage = { role: 'user', content };
      const updatedMessages = [...messagesRef.current, userMessage];
      updateMessages(updatedMessages);
      setIsLoading(true);

      // Use the most recent messages for the API call
      const data = await apiSendMessage(updatedMessages);

      // Append the assistant response
      const newMessages = [...messagesRef.current, { role: 'assistant', content: data.response }];
      updateMessages(newMessages);
      return data;
    } catch (error) {
      console.error('Chat error:', error);
      updateMessages([
        ...messagesRef.current,
        { role: 'system', content: 'Sorry, there was an error processing your message.' }
      ]);
      throw error;
    } finally {
      setIsLoading(false);
    }
  };

  return {
    messages,
    sendMessage,
    setMessages: updateMessages,
    isLoading,
  };
};
