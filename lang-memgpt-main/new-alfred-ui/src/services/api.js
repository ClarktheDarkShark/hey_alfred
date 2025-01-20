const API_URL = process.env.REACT_APP_API_URL || 'https://alfred-demo-311fd5c8f0bf.herokuapp.com';

// Ensure URL always uses HTTPS
const getSecureUrl = (url) => {
  return url.replace('http://', 'https://');
};

export const sendMessage = async (messages, configurable = {}) => {
  try {
    const response = await fetch(getSecureUrl(`${API_URL}/api/chat`), {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        messages: messages.map(msg => ({
          role: msg.role,
          content: msg.content
        })),
        configurable: {
          user_id: 'default-user',
          model: 'gpt-4o',
          ...configurable
        }
      }),
    });
    
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
    }
    
    return await response.json(); // Must contain { response: "..." }
  } catch (error) {
    console.error('API Error:', error);
    throw error;
  }
};
