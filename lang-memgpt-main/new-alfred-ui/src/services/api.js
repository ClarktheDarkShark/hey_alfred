// api.js

const API_URL = process.env.REACT_APP_API_URL || 'https://alfred-demo-311fd5c8f0bf.herokuapp.com';

/**
 * Ensures that the provided URL uses HTTPS.
 * If the URL starts with 'http://', it is replaced with 'https://'.
 *
 * @param {string} url - The URL to secure.
 * @returns {string} - The secure URL.
 */
const getSecureUrl = (url) => {
  if (url.startsWith('http://')) {
    return url.replace('http://', 'https://');
  }
  return url;
};

/**
 * Sends a chat message to the API and returns the response.
 *
 * @param {Array} messages - List of message objects with { role, content }.
 * @param {object} [configurable={}] - Optional additional configuration.
 * @returns {Promise<object>} - A promise that resolves to the JSON response.
 */
export const sendMessage = async (messages, configurable = {}) => {
  try {
    const secureApiUrl = getSecureUrl(API_URL);
    const response = await fetch(`${secureApiUrl}/api/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        messages: messages.map(msg => ({
          role: msg.role,
          content: msg.content,
        })),
        configurable: {
          user_id: 'default-user',
          model: 'gpt-4o',
          ...configurable,
        }
      }),
    });
    
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('API Error:', error);
    throw error;
  }
};

export default { sendMessage };
