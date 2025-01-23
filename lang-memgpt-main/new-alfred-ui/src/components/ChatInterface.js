// ChatInterface.js
import React, { useState, useRef, useEffect } from 'react';
import {
  Box,
  Container,
  Paper,
  TextField,
  IconButton,
  Typography,
  Drawer,
} from '@mui/material';
import { styled } from '@mui/material/styles';
import { Mic, Send, Upload } from '@mui/icons-material';
import MessageBubble from './MessageBubble';
import FileUpload from './FileUpload';
import { useChat } from '../hooks/useChat';
import { useSpeechRecognition } from '../hooks/useSpeechRecognition';
import LoadingMessage from './LoadingMessage';
import Snackbar from '@mui/material/Snackbar';
import Alert from '@mui/material/Alert';

const ChatContainer = styled(Box)(({ theme }) => ({
  height: '100vh',
  display: 'flex',
  flexDirection: 'column',
  padding: theme.spacing(2),
  backgroundColor: theme.palette.background.default,
  position: 'relative',
  width: '100%', // Ensures full width within Container
}));

const Header = styled(Box)(({ theme }) => ({
  padding: theme.spacing(1, 0),
  textAlign: 'center',
}));

const MessagesContainer = styled(Paper)(({ theme }) => ({
  flex: 1,
  overflowY: 'auto',
  padding: theme.spacing(2),
  display: 'flex',
  flexDirection: 'column',
  gap: theme.spacing(1),
  marginTop: theme.spacing(1),
  width: '100%', // Ensures full width
}));

const InputContainer = styled(Box)(({ theme }) => ({
  display: 'flex',
  gap: theme.spacing(1),
  padding: theme.spacing(2, 0),
  backgroundColor: theme.palette.background.paper,
  width: '100%', // Ensures full width
}));

function ChatInterface() {
  const { messages, sendMessage, setMessages, isLoading } = useChat();
  const { isListening, startListening, stopListening, transcript } = useSpeechRecognition();
  const [input, setInput] = useState('');
  const [isDrawerOpen, setIsDrawerOpen] = useState(false);
  const messagesEndRef = useRef(null);
  const [ackMessage, setAckMessage] = useState(null);

  const initialized = useRef(false); // Properly declare the ref

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    if (transcript && isListening) {
      setInput(prev => prev + ' ' + transcript);
    }
  }, [transcript, isListening]);

  useEffect(() => {
    if (!initialized.current && messages.length === 0) {
      setMessages([
        {
          role: 'system',
          content: `Hey, I'm Alfred. I can \n- Fetch TAF (Terminal Aerodrome Forecast) data\n- Provide insights on provided documents,\n- And much more...\nHow can I help you?`,
        },
      ]);
      initialized.current = true; // Mark as initialized
    }
  }, [messages, setMessages]); // Include dependencies

  const handleSend = async () => {
    if (input.trim()) {
      const currentInput = input;
      setInput('');
      try {
        await sendMessage(currentInput);
      } catch (error) {
        console.error('Failed to send message:', error);
      }
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleFileUploadSuccess = () => {
    setAckMessage('File uploaded successfully!');
    setIsDrawerOpen(false);
  };

  const handleFileUpload = (fileData) => {
    setMessages(prev => [
      ...prev,
      {
        role: 'system',
        content: `File uploaded: ${fileData.filename}`,
      },
    ]);
  };

  // Determine if the initial system message is present
  const isInitialMessage = messages.length === 1 && messages[0].role === 'system';

  return (
    <Container
      maxWidth={false} // Removes default maxWidth
      sx={{
        width: {
          xs: '100%',   // 100% on extra-small devices
          sm: '100%',   // 100% on small devices
          md: '90%',    // 90% on medium devices (e.g., tablets)
          lg: '80%',    // 80% on large devices (e.g., laptops)
          xl: '70%',    // 70% on extra-large devices
        },
        margin: '0 auto', // Centers the container
        padding: 0,        // Removes default padding
      }}
    >
      <ChatContainer>
        {/* Minimalist Header */}
        <Header>
          <Typography
            variant="h2"
            sx={{
              fontWeight: 700,
              color: 'text.primary',
            }}
          >
            Alfred
          </Typography>
          <Typography
            variant="subtitle1"
            sx={{
              color: 'text.secondary',
            }}
          >
            Your Future AI Assistant
          </Typography>
        </Header>

        {/* Messages Area */}
        <MessagesContainer>
          {messages
            .filter((message) => message.role !== 'system') // Exclude system messages
            .map((message, index) => (
              <MessageBubble
                key={index}
                message={message.content}
                isUser={message.role === 'user'}
              />
            ))}
          {isLoading && <LoadingMessage />}
          <div ref={messagesEndRef} />
        </MessagesContainer>

        {/* Initial Message Box */}
        {isInitialMessage && (
          <Box
            sx={{
              position: 'absolute',
              bottom: '150px', // Adjust based on InputContainer's height
              left: '50%',
              transform: 'translateX(-50%)',
              backgroundColor: 'background.paper',
              padding: 2,
              borderRadius: 2,
              boxShadow: 3,
              maxWidth: '120%',
              textAlign: 'left',
              zIndex: 1, // Ensure it appears above other elements
            }}
          >
            <Typography variant="h7">
              Hey, I'm Alfred. Let me know if you want to:<br />- Fetch TAF (Terminal Aerodrome Forecast) data (e.g. Get TAF data for KJFK and KDCA)<br />- Fetch METAR data (e.g Get METAR data for KJFK, KNYL, KNJK) <br />- Upload documents and provide insights<br />- And much more...<br /><br />How can I help you?
            </Typography>
          </Box>
        )}

        {/* Input Area */}
        <InputContainer>
          {/* File Upload Button */}
          <IconButton
            color="primary"
            onClick={() => setIsDrawerOpen(true)}
            aria-label="Upload File"
          >
            <Upload />
          </IconButton>

          {/* Text Input Field */}
          <TextField
            fullWidth
            multiline
            maxRows={4}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Message Alfred..."
            variant="outlined"
            sx={{
              backgroundColor: 'background.paper',
              borderRadius: 1,
            }}
          />

          {/* Microphone Button */}
          <IconButton
            color="primary"
            onClick={() => (isListening ? stopListening() : startListening())}
            aria-label="Voice Input"
          >
            <Mic color={isListening ? 'error' : 'inherit'} />
          </IconButton>

          {/* Send Button */}
          <IconButton color="primary" onClick={handleSend} aria-label="Send Message">
            <Send />
          </IconButton>
        </InputContainer>
      </ChatContainer>

      {/* File Upload Drawer */}
      <Drawer
        anchor="right"
        open={isDrawerOpen}
        onClose={() => setIsDrawerOpen(false)}
      >
        <FileUpload onClose={handleFileUploadSuccess} />
      </Drawer>

      {/* Acknowledgment Snackbar */}
      <Snackbar
        open={!!ackMessage}
        autoHideDuration={5000}
        onClose={() => setAckMessage(null)}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert
          onClose={() => setAckMessage(null)}
          severity="success"
          sx={{ width: '100%' }}
        >
          {ackMessage}
        </Alert>
      </Snackbar>
    </Container>
  );
}

export default ChatInterface;
