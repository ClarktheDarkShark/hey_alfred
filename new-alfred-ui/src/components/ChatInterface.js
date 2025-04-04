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
  useMediaQuery,
  useTheme,
  Snackbar,
  Alert,
} from '@mui/material';
import { styled } from '@mui/material/styles';
import { Mic, Send, Upload } from '@mui/icons-material';
import MessageBubble from './MessageBubble';
import FileUpload from './FileUpload';
import { useChat } from '../hooks/useChat';
import { useSpeechRecognition } from '../hooks/useSpeechRecognition';
import LoadingMessage from './LoadingMessage';

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

// Define responsive styles for the Initial Message Box
const getInitialBoxStyles = (theme) => ({
  position: 'absolute',
  bottom: {
    xs: '100px',
    sm: '110px',
    md: '120px',
    lg: '130px',
    xl: '140px',
  },
  left: '50%',
  transform: 'translateX(-50%)',
  backgroundColor: theme.palette.background.paper,
  padding: {
    xs: theme.spacing(1),
    sm: theme.spacing(2),
    md: theme.spacing(3),
  },
  borderRadius: theme.shape.borderRadius,
  boxShadow: theme.shadows[3],
  width: {
    xs: '90%',
    sm: '80%',
    md: '60%',
    lg: '50%',
    xl: '40%',
  },
  maxWidth: '600px',
  textAlign: 'left',
  zIndex: 1,
});

function ChatInterface() {
  const theme = useTheme();
  const isSmallScreen = useMediaQuery(theme.breakpoints.down('sm'));
  const isMediumScreen = useMediaQuery(theme.breakpoints.between('sm', 'md'));
  const isLargeScreen = useMediaQuery(theme.breakpoints.up('lg'));

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
      setInput((prev) => prev + ' ' + transcript);
    }
  }, [transcript, isListening]);

  useEffect(() => {
    if (!initialized.current && messages.length === 0) {
      setMessages([
        {
          role: 'system',
          content: `Hello, I'm Alfred, your AI assistant. I can assist you with:
- Retrieving Terminal Aerodrome Forecast (TAF) data for airports like KJFK and KDCA
- Accessing METAR data for locations such as KJFK, KNYL, and KNJK
- Uploading documents for detailed analysis and insights
- And much more...
How may I assist you today?`,
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
    setMessages((prev) => [
      ...prev,
      {
        role: 'system',
        content: `File uploaded: ${fileData.filename}`,
      },
    ]);
  };

  // Determine if the initial system message is present
  const isInitialMessage = messages.length === 1 && messages[0].role === 'system';

  // Determine responsive typography variant
  const getTypographyVariant = () => {
    if (isSmallScreen) return 'body1';
    if (isMediumScreen) return 'h6';
    if (isLargeScreen) return 'h5';
    return 'h6';
  };

  return (
    <Container
      maxWidth={false} // Removes default maxWidth
      sx={{
        width: {
          xs: '100%', // 100% on extra-small devices
          sm: '100%', // 100% on small devices
          md: '90%', // 90% on medium devices (e.g., tablets)
          lg: '80%', // 80% on large devices (e.g., laptops)
          xl: '70%', // 70% on extra-large devices
        },
        margin: '0 auto', // Centers the container
        padding: 0, // Removes default padding
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
          <Box sx={getInitialBoxStyles(theme)}>
            <Typography variant={getTypographyVariant()}>
              I'm Alfred, your AI assistant. I can assist you with:
              <Box component="ul" sx={{ pl: 2, mt: 1 }}>
                <li>Retrieving Terminal Aerodrome Forecast (TAF) data for airports like KJFK and KDCA</li>
                <li>Accessing METAR data for locations such as KJFK, KNYL, and KNJK</li>
                <li>Uploading documents for analysis and insights</li>
              </Box>
              <Box mt={2}>How may I assist you today?</Box>
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
      <Drawer anchor="right" open={isDrawerOpen} onClose={() => setIsDrawerOpen(false)}>
        <FileUpload onClose={handleFileUploadSuccess} />
      </Drawer>

      {/* Acknowledgment Snackbar */}
      <Snackbar
        open={!!ackMessage}
        autoHideDuration={5000}
        onClose={() => setAckMessage(null)}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert onClose={() => setAckMessage(null)} severity="success" sx={{ width: '100%' }}>
          {ackMessage}
        </Alert>
      </Snackbar>
    </Container>
  );
}

export default ChatInterface;
