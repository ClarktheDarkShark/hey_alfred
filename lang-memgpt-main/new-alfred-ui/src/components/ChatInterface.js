import React, { useState, useRef, useEffect } from 'react';
import {
  Box,
  Container,
  Paper,
  TextField,
  IconButton,
  Typography,
  Drawer,
  Toolbar,
  AppBar
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
  height: 'calc(100vh - 128px)',
  display: 'flex',
  flexDirection: 'column',
  gap: theme.spacing(2),
  padding: theme.spacing(2)
}));

const MessagesContainer = styled(Paper)(({ theme }) => ({
  flex: 1,
  overflowY: 'auto',
  padding: theme.spacing(2),
  display: 'flex',
  flexDirection: 'column',
  gap: theme.spacing(1)
}));

const InputContainer = styled(Box)(({ theme }) => ({
  display: 'flex',
  gap: theme.spacing(1),
  padding: theme.spacing(2),
  backgroundColor: theme.palette.background.paper,
  position: 'sticky',
  bottom: 0
}));

// Update StyledToolbar to use a column layout and center items
const StyledToolbar = styled(Toolbar)({
  display: 'flex',
  flexDirection: 'column', // Stack items vertically
  alignItems: 'center',    // Center horizontally
  justifyContent: 'center',
  position: 'relative'
});

const UploadButton = styled(IconButton)({
  position: 'absolute',
  right: 16
});

function ChatInterface() {
  const { messages, sendMessage, setMessages, isLoading } = useChat();
  const { isListening, startListening, stopListening, transcript } = useSpeechRecognition();
  const [input, setInput] = useState('');
  const [isDrawerOpen, setIsDrawerOpen] = useState(false);
  const messagesEndRef = useRef(null);
  const [ackMessage, setAckMessage] = useState(null);

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
        content: `File uploaded: ${fileData.filename}`
      }
    ]);
  };

  return (
    <Box>
      <AppBar position="static">
        <StyledToolbar>
          <Typography
            variant="h4"
            sx={{
              fontWeight: 700,
              fontSize: '2rem',
              lineHeight: 1.2,
              color: 'white',
              letterSpacing: '0.05em'
            }}
          >
            Alfred
          </Typography>
          <Typography
            variant="h3"
            sx={{
              mt: 1, // spacing between lines
              fontWeight: 700,
              fontSize: '2rem',
              lineHeight: 1.2,
              color: 'white',
              letterSpacing: '0.05em'
            }}
          >
            <span style={{ fontSize: '1.5rem' }}>Your Future AI Assistant</span>
          </Typography>
          <UploadButton
            color="inherit"
            onClick={() => setIsDrawerOpen(true)}
            sx={{
              backgroundColor: 'rgba(255, 255, 255, 0.2)',
              '&:hover': {
                backgroundColor: 'rgba(255, 255, 255, 0.3)'
              },
              borderRadius: 2,
              padding: '8px'
            }}
          >
            <Upload />
          </UploadButton>
        </StyledToolbar>
      </AppBar>

      <Container maxWidth="lg">
        <ChatContainer>
          <MessagesContainer>
            {messages.map((message, index) => (
              <MessageBubble
                key={index}
                message={message.content}
                isUser={message.role === 'user'}
              />
            ))}
            {isLoading && <LoadingMessage />}
            <div ref={messagesEndRef} />
          </MessagesContainer>

          <InputContainer>
            <TextField
              fullWidth
              multiline
              maxRows={4}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Message Alfred..."
              variant="outlined"
            />
            <IconButton
              color="primary"
              onClick={() => (isListening ? stopListening() : startListening())}
            >
              <Mic color={isListening ? 'error' : 'inherit'} />
            </IconButton>
            <IconButton color="primary" onClick={handleSend}>
              <Send />
            </IconButton>
          </InputContainer>
        </ChatContainer>
      </Container>

      <Drawer
        anchor="right"
        open={isDrawerOpen}
        onClose={() => setIsDrawerOpen(false)}
      >
        <FileUpload onClose={handleFileUploadSuccess} />
      </Drawer>

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
    </Box>
  );
}

export default ChatInterface;
