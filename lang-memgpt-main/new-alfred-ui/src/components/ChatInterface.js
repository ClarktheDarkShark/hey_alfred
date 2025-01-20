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
  AppBar,
  List,
  ListItem,
  ListItemText
} from '@mui/material';
import { styled } from '@mui/material/styles';
import { Mic, Send, Upload } from '@mui/icons-material';
import MessageBubble from './MessageBubble';
import FileUpload from './FileUpload';
import { useChat } from '../hooks/useChat';
import { useSpeechRecognition } from '../hooks/useSpeechRecognition';
import LoadingMessage from './LoadingMessage';

const ChatContainer = styled(Box)(({ theme }) => ({
  height: 'calc(100vh - 128px)',
  display: 'flex',
  flexDirection: 'column',
  gap: theme.spacing(2),
  padding: theme.spacing(2),
  position: 'relative',
  zIndex: 0
}));

const MessagesContainer = styled(Paper)(({ theme }) => ({
  flex: 1,
  overflowY: 'auto',
  padding: theme.spacing(2),
  display: 'flex',
  flexDirection: 'column',
  gap: theme.spacing(1),
}));

const InputContainer = styled(Box)(({ theme }) => ({
  display: 'flex',
  gap: theme.spacing(1),
  padding: theme.spacing(2),
  backgroundColor: theme.palette.background.paper,
  position: 'sticky',
  bottom: 0,
}));

const StyledToolbar = styled(Toolbar)({
  display: 'flex',
  justifyContent: 'center',
  position: 'relative',
});

const UploadButton = styled(IconButton)({
  position: 'absolute',
  right: 16,
});

function ChatInterface() {
  const { messages, sendMessage, setMessages, isLoading } = useChat();
  const { isListening, startListening, stopListening, transcript } = useSpeechRecognition();
  const [input, setInput] = useState('');
  const [isDrawerOpen, setIsDrawerOpen] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Update local input if speech recognition captured text
  useEffect(() => {
    if (transcript && isListening) {
      setInput(prev => prev + ' ' + transcript);
    }
  }, [transcript, isListening]);

  const handleSend = async () => {
    if (input.trim()) {
      const currentInput = input;
      setInput(''); // Clear input immediately
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

  const handleFileUpload = (fileData) => {
    // Example usage: inject a "system" note about file upload
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
          <Typography variant="h6">Alfred — Your Future AI Agent</Typography>
          <UploadButton color="inherit" onClick={() => setIsDrawerOpen(true)}>
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
              onClick={() => isListening ? stopListening() : startListening()}
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
        <FileUpload onFileUpload={handleFileUpload} />
      </Drawer>
    </Box>
  );
}

export default ChatInterface;
