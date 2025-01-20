import React, { useState, useRef, useEffect } from 'react';
import {
  Box,
  Container,
  Paper,
  TextField,
  IconButton,
  Typography,
  Drawer,
  List,
  ListItem,
  ListItemText,
  AppBar,
  Toolbar,
} from '@mui/material';
import { styled } from '@mui/material/styles';
import { Mic, Send, Upload } from '@mui/icons-material';
import MessageBubble from './MessageBubble';
import FileUpload from './FileUpload';
import { useChat } from '../hooks/useChat';
import { useSpeechRecognition } from '../hooks/useSpeechRecognition';

const ChatContainer = styled(Box)(({ theme }) => ({
  height: 'calc(100vh - 128px)',
  display: 'flex',
  flexDirection: 'column',
  gap: theme.spacing(2),
  padding: theme.spacing(2),
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

function ChatInterface() {
  const { messages, sendMessage, setMessages } = useChat();
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

  const handleSend = async () => {
    if (input.trim()) {
      await sendMessage(input);
      setInput('');
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleFileUpload = (fileData) => {
    setMessages(prev => [...prev, {
      type: 'system',
      content: `File uploaded: ${fileData.filename}`
    }]);
  };

  return (
    <Box>
      <AppBar position="static">
        <Toolbar>
          <Typography variant="h6">Alfred — Your Future AI Agent</Typography>
          <IconButton color="inherit" onClick={() => setIsDrawerOpen(true)}>
            <Upload />
          </IconButton>
        </Toolbar>
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
        <FileUpload onFileUpload={handleFileUpload} onClose={() => setIsDrawerOpen(false)} />
      </Drawer>
    </Box>
  );
}

export default ChatInterface;
