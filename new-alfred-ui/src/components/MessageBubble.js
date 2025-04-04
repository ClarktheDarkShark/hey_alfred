// MessageBubble.js
import React from 'react';
import { Paper } from '@mui/material';
import { styled } from '@mui/material/styles';
import ReactMarkdown from 'react-markdown';

const StyledPaper = styled(Paper, {
  shouldForwardProp: (prop) => prop !== 'isUser',
})(({ theme, isUser }) => ({
  padding: theme.spacing(0, 2),
  maxWidth: '80%',
  alignSelf: isUser ? 'flex-end' : 'flex-start',
  backgroundColor: isUser
    ? theme.palette.primary.main
    : 'rgba(0, 0, 0, 0.7)', // Less transparent background for non-user messages
  color: isUser ? theme.palette.primary.contrastText : theme.palette.text.primary,
  borderRadius: theme.spacing(2),
  textAlign: 'left',
  position: 'relative', // Establish positioning context
  zIndex: 1000, // Set a high z-index value to ensure it stays on top
}));

function MessageBubble({ message, isUser }) {
  return (
    <StyledPaper isUser={isUser}>
      <ReactMarkdown>{message}</ReactMarkdown>
    </StyledPaper>
  );
}

export default MessageBubble;
