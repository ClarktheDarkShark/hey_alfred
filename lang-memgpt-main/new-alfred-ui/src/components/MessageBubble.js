// MessageBubble.js
import React from 'react';
import { Paper } from '@mui/material';
import { styled } from '@mui/material/styles';
import ReactMarkdown from 'react-markdown';

const StyledPaper = styled(Paper, {
  shouldForwardProp: (prop) => prop !== 'isUser',
})(({ theme, isUser }) => ({
  padding: theme.spacing(1, 2),
  maxWidth: '70%',
  alignSelf: isUser ? 'flex-end' : 'flex-start',
  
  backgroundColor: isUser
  ? theme.palette.primary.main
  : '#333333', // Use a darker, non-transparent color

  color: isUser ? theme.palette.primary.contrastText : theme.palette.text.primary,
  borderRadius: theme.spacing(2),
  textAlign: 'left',  // Add this line to force left alignment
}));

function MessageBubble({ message, isUser }) {
  return (
    <StyledPaper isUser={isUser}>
      <ReactMarkdown>{message}</ReactMarkdown>
    </StyledPaper>
  );
}

export default MessageBubble;
