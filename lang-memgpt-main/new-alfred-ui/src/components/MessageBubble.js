import React from 'react';
import { Paper, Typography } from '@mui/material';
import { styled } from '@mui/material/styles';

const StyledPaper = styled(Paper, {
  shouldForwardProp: (prop) => prop !== 'isUser'
})(({ theme, isUser }) => ({
  padding: theme.spacing(1, 2),
  maxWidth: '70%',
  alignSelf: isUser ? 'flex-end' : 'flex-start',
  backgroundColor: isUser ? theme.palette.primary.main : theme.palette.background.paper,
  color: isUser ? theme.palette.primary.contrastText : theme.palette.text.primary,
  borderRadius: theme.spacing(2),
  textAlign: 'left',  // Add this line to force left alignment
}));

function MessageBubble({ message, isUser }) {
  return (
    <StyledPaper isUser={isUser}>
      <Typography variant="body1">{message}</Typography>
    </StyledPaper>
  );
}

export default MessageBubble;
