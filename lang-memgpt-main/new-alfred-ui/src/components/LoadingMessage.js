
// Loading Messages.js

import React from 'react';
import { Paper, CircularProgress, Typography } from '@mui/material';
import { styled } from '@mui/material/styles';

const StyledPaper = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(2),
  maxWidth: '70%',
  alignSelf: 'flex-start',
  backgroundColor: theme.palette.background.paper,
  color: theme.palette.text.primary,
  borderRadius: theme.spacing(2),
  display: 'flex',
  alignItems: 'center',
  gap: theme.spacing(2),
}));

function LoadingMessage() {
  return (
    <StyledPaper>
      <CircularProgress size={20} />
      <Typography variant="body1">Alfred is thinking...</Typography>
    </StyledPaper>
  );
}

export default LoadingMessage;
