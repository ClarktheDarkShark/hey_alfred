// FileUpload.jsx
import React, { useState } from 'react';
import { Button, Box, Typography, CircularProgress } from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import { sendMessage } from '../services/api';

const FileUpload = ({ onChatUpdate, onClose, existingMessages = [] }) => {
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (!file) return;

    const reader = new FileReader();

    reader.onload = async () => {
      try {
        setUploading(true);
        setError(null);

        // Construct a chat message that carries the file data.
        // The expected format (which the backend will parse) is:
        // "File uploaded: <filename>
        // Content: <DataURL>"
        const fileMessage = {
          role: 'user',
          content: `File uploaded: ${file.name}\nContent: ${reader.result}`,
        };

        const messages = [...existingMessages, fileMessage];

        const responseData = await sendMessage(messages);

        onChatUpdate && onChatUpdate(responseData);
        onClose && onClose();
      } catch (err) {
        console.error('Error during file upload submission:', err);
        setError(err.message || 'File upload failed');
      } finally {
        setUploading(false);
      }
    };

    reader.onerror = () => {
      setError('Error reading file.');
      setUploading(false);
    };

    // Read file as a Data URL (base64 encoded)
    reader.readAsDataURL(file);
  };

  return (
    <Box sx={{ mb: 2, p: 2 }}>
      <input
        accept=".pdf,.csv,.xlsx,.doc,.docx"
        style={{ display: 'none' }}
        id="file-upload"
        type="file"
        onChange={handleFileChange}
      />
      <label htmlFor="file-upload">
        <Button
          variant="contained"
          component="span"
          startIcon={uploading ? <CircularProgress size={20} /> : <CloudUploadIcon />}
          disabled={uploading}
        >
          {uploading ? 'Uploading...' : 'Upload Document'}
        </Button>
      </label>
      {error && (
        <Typography color="error" variant="body2" sx={{ mt: 1 }}>
          Error: {error}
        </Typography>
      )}
      <Typography variant="caption" display="block" sx={{ mt: 1 }}>
        Supported formats: PDF, CSV, Excel, Word
      </Typography>
    </Box>
  );
};

export default FileUpload;
