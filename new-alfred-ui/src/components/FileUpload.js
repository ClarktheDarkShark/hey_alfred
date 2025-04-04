import React, { useState } from 'react';
import { Button, Box, Typography, CircularProgress } from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import { useChat } from '../hooks/useChat';

const FileUpload = ({ onClose }) => {
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState(null);
  const { sendMessage } = useChat(); // Use the global chat hook

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (!file) return;

    const reader = new FileReader();

    reader.onload = async () => {
      try {
        setUploading(true);
        setError(null);

        // Construct the file upload message.
        // Format: "File uploaded: <filename>\nContent: <DataURL>"
        const fileMessageContent = `File uploaded: ${file.name}\nContent: ${reader.result}`;

        // Await the API call.
        await sendMessage(fileMessageContent);

        // After a successful upload, close the uploader.
        if (onClose) {
          onClose();
        }
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

    // Read the file as a Data URL (base64 encoded)
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
