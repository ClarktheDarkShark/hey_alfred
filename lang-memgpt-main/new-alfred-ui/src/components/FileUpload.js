// FileUpload.jsx
import React, { useState } from 'react';
import { Button, Box, Typography, CircularProgress } from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import { uploadFile } from '../services/api'; // Make sure the path is correct

const FileUpload = ({ onFileUpload, onClose }) => {
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileChange = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    try {
      setUploading(true);
      setError(null);
      
      const formData = new FormData();
      formData.append('file', file);

      // Debug logging
      console.log('Uploading file using secure API URL');
      
      const data = await uploadFile(formData);
      onFileUpload && onFileUpload(data);
      onClose && onClose();
      
    } catch (error) {
      console.error('Upload error:', error);
      setError(error.message || 'Failed to upload file');
    } finally {
      setUploading(false);
    }
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
