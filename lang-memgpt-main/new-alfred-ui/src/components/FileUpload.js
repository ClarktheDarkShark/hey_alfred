import React, { useState } from 'react';
import { Button, Box, Typography, CircularProgress } from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';

const FileUpload = ({ onFileUpload }) => {
  const [uploading, setUploading] = useState(false);

  const handleFileChange = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    try {
      setUploading(true);
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch(`${process.env.REACT_APP_API_URL}/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Upload failed');
      }

      const data = await response.json();
      onFileUpload && onFileUpload(data);
    } catch (error) {
      console.error('Upload error:', error);
    } finally {
      setUploading(false);
    }
  };

  return (
    <Box sx={{ mb: 2 }}>
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
      <Typography variant="caption" display="block" sx={{ mt: 1 }}>
        Supported formats: PDF, CSV, Excel, Word
      </Typography>
    </Box>
  );
};

export default FileUpload;
