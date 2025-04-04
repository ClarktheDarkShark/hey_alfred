// .eslintrc.js
module.exports = {
  env: {
    browser: true,
    es2021: true,
  },
  extends: [
    'eslint:recommended',
    'plugin:react/recommended',
    'plugin:react-hooks/recommended', // Add this line
    // 'airbnb', // If you used Airbnb's style guide
  ],
  parserOptions: {
    ecmaFeatures: {
      jsx: true,
    },
    ecmaVersion: 12, // Adjust as needed
    sourceType: 'module',
  },
  plugins: [
    'react',
    'react-hooks', // Add this line
    // ...other plugins
  ],
  rules: {
    // Define or override specific rules here
    // Example:
    // 'react/react-in-jsx-scope': 'off', // Not needed in React 17+
  },
};
