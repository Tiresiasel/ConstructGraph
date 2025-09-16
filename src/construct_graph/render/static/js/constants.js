// constants.js
// Centralized visual constants used across modules

export const FIXED_BASE_STYLES = {
  node: {
    color: {
      // Normal (state 1) node fill slightly dimmer than highlight for clear contrast
      background: '#d8dde6',
      border: '#c9d1d9',
      highlight: { background: '#f5f6f8', border: '#c9d1d9' },
    },
    font: {
      // Normal label color for nodes (state 1)
      color: '#c9d1d9',
      size: 11.2,
      face: 'Times New Roman, Times, serif',
    },
    size: 8.96, // reduce another 20% from 11.2
    opacity: 1.0,
  },
  edge: {
    color: {
      // Neutral light gray (no blue tint) — slightly darker for normal state
      color: '#b4bbc2',
      highlight: '#f5f6f8',
      hover: '#ffffff',
    },
    font: {
      color: '#c9d1d9',
      size: 12,
      face: 'Times New Roman, Times, serif',
    },
    width: 1.2,
    opacity: 0.5,
  },
};




