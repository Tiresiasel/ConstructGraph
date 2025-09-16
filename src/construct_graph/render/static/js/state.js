// state.js
// Centralized runtime state and simple getters/setters.

export const appState = {
  constructsData: [],
  relationshipsData: [],
  papersData: [],
  selectedPaperIds: new Set(),
  layoutMode: 'centrality',
  // Current view preset: 'overview' | 'causal' | 'correlation'
  preset: 'overview',
  // UI language: 'zh' | 'en'
  language: 'zh',
  embeddingPositions: {},
  centralityPositions: {},
  nodes: null,
  edges: null,
  network: null,
};

export function setData({ constructs, relationships, papers }) {
  if (Array.isArray(constructs)) appState.constructsData = constructs;
  if (Array.isArray(relationships)) appState.relationshipsData = relationships;
  if (Array.isArray(papers)) appState.papersData = papers;
  if (!(appState.selectedPaperIds instanceof Set)) appState.selectedPaperIds = new Set();
}

export function setLayouts({ embed_pos, central_pos }) {
  appState.embeddingPositions = embed_pos || {};
  appState.centralityPositions = central_pos || {};
}

export function setLayoutMode(mode) {
  appState.layoutMode = mode === 'embedding' ? 'embedding' : 'centrality';
}

export function setPreset(preset) {
  const allowed = new Set(['overview', 'causal', 'correlation']);
  appState.preset = allowed.has(preset) ? preset : 'overview';
}

export function bindVis({ nodes, edges, network }) {
  appState.nodes = nodes || appState.nodes;
  appState.edges = edges || appState.edges;
  appState.network = network || appState.network;
  if (typeof window !== 'undefined') {
    window.nodes = appState.nodes;
    window.edges = appState.edges;
    window.network = appState.network;
  }
}

// Expose for legacy inline script interop
if (typeof window !== 'undefined') {
  window.__appState = appState;
  // Expose setters for debugging/interop
  window.__setPreset = (p) => { try { setPreset(p); } catch(_) {} };
}


