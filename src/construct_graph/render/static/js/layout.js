// layout.js
// Manage deterministic node coordinates and switching between layouts.

import { appState } from './state.js';

const savedPositions = (() => {
  try { return JSON.parse(localStorage.getItem('kg_saved_positions') || '{}'); } catch { return {}; }
})();

function hashCode(str) {
  let h = 0; for (let i = 0; i < str.length; i++) { h = ((h << 5) - h) + str.charCodeAt(i); h |= 0; }
  return h;
}

function deterministicFallbackPosition(id) {
  const h = Math.abs(hashCode(String(id)));
  const angle = (h % 360) / 360 * Math.PI * 2;
  const ring = (Math.floor(h / 360) % 5) + 1;
  const radius = ring * 320;
  return { x: Math.cos(angle) * radius, y: Math.sin(angle) * radius };
}

export function cacheLayoutPositions() {
  const { constructsData, embeddingPositions, centralityPositions } = appState;
  (constructsData || []).forEach(c => {
    if (!c || !c.name) return;
    const name = c.name;
    const cen = (centralityPositions && centralityPositions[name]) || null;
    const emb = (embeddingPositions && embeddingPositions[name]) || null;
    if (cen && Number.isFinite(cen.x) && Number.isFinite(cen.y)) {
      savedPositions[`centrality::${name}`] = { x: cen.x, y: cen.y };
    } else if (!savedPositions[`centrality::${name}`]) {
      savedPositions[`centrality::${name}`] = deterministicFallbackPosition(name);
    }
    if (emb && Number.isFinite(emb.x) && Number.isFinite(emb.y)) {
      savedPositions[`embedding::${name}`] = { x: emb.x, y: emb.y };
    } else if (!savedPositions[`embedding::${name}`]) {
      const p = deterministicFallbackPosition(name);
      savedPositions[`embedding::${name}`] = { x: p.x + 40, y: p.y + 20 };
    }
  });
  try { localStorage.setItem('kg_saved_positions', JSON.stringify(savedPositions)); } catch {}
}

export function setAllNodesToLayout(mode) {
  const { nodes } = appState;
  if (!nodes) return;
  const prefix = mode === 'embedding' ? 'embedding::' : 'centrality::';
  const updates = [];
  nodes.getIds().forEach(id => {
    const key = `${prefix}${id}`;
    const pos = savedPositions[key] || deterministicFallbackPosition(id);
    // 允许用户临时拖拽，所以不要固定节点
    updates.push({ id, x: pos.x, y: pos.y, fixed: false });
  });
  if (updates.length) nodes.update(updates);
}


