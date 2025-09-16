// highlight.js
// Fading others while keeping selected nodes/edges opaque.

import { appState } from './state.js';
import { FIXED_BASE_STYLES } from './constants.js';

function toRgba(color, alpha) {
  const c = String(color || '').trim();
  if (!c) return `rgba(229,231,235,${alpha})`;
  if (c.startsWith('rgba(')) {
    const parts = c.slice(5,-1).split(',').map(x=>x.trim());
    return `rgba(${parts[0]}, ${parts[1]}, ${parts[2]}, ${alpha})`;
  }
  if (c.startsWith('rgb(')) {
    const parts = c.slice(4,-1).split(',').map(x=>x.trim());
    return `rgba(${parts[0]}, ${parts[1]}, ${parts[2]}, ${alpha})`;
  }
  if (c.startsWith('#')) {
    const bigint = parseInt(c.replace('#','').length === 3 ? c.replace('#','').split('').map(ch=>ch+ch).join('') : c.replace('#',''), 16);
    const r = (bigint >> 16) & 255, g = (bigint >> 8) & 255, b = bigint & 255;
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
  }
  return c;
}

function ensureOpaque(color) {
  const c = String(color || '').trim();
  if (c.startsWith('rgba(')) {
    const parts = c.slice(5,-1).split(',').map(x=>x.trim());
    return `rgba(${parts[0]}, ${parts[1]}, ${parts[2]}, 1)`;
  }
  return c || '#e5e7eb';
}

// Keep immutable base styles so we can restore highlight colors/fonts exactly
const nodeBase = (typeof window !== 'undefined' && window.__baseNodeStyles) ? window.__baseNodeStyles : new Map();
const edgeBase = (typeof window !== 'undefined' && window.__baseEdgeStyles) ? window.__baseEdgeStyles : new Map();
if (typeof window !== 'undefined') {
  window.__baseNodeStyles = nodeBase;
  window.__baseEdgeStyles = edgeBase;
}

export function fadeAllExcept(highlightNodeIds, highlightEdgeIds, lowAlpha=0.1) {
  const { nodes, edges } = appState;
  const hiNodes = new Set(highlightNodeIds || []);
  const hiEdges = new Set(highlightEdgeIds || []);
  window.__highlightNodes = hiNodes;
  window.__highlightEdges = hiEdges;
  window.__lowAlpha = lowAlpha;

  nodes.getIds().forEach(id => {
    const n = nodes.get(id); if (!n) return;
    if (!nodeBase.has(id)) {
      nodeBase.set(id, { color: n.color, font: n.font, size: n.size });
    }
    const isHi = hiNodes.has(id);
    if (isHi) {
      const base = nodeBase.get(id) || n;
      const baseColor = base.color || { background: FIXED_BASE_STYLES.node.color.background, border: FIXED_BASE_STYLES.node.color.border };
      const bg = ensureOpaque(baseColor.background || baseColor.color || '#e5e7eb');
      const bd = ensureOpaque(baseColor.border || baseColor.color || '#c9d1d9');
      const fontStyle = base.font || { color: FIXED_BASE_STYLES.node.font.color };
      // Highight node label uses opaque white for stronger contrast
      nodes.update({ id, color: { background: '#f5f6f8', border: bd, highlight: { background: '#f5f6f8', border: bd } }, font: Object.assign({}, fontStyle, { color: '#e5e7eb' }) });
    } else {
      const bg = toRgba((n.color && n.color.background) || FIXED_BASE_STYLES.node.color.background, lowAlpha);
      const bd = toRgba((n.color && n.color.border) || FIXED_BASE_STYLES.node.color.border, lowAlpha);
      const fcol = toRgba((n.font && n.font.color) || FIXED_BASE_STYLES.node.font.color, lowAlpha);
      nodes.update({ id, color: { background: bg, border: bd, highlight: { background: bg, border: bd } }, font: Object.assign({}, n.font||{}, { color: fcol }) });
    }
  });

  edges.getIds().forEach(id => {
    const e = edges.get(id); if (!e) return;
    if (!edgeBase.has(id)) {
      const colorObj = (e.color && typeof e.color === 'object') ? JSON.parse(JSON.stringify(e.color)) : { color: (e.color || FIXED_BASE_STYLES.edge.color.color) };
      edgeBase.set(id, { color: (e.color && (e.color.color || e.color)) || FIXED_BASE_STYLES.edge.color.color, colorObj, font: Object.assign({}, e.font || {}), width: e.width || 1.8, dashes: e.dashes, arrows: e.arrows, shadow: e.shadow });
    }
    const isHi = hiEdges.has(id);
    if (isHi) {
      const base = edgeBase.get(id);
      const baseColor = ensureOpaque((base.colorObj && base.colorObj.color) || base.color || '#e5e7eb');
      const baseFont = Object.assign({}, base.font || { color: '#e5e7eb' });
      const width = Math.max(base.width || 1.8, 3);
      // 强制高亮文字颜色为不透明白色
      baseFont.color = '#e5e7eb';
      edges.update({ id, color: { color: baseColor, highlight: baseColor, hover: baseColor }, font: baseFont, width });
    } else {
      const baseEdgeColor = (e.color && (e.color.color || e.color)) || '#6b7280';
      const faded = toRgba(baseEdgeColor, lowAlpha);
      const baseFontColor = (edgeBase.get(id)?.font?.color) || (e.font && e.font.color) || '#e5e7eb';
      const fadedFont = toRgba(baseFontColor, lowAlpha);
      edges.update({ id, color: { color: faded, highlight: faded, hover: faded }, font: Object.assign({}, e.font||{}, { color: fadedFont }), width: Math.max(1, (e.width || 1.8)) });
    }
  });
}

export function clearHighlight() {
  const { nodes, edges } = appState;
  try {
    // Restore nodes
    nodes.getIds().forEach(id => {
      const base = (window.__baseNodeStyles && window.__baseNodeStyles.get(id)) || null;
      if (base) {
        const bg = ensureOpaque(base.color?.background || base.color?.color || FIXED_BASE_STYLES.node.color.background);
        const bd = ensureOpaque(base.color?.border || base.color?.color || FIXED_BASE_STYLES.node.color.border);
        const font = Object.assign({}, base.font || FIXED_BASE_STYLES.node.font);
        nodes.update({ id, color: { background: bg, border: bd, highlight: { background: bg, border: bd } }, font });
      } else {
        // Fallback to fixed base style
        nodes.update({ id, color: {
          background: FIXED_BASE_STYLES.node.color.background,
          border: FIXED_BASE_STYLES.node.color.border,
          highlight: { background: FIXED_BASE_STYLES.node.color.highlight.background, border: FIXED_BASE_STYLES.node.color.highlight.border }
        }, font: {
          color: FIXED_BASE_STYLES.node.font.color,
          size: FIXED_BASE_STYLES.node.font.size,
          face: FIXED_BASE_STYLES.node.font.face,
        }});
      }
    });

    // Restore edges (including label color)
    edges.getIds().forEach(id => {
      const base = (window.__baseEdgeStyles && window.__baseEdgeStyles.get(id)) || null;
      if (base) {
        const font = Object.assign({}, base.font || FIXED_BASE_STYLES.edge.font);
        const colorObj = base.colorObj ? JSON.parse(JSON.stringify(base.colorObj)) : { color: ensureOpaque(base.color || FIXED_BASE_STYLES.edge.color.color), highlight: FIXED_BASE_STYLES.edge.color.highlight, hover: FIXED_BASE_STYLES.edge.color.hover };
        const upd = { id, color: colorObj, font, width: base.width || FIXED_BASE_STYLES.edge.width };
        if (typeof base.dashes !== 'undefined') upd.dashes = base.dashes;
        if (typeof base.arrows !== 'undefined') upd.arrows = base.arrows;
        if (typeof base.shadow !== 'undefined') upd.shadow = base.shadow;
        edges.update(upd);
      } else {
        edges.update({ id, color: {
          color: FIXED_BASE_STYLES.edge.color.color,
          highlight: FIXED_BASE_STYLES.edge.color.highlight,
          hover: FIXED_BASE_STYLES.edge.color.hover,
        }, font: {
          color: FIXED_BASE_STYLES.edge.font.color,
          size: FIXED_BASE_STYLES.edge.font.size,
          face: FIXED_BASE_STYLES.edge.font.face,
        }, width: FIXED_BASE_STYLES.edge.width });
      }
    });
  } catch (e) {
    console.warn('clearHighlight restore error:', e);
  }

  // Reset markers
  window.__highlightNodes = null;
  window.__highlightEdges = null;
  window.__lowAlpha = null;
  try { if (appState.network && typeof appState.network.redraw === 'function') appState.network.redraw(); } catch(_) {}
}

// Capture current datasets as the new "normal" baseline (for case 1: 无选择状态)
export function captureBaseStyles() {
  const { nodes, edges } = appState;
  try {
    const nb = new Map();
    const eb = new Map();
    nodes.get().forEach(n => { nb.set(n.id, { color: n.color, font: n.font, size: n.size }); });
    edges.get().forEach(e => { 
      const baseColor = (e.color && (e.color.color || e.color)) || FIXED_BASE_STYLES.edge.color.color; // 若无显式颜色，用正常态色
      eb.set(e.id, { color: baseColor, font: Object.assign({}, e.font || {}), width: e.width || 1.8 }); 
    });
    window.__baseNodeStyles = nb;
    window.__baseEdgeStyles = eb;
  } catch (e) { console.warn('captureBaseStyles error:', e); }
}

// Expose to global so other modules can refresh baseline after数据重算
if (typeof window !== 'undefined') {
  window.__captureBaseStyles = captureBaseStyles;
}


