// labels.js
// External label layer rendering synced with vis.Network.

import { appState } from './state.js';
import { FIXED_BASE_STYLES } from './constants.js';

let afterDrawingBound = false;

export function setupLabelLayer() {
  const container = document.getElementById('network-container');
  if (!container) return;
  if (!window.__labelLayer) {
    window.__labelLayer = document.createElement('div');
    window.__labelLayer.style.position = 'absolute';
    window.__labelLayer.style.left = '0';
    window.__labelLayer.style.top = '0';
    window.__labelLayer.style.pointerEvents = 'none';
    container.appendChild(window.__labelLayer);
  }
}

export function drawLabels() {
  const { network, nodes } = appState;
  if (!window.__labelLayer || !network || !nodes) return;
  window.__labelLayer.innerHTML = '';
  const scale = network.getScale();
  nodes.forEach(n => {
    if (n && n.hidden) return;
    const pos = network.canvasToDOM(network.getPositions([n.id])[n.id]);
    const el = document.createElement('div');
    el.style.position = 'absolute';
    const nodeRadius = (n.size || 10) * (scale || 1);
    el.style.left = pos.x + 'px';
    const vOffset = nodeRadius + Math.max(6, 2 * scale);
    el.style.top = (pos.y + vOffset) + 'px';
    el.style.transform = 'translate(-50%, 0)';
    const baseOpacity = Math.min(1, Math.max(0, (scale - 0.25) / 0.6));
    const isHighlight = window.__highlightNodes && window.__highlightNodes.has(n.id);
    const isDim = window.__highlightNodes && !window.__highlightNodes.has(n.id);
    const lowAlpha = (typeof window.__lowAlpha === 'number') ? window.__lowAlpha : 0.1;
    const finalOpacity = isDim ? Math.min(baseOpacity, lowAlpha) : baseOpacity;
    if (finalOpacity <= 0.02) return;
    el.style.opacity = String(finalOpacity);
    // Tri-state color for node labels
    if (isHighlight) {
      el.style.color = '#e5e7eb';
    } else if (isDim) {
      el.style.color = 'rgba(233,236,239,' + String(lowAlpha) + ')';
    } else {
      el.style.color = FIXED_BASE_STYLES.node.font.color;
    }
    const fontPx = Math.max(10, Math.min(42, 10 + nodeRadius * 0.35));
    el.style.font = fontPx + 'px Times New Roman';
    el.style.whiteSpace = 'nowrap';
    el.textContent = n.id;
    window.__labelLayer.appendChild(el);
  });
}

export function bindAfterDrawing() {
  const { network } = appState;
  if (!network || afterDrawingBound) return;
  network.off && window.__afterDrawingLabels && network.off('afterDrawing', window.__afterDrawingLabels);
  window.__afterDrawingLabels = drawLabels;
  network.on('afterDrawing', window.__afterDrawingLabels);
  afterDrawingBound = true;
}


