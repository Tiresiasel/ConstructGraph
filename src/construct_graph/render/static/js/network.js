// network.js
// Create vis.Network, datasets, and wire basic listeners.

import { appState, bindVis } from './state.js';
import { FIXED_BASE_STYLES } from './constants.js';

export function createNetwork(container) {
  const nodes = new vis.DataSet();
  const edges = new vis.DataSet();

  bindVis({ nodes, edges });

  const options = {
    physics: { enabled: false },
    interaction: { hover: true, zoomView: true, dragView: true },
    nodes: { borderWidth: 2, shadow: true },
    edges: { smooth: false, shadow: true, color: { inherit: false } },
    layout: { improvedLayout: true, hierarchical: false, randomSeed: 1337 },
  };

  const network = new vis.Network(container, { nodes, edges }, options);
  // 拖拽增强（默认单节点临时拖动；按住 Ctrl/Meta 联动邻居）
  let ctrlPressed = false;
  window.addEventListener('keydown', (e) => { if (e.key === 'Control' || e.metaKey) ctrlPressed = true; });
  window.addEventListener('keyup', (e) => { if (e.key === 'Control' || !e.metaKey) ctrlPressed = false; });

  let draggingNodeId = null;
  let dragStartPos = null;
  network.on('dragStart', (params) => {
    if (params && params.nodes && params.nodes.length === 1) {
      draggingNodeId = params.nodes[0];
      dragStartPos = network.getPositions([draggingNodeId])[draggingNodeId];
    } else {
      draggingNodeId = null; dragStartPos = null;
    }
  });
  network.on('dragging', (params) => {
    if (!draggingNodeId || !(ctrlPressed || (params && params.event && (params.event.srcEvent?.metaKey || params.event.srcEvent?.ctrlKey)))) return;
    const currentPos = network.getPositions([draggingNodeId])[draggingNodeId];
    const dx = currentPos.x - (dragStartPos ? dragStartPos.x : currentPos.x);
    const dy = currentPos.y - (dragStartPos ? dragStartPos.y : currentPos.y);
    dragStartPos = currentPos;
    const neighborIds = new Set((network.getConnectedNodes(draggingNodeId) || []));
    neighborIds.add(draggingNodeId);
    const updates = [];
    neighborIds.forEach(id => {
      const p = network.getPositions([id])[id];
      updates.push({ id, x: p.x + dx, y: p.y + dy });
    });
    if (updates.length) nodes.update(updates);
  });
  network.on('dragEnd', () => { draggingNodeId = null; dragStartPos = null; });
  bindVis({ network });
  return { nodes, edges, network };
}


