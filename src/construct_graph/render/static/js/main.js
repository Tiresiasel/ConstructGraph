// main.js
// Entry: load data, create network, cache layouts, prebuild nodes/edges (hidden), then apply filter.

import { setData, setLayouts, appState } from './state.js';
import { loadAllData } from './api.js';
import { createNetwork } from './network.js';
import { cacheLayoutPositions, setAllNodesToLayout } from './layout.js';
import { applyFilter } from './filter.js';
import { initUI } from './ui.js';
import { setupLabelLayer, drawLabels, bindAfterDrawing } from './labels.js';
import { bindTooltipEvents } from './tooltips.js';
import { bindClickHighlight } from './interactions.js';
import { initSearch } from './search.js';

export async function initApp({ embed_pos = {}, central_pos = {} } = {}) {
  try {
    console.log('Initializing ConstructGraph app...');
    console.log('Available modules:', { 
      setData: typeof setData, 
      setLayouts: typeof setLayouts, 
      loadAllData: typeof loadAllData,
      createNetwork: typeof createNetwork 
    });
    
    // Disable inline template filter if present (use modular pipeline)
    try { window.__MODULAR_FILTER_ENABLED__ = true; } catch(_) {}

    // 1) Data and layouts
    console.log('Loading data...');
    const data = await loadAllData();
    console.log('Data loaded:', { constructs: data.constructs?.length, relationships: data.relationships?.length, papers: data.papers?.length });
    setData(data);
    setLayouts({ embed_pos, central_pos });

    // 2) Network
    console.log('Creating network...');
    const container = document.getElementById('network-container');
    if (!container) {
      throw new Error('Network container not found');
    }
    const { nodes, edges, network } = createNetwork(container);
    console.log('Network created successfully');

    // 3) Cache positions and add nodes hidden with initial coordinates (nodes are NOT fixed to allow temporary dragging)
    cacheLayoutPositions();
    const toAddNodes = [];
    // 计算每个 construct 被不同论文提及的次数（去重）
    const mentionCounts = new Map();
    try {
      (appState.relationshipsData || []).forEach(rel => {
        if (!rel || !rel.source_construct || !rel.target_construct) return;
        const involved = new Set([rel.source_construct, rel.target_construct]);
        const paperIds = new Set();
        const flat = Array.isArray(rel.paper_ids) ? rel.paper_ids.flat() : [];
        flat.forEach(pid => { if (pid) paperIds.add(pid); });
        involved.forEach(cid => {
          if (!mentionCounts.has(cid)) mentionCounts.set(cid, new Set());
          const s = mentionCounts.get(cid);
          paperIds.forEach(pid => s.add(pid));
        });
      });
    } catch (_) {}

    const baseSize = 8.96;
    const minSize = 7.5;
    const maxSize = 22;
    function sizeFor(id) {
      const set = mentionCounts.get(id);
      const k = set ? set.size : 0;
      // 简单线性映射：1次≈baseSize，10次≈baseSize*1.8，上限maxSize
      const scale = 1 + Math.min(0.8, (k > 1 ? (k - 1) * 0.09 : 0));
      return Math.max(minSize, Math.min(maxSize, baseSize * scale));
    }

    (appState.constructsData || []).forEach(c => {
      if (!c || !c.name) return;
      const name = c.name;
      const key = (appState.layoutMode === 'embedding') ? `embedding::${name}` : `centrality::${name}`;
      const all = JSON.parse(localStorage.getItem('kg_saved_positions') || '{}');
      const pos = all[key] || { x: 0, y: 0 };
      toAddNodes.push({ id: name, label: '', x: pos.x, y: pos.y, fixed: false, hidden: true, shape: 'dot', size: sizeFor(name), color: { background: '#e5e7eb', border: '#c9d1d9', highlight: { background: '#f5f6f8', border: '#c9d1d9' } } });
    });
    if (toAddNodes.length) nodes.add(toAddNodes);

    // 4) Prebuild all edges hidden; visibility controlled by applyFilter
    const allEdgesToAdd = [];
    (appState.relationshipsData || []).forEach(rel => {
      if (!rel || !rel.source_construct || !rel.target_construct) return;
      const fromId = rel.source_construct;
      const toId = rel.target_construct;
      const mainEdgeId = `main__${fromId}__${toId}`;
      allEdgesToAdd.push({ id: mainEdgeId, from: fromId, to: toId, hidden: true });
      const instances = Array.isArray(rel.relationship_instances) ? rel.relationship_instances : [];
      instances.forEach(ri => {
        (ri.moderators || []).forEach(modName => {
          if (!modName) return;
          const k = `${modName}__${fromId}__${toId}`;
          allEdgesToAdd.push({ id: `mod__${modName}__${fromId}__${k}`, from: modName, to: fromId, dashes: true, shadow: false, color: { color: '#b4bbc2', highlight: '#f5f6f8', hover: '#ffffff' }, arrows: { to: { enabled: false } }, hidden: true, moderatorInfo: { moderator: modName, source: fromId, target: toId, relationship: rel } });
          allEdgesToAdd.push({ id: `mod__${modName}__${toId}__${k}`, from: modName, to: toId, dashes: true, shadow: false, color: { color: '#b4bbc2', highlight: '#f5f6f8', hover: '#ffffff' }, arrows: { to: { enabled: false } }, hidden: true, moderatorInfo: { moderator: modName, source: fromId, target: toId, relationship: rel } });
        });
        (ri.mediators || []).forEach(medName => {
          if (!medName) return;
          const k = `${medName}__${fromId}__${toId}`;
          allEdgesToAdd.push({ id: `med__${fromId}__${medName}__${k}`, from: fromId, to: medName, dashes: [3,3], shadow: false, color: { color: '#b4bbc2', highlight: '#f5f6f8', hover: '#ffffff' }, arrows: { to: { enabled: true, scaleFactor: 0.8 } }, hidden: true, mediatorInfo: { mediator: medName, source: fromId, target: toId, relationship: rel } });
          allEdgesToAdd.push({ id: `med__${medName}__${toId}__${k}`, from: medName, to: toId, dashes: [3,3], shadow: false, color: { color: '#b4bbc2', highlight: '#f5f6f8', hover: '#ffffff' }, arrows: { to: { enabled: true, scaleFactor: 0.8 } }, hidden: true, mediatorInfo: { mediator: medName, source: fromId, target: toId, relationship: rel } });
        });
      });
    });
    const unique = [];
    const seen = new Set();
    allEdgesToAdd.forEach(e => { if (!seen.has(e.id)) { seen.add(e.id); unique.push(e); } });
    if (unique.length) edges.add(unique);

    // 5) UI and first filter
    console.log('Initializing UI components...');
    initUI();
    initSearch();
    setupLabelLayer();
    bindAfterDrawing();
    bindTooltipEvents();
    bindClickHighlight();
    setAllNodesToLayout(appState.layoutMode);
    
    // Select all papers by default to show all constructs and relationships
    if (appState.papersData && appState.papersData.length > 0) {
      appState.selectedPaperIds = new Set(appState.papersData.map(p => p.id));
      console.log('Selected all papers by default:', appState.selectedPaperIds.size);
      // Sync UI checkboxes if UI is already rendered
      try { window.__ui?.renderPaperList?.(document.getElementById('paper-search')?.value || ''); } catch(_) {}
      try { window.__ui?.updatePapersCount?.(); } catch(_) {}
    }
    
    applyFilter();
    // 生成初次渲染后的“正常效果”基线，供背景点击恢复
    try { window.__captureBaseStyles && window.__captureBaseStyles(); } catch(_) {}
    console.log('App initialization completed successfully');
  } catch (e) {
    console.error('initApp failed:', e);
    console.error('Error stack:', e.stack);
    const container = document.getElementById('network-container');
    if (container) {
      container.innerHTML = `
        <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:100%;color:#e5e7eb;padding:20px;text-align:center;">
          <h3>初始化失败</h3>
          <p>错误信息: ${e.message}</p>
          <p>请检查浏览器控制台获取详细信息</p>
          <button onclick="location.reload()" style="margin-top:10px;padding:10px 20px;background:#007acc;color:white;border:none;border-radius:5px;cursor:pointer;">重新加载</button>
        </div>
      `;
    }
  }
}


