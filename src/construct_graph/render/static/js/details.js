// details.js
// Render right-side details panel for node and edge clicks (mirrors constructs_network.html logic)

import { appState } from './state.js';

function getDetailsEl() {
  try { return document.getElementById('details-panel'); } catch (_) { return null; }
}

function htmlWithMathSafe(text) {
  if (text == null) return '';
  try {
    let s = String(text);
    // Wrap display math blocks as centered blocks
    s = s.replace(/(\$\$[^$]+\$\$|\\\[[^\]]+\\\])/g, (m) => `<div class="math-block">${m}</div>`);
    // Basic PDF artifact cleanups
    s = s.replace(/\/lparenori/g, '(')
         .replace(/\/rparenori/g, ')')
         .replace(/\/commaori/g, ',')
         .replace(/\/lbracketori/g, '[')
         .replace(/\/rbracketori/g, ']');
    return s;
  } catch (_) { return String(text); }
}

function buildPapersById() {
  const map = {};
  try { (appState.papersData || []).forEach(p => { if (p && p.id) map[p.id] = p; }); } catch (_) {}
  return map;
}

export function renderNodeDetails(nodeId) {
  const detailsEl = getDetailsEl();
  if (!detailsEl) return;
  const { constructsData, nodes, edges, selectedPaperIds } = appState;
  if (!constructsData) return;
  // Match by preferred name (id) first; fallback uniqueId
  let construct = constructsData.find(c => c && c.name === nodeId);
  if (!construct) construct = constructsData.find(c => c && c.uniqueId === nodeId);
  if (!construct) return;

  let html = `<div class="detail-section"><strong style="font-size:1.1rem">${construct.name}</strong></div>`;

  if (construct.best_description) {
    html += `<div class="detail-section no-border">
      <strong>摘要定义:</strong><br>
      <div style="margin-top:8px;">${htmlWithMathSafe(construct.best_description)}</div>`;
    if (construct.parent_constructs && construct.parent_constructs.length > 0) {
      html += `<div style="margin-top:16px;"><strong>所属构型:</strong> ${construct.parent_constructs.join(', ')}</div>`;
    }
    html += `</div>`;
  }

  if (construct.dimensions && construct.dimensions.length > 0) {
    html += `<div class="detail-section"><strong>维度:</strong><br>${construct.dimensions.map(dim => '• ' + dim).join('<br>')}</div>`;
  }

  // Similar constructs (combined)
  const allSimilar = [];
  (construct.similar_constructs || []).forEach(s => { if (s && s.name) allSimilar.push(s.name); });
  (construct.similar_to_constructs || []).forEach(s => { if (s && s.name) allSimilar.push(s.name); });
  if (allSimilar.length > 0) {
    html += `<div class="detail-section no-border"><strong>相似构型:</strong> ${allSimilar.join(', ')}</div>`;
  }

  // Moderator info section if acts as moderator
  let hasModeratorSection = false;
  try {
    const moderatorEdges = edges.get().filter(e => e.moderatorInfo && e.moderatorInfo.moderator === construct.name);
    if (moderatorEdges.length > 0) {
      hasModeratorSection = true;
      const moderatorInfo = moderatorEdges[0].moderatorInfo;
      html += `<div style="margin: 16px 0 0 0;">
        <div class="detail-section no-border">
          <strong style="color: #6b7280; font-size: 1.05rem;">调节变量信息</strong>
          <div style="margin-top: 12px; opacity: 0.9;">
            <div style="margin-bottom: 8px;"><strong>调节的关系:</strong> ${moderatorInfo.source} → ${moderatorInfo.target}</div>
            <div style="margin-bottom: 8px;"><strong>调节作用:</strong> 作为调节变量影响上述关系的强度和方向</div>
            <div style="margin-bottom: 8px;"><strong>关系状态:</strong> ${moderatorInfo.relationship.status || 'N/A'}</div>
            <div style="margin-bottom: 8px;"><strong>证据类型:</strong> ${moderatorInfo.relationship.evidence_type || 'N/A'}</div>
            <div style="margin-bottom: 8px;"><strong>效应方向:</strong> ${moderatorInfo.relationship.effect_direction || 'N/A'}</div>
          </div>
        </div>
      </div>`;
    }
  } catch (_) {}

  // Group definitions/measurements by paper
  const byPaper = new Map();
  (construct.definitions || []).forEach(d => { if (!d || !d.paper_uid) return; if (!byPaper.has(d.paper_uid)) byPaper.set(d.paper_uid, { defs: [], meas: [] }); byPaper.get(d.paper_uid).defs.push(d); });
  (construct.measurements || []).forEach(m => { if (!m || !m.paper_uid) return; if (!byPaper.has(m.paper_uid)) byPaper.set(m.paper_uid, { defs: [], meas: [] }); byPaper.get(m.paper_uid).meas.push(m); });

  const papersWithContent = [];
  (appState.papersData || []).forEach(p => {
    if (!(selectedPaperIds instanceof Set) || !selectedPaperIds.has(p.id)) return;
    const entry = byPaper.get(p.id);
    if (entry && ((entry.defs && entry.defs.length) || (entry.meas && entry.meas.length))) {
      papersWithContent.push({ paper: p, entry, hasDefinitions: entry.defs.length > 0, hasMeasurements: entry.meas.length > 0 });
    }
  });

  // Do not add an extra separator before papers to avoid double lines under the header

  papersWithContent.forEach(({ paper: p, entry, hasDefinitions, hasMeasurements }) => {
    html += `<div class="detail-section">
      <div style="margin-bottom:16px;">
        <div style="opacity:.85; font-weight:bold;">${p.title || '无标题'}</div>
        <div style="opacity:.7; margin-top:4px;">${(p.authors || []).join(', ')} (${p.year || 'N/A'})</div>
      </div>`;

    if (hasDefinitions) {
      const manyDefs = entry.defs.length > 1;
      html += `<div style="margin-top:16px;"><strong>定义:</strong>`;
      entry.defs.forEach(d => {
        html += `<div style="margin-top:8px; display:flex; align-items:flex-start; gap:8px;">` +
                (manyDefs ? `<span style=\"color:#9ca3af; font-size:12px; margin-top:4px;\">•</span>` : `<span style=\"width:0\"></span>`) +
                `<div style=\"flex:1; word-wrap:break-word;\">${htmlWithMathSafe(d.definition || '')}</div>` +
                `</div>`;
      });
      html += `</div>`;
    }

    if (hasMeasurements) {
      const manyMeas = entry.meas.length > 1;
      html += `<div style="margin-top:16px;"><strong>测量:</strong>`;
      entry.meas.forEach(m => {
        let measHtml = `<div style=\"margin-top:8px; display:flex;align-items:flex-start; gap:8px;\">` +
                       (manyMeas ? `<span style=\"color:#9ca3af; font-size:12px; margin-top:4px;\">•</span>` : `<span style=\"width:0\"></span>`) +
                       `<div style=\"flex:1; word-wrap:break-word;\"><strong>${m.name || ''}</strong>`;
        if (m.description) measHtml += `: ${htmlWithMathSafe(m.description)}`;
        measHtml += `</div></div>`;
        html += measHtml;
      });
      html += `</div>`;
    }

    html += `</div>`; // detail-section
  });

  detailsEl.innerHTML = html || '<div style="opacity:.8">所选论文中暂无该构型的详细信息</div>';
  try { if (window.MathJax) { window.MathJax.typesetPromise([detailsEl]).catch(() => {}); } } catch (_) {}
}

export function renderRelationshipDetailsFromEdge(edge) {
  const detailsEl = getDetailsEl(); if (!detailsEl) return;
  const { relationshipsData, selectedPaperIds } = appState;
  if (!relationshipsData) return;

  // Map any clicked edge (main/mod/med) to the underlying main relationship
  let rel = null;
  if (edge && edge.moderatorInfo) {
    const mi = edge.moderatorInfo;
    rel = relationshipsData.find(r => (r.source_construct === mi.source && r.target_construct === mi.target) || (r.source_construct === mi.target && r.target_construct === mi.source));
  } else if (edge && edge.mediatorInfo) {
    const mi = edge.mediatorInfo;
    rel = relationshipsData.find(r => (r.source_construct === mi.source && r.target_construct === mi.target) || (r.source_construct === mi.target && r.target_construct === mi.source));
  } else if (edge) {
    rel = relationshipsData.find(r => (r.source_construct === edge.from && r.target_construct === edge.to) || (r.source_construct === edge.to && r.target_construct === edge.from));
  }
  if (!rel) return;

  let html = `<div class="detail-section"><strong style="font-size:1.05rem">关系：${rel.source_construct} → ${rel.target_construct}</strong>
    <div style="margin-top:12px;opacity:.8">
      <div style="margin-bottom:8px;display:flex;align-items:center;gap:8px;"><span style="color:#9ca3af; font-size:12px;">•</span><strong>状态:</strong> ${rel.status || 'N/A'}</div>
      <div style="margin-bottom:8px;display:flex;align-items:center;gap:8px;"><span style="color:#9ca3af; font-size:12px;">•</span><strong>证据类型:</strong> ${rel.evidence_type || 'N/A'}</div>
      <div style="margin-bottom:8px;display:flex;align-items:center;gap:8px;"><span style="color:#9ca3af; font-size:12px;">•</span><strong>方向:</strong> ${rel.effect_direction || 'N/A'}</div>
      <div style="margin-bottom:8px;display:flex;align-items:center;gap:8px;"><span style="color:#9ca3af; font-size:12px;">•</span><strong>因果验证:</strong> ${(rel.relationship_instances || []).some(ri => ri.is_validated_causality === true) ? '是' : '否'}</div>
      <div style="margin-bottom:8px;display:flex;align-items:center;gap:8px;"><span style="color:#9ca3af; font-size:12px;">•</span><strong>元分析:</strong> ${rel.is_meta_analysis ? '是' : '否'}</div>
    </div></div>`;

  // Group by paper
  const paperInstances = new Map();
  (rel.relationship_instances || []).forEach(ri => {
    if (!ri || !ri.paper_uid) return;
    if (!paperInstances.has(ri.paper_uid)) paperInstances.set(ri.paper_uid, []);
    paperInstances.get(ri.paper_uid).push(ri);
  });

  (appState.papersData || []).forEach(p => {
    if (!(selectedPaperIds instanceof Set) || !selectedPaperIds.has(p.id)) return;
    const instances = paperInstances.get(p.id);
    if (!instances || instances.length === 0) return;

    html += `<div class="detail-section">
      <div style="margin-bottom:16px;">
        <div style="opacity:.85; font-weight:bold;">${p.title || '无标题'}</div>
        <div style="opacity:.7; margin-top:4px;">${(p.authors || []).join(', ')} (${p.year || 'N/A'})</div>
      </div>`;

    instances.forEach(ri => {
      html += `<div style="margin-top:12px;padding:12px;background:rgba(255,255,255,0.05);border-radius:6px;word-wrap:break-word;overflow-wrap:break-word;">`;
      if (ri.description) {
        html += `<div style=\"margin-bottom:8px;word-wrap:break-word;\"><strong>描述:</strong> ${ri.description}</div>`;
      }
      if (ri.context_snippet) {
        html += `<div style=\"margin-top:8px;font-size:0.9em;opacity:0.8;font-style:italic;display:flex;align-items:flex-start;gap:8px;\">`
             + `<span style=\"color:#9ca3af; font-size:12px; margin-top:4px;\">•</span>`
             + `<div style=\"flex:1; word-wrap:break-word;\">"${ri.context_snippet}"</div>`
             + `</div>`;
      }
      let stats = null; try { stats = ri.statistical_details ? JSON.parse(ri.statistical_details) : null; } catch(_) { stats = ri.statistical_details; }
      if (stats && Object.keys(stats).length > 0) {
        const items = [];
        if (stats.p_value !== undefined) items.push(`P值: ${stats.p_value}`);
        if (stats.beta_coefficient !== undefined) items.push(`β: ${stats.beta_coefficient}`);
        if (stats.correlation !== undefined) items.push(`r: ${stats.correlation}`);
        html += `<div style=\"margin-top:4px;word-wrap:break-word;\"><strong>统计信息:</strong> ${items.join('，') || '无'}</div>`;
      }
      if (ri.qualitative_finding) html += `<div style=\"margin-top:4px;word-wrap:break-word;\"><strong>定性发现:</strong> ${ri.qualitative_finding}</div>`;
      if (ri.boundary_conditions) html += `<div style=\"margin-top:4px;word-wrap:break-word;\"><strong>边界条件:</strong> ${ri.boundary_conditions}</div>`;
      if (ri.replication_outcome) html += `<div style=\"margin-top:4px;word-wrap:break-word;\"><strong>复制结果:</strong> ${ri.replication_outcome}</div>`;
      if (ri.theories && ri.theories.length > 0) html += `<div style=\"margin-top:4px;word-wrap:break-word;\"><strong>理论基础:</strong> ${ri.theories.join(', ')}</div>`;
      if (ri.moderators && ri.moderators.length > 0) html += `<div style=\"margin-top:4px;word-wrap:break-word;\"><strong>调节变量:</strong> ${ri.moderators.join(', ')}</div>`;
      if (ri.mediators && ri.mediators.length > 0) html += `<div style=\"margin-top:4px;word-wrap:break-word;\"><strong>中介变量:</strong> ${ri.mediators.join(', ')}</div>`;
      html += `</div>`;
    });

    html += `</div>`; // detail-section
  });

  detailsEl.innerHTML = html || '<div style="opacity:.8">所选论文中暂无该关系的详细信息</div>';
  try { if (window.MathJax) { window.MathJax.typesetPromise([detailsEl]).catch(() => {}); } } catch (_) {}
}

export function resetDetailsPanel() {
  const el = getDetailsEl(); if (!el) return;
  el.innerHTML = '<div style="opacity:0.8">点击中间的节点或连线查看详细信息</div>';
}


