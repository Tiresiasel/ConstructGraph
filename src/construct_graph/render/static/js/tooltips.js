// tooltips.js
// Node and edge tooltip rendering.

import { appState } from './state.js';

let tooltip = null;
let tooltipTimeout = null;
let __papersById = null;

function buildPapersById() {
  if (__papersById) return __papersById;
  __papersById = {};
  try {
    (appState.papersData || []).forEach(p => { if (p && p.id) __papersById[p.id] = p; });
  } catch (_) {}
  return __papersById;
}

function htmlWithMathSafe(text) {
  if (text == null) return '';
  try {
    let s = String(text);
    // Lightweight normalization; keep $$...$$ blocks untouched but wrap as block for readability
    s = s.replace(/(\$\$[^$]+\$\$|\\\[[^\]]+\\\])/g, (m) => `<div class=\"math-block\">${m}</div>`);
    // Basic PDF artifact cleanups
    s = s.replace(/\/lparenori/g, '(')
         .replace(/\/rparenori/g, ')')
         .replace(/\/commaori/g, ',')
         .replace(/\/lbracketori/g, '[')
         .replace(/\/rbracketori/g, ']');
    return s;
  } catch (_) {
    return String(text);
  }
}

export function ensureTooltip() {
  tooltip = document.getElementById('tooltip');
  if (!tooltip) {
    tooltip = document.createElement('div');
    tooltip.id = 'tooltip';
    tooltip.className = 'tooltip';
    document.body.appendChild(tooltip);
  }
}

export function hideTooltip() {
  if (tooltipTimeout) clearTimeout(tooltipTimeout);
  if (tooltip) tooltip.classList.remove('show');
}

export function onHoverNode(nodeId, event) {
  const { constructsData, edges } = appState;
  if (tooltipTimeout) clearTimeout(tooltipTimeout);
  tooltipTimeout = setTimeout(() => {
    // Match by preferred name (id) first; fallback uniqueId
    let construct = constructsData.find(c => c.name === nodeId);
    if (!construct) construct = constructsData.find(c => c.uniqueId === nodeId);
    if (!construct) return;
    const papersById = buildPapersById();
    let content = `<div class=\"tooltip-title\">${construct.name}</div>`;

    // Definitions
    if (Array.isArray(construct.definitions) && construct.definitions.length > 0) {
      content += `<div class=\"tooltip-section\"><div class=\"tooltip-title\">定义来源</div>`;
      construct.definitions.forEach(def => {
        if (def && def.definition && def.paper_title) {
          let block = `<div class=\"paper-info\"><strong>定义:</strong> ${htmlWithMathSafe(def.definition)}<br><strong>来源:</strong> ${def.paper_title} ${(def.paper_authors || []).join(', ')} (${def.paper_year || 'N/A'})`;
          try {
            const meta = def.paper_uid ? papersById[def.paper_uid] : null;
            if (meta) { block += `<br><strong>期刊:</strong> ${meta.journal || 'N/A'}`; }
          } catch (_) {}
          block += `</div>`;
          content += block;
        }
      });
      content += `</div>`;
    }

    // Dimensions
    if (construct.dimensions && construct.dimensions.length) {
      content += `<div class=\"tooltip-section\"><div class=\"tooltip-title\">维度</div><div class=\"tooltip-content\">${construct.dimensions.join(', ')}</div></div>`;
    }

    // Parent constructs
    if (construct.parent_constructs && construct.parent_constructs.length) {
      content += `<div class=\"tooltip-section\"><div class=\"tooltip-title\">所属构型</div><div class=\"tooltip-content\">${construct.parent_constructs.join(', ')}</div></div>`;
    }

    // Similar constructs (merged two directions)
    {
      const allSimilar = [];
      (construct.similar_constructs || []).forEach(s => { if (s && s.name) allSimilar.push(s.name); });
      (construct.similar_to_constructs || []).forEach(s => { if (s && s.name) allSimilar.push(s.name); });
      const uniqueSimilar = Array.from(new Set(allSimilar));
      if (uniqueSimilar.length > 0) {
        content += `<div class=\"tooltip-section\"><div class=\"tooltip-title\">相似构型</div>` + uniqueSimilar.map(n => `<div class=\"tooltip-content\">• ${n}</div>`).join('') + `</div>`;
      }
    }

    // Measurements
    if (Array.isArray(construct.measurements) && construct.measurements.length > 0) {
      content += `<div class=\"tooltip-section\"><div class=\"tooltip-title\">测量方式</div>`;
      construct.measurements.forEach(m => {
        if (m && m.name && m.paper_title) {
          let block = `<div class=\"paper-info\"><strong>测量:</strong> ${m.name}<br><strong>来源:</strong> ${m.paper_title} ${(m.paper_authors || []).join(', ')} (${m.paper_year || 'N/A'})`;
          try {
            const meta = m.paper_uid ? papersById[m.paper_uid] : null;
            if (meta) { block += `<br><strong>期刊:</strong> ${meta.journal || 'N/A'}`; }
          } catch (_) {}
          block += `</div>`;
          content += block;
        }
      });
      content += `</div>`;
    }

    // Moderator info if this node acts as a moderator
    try {
      const modEdges = edges.get().filter(e => e.moderatorInfo && e.moderatorInfo.moderator === construct.name);
      if (modEdges && modEdges.length > 0) {
        const mi = modEdges[0].moderatorInfo;
        content += `<div class=\"tooltip-section\" style=\"border-top: 2px solid #6b7280; margin-top: 16px; padding-top: 16px;\"><div class=\"tooltip-title\" style=\"color:#6b7280;\">调节变量信息</div><div class=\"tooltip-content\"><strong>调节的关系:</strong> ${mi.source} → ${mi.target}<br><strong>调节作用:</strong> 作为调节变量影响上述关系的强度和方向<br><strong>关系状态:</strong> ${mi.relationship?.status || 'N/A'}<br><strong>证据类型:</strong> ${mi.relationship?.evidence_type || 'N/A'}<br><strong>效应方向:</strong> ${mi.relationship?.effect_direction || 'N/A'}</div></div>`;
      }
    } catch (_) {}
    tooltip.innerHTML = content;
    tooltip.style.left = event.pageX + 15 + 'px';
    tooltip.style.top = event.pageY - 15 + 'px';
    tooltip.classList.add('show');
  }, 200);
}

export function onHoverEdge(edgeId, event) {
  const { edges, relationshipsData } = appState;
  if (tooltipTimeout) clearTimeout(tooltipTimeout);
  const e = edges.get(edgeId);
  if (!e) return;
  tooltipTimeout = setTimeout(() => {
    if (e.moderatorInfo) {
      const mi = e.moderatorInfo;
      let content = `<div class=\"tooltip-title\" style=\"color:#6b7280;\">调节变量连线</div>`;
      content += `<div class=\"tooltip-section\"><div class=\"tooltip-content\"><strong>调节变量:</strong> ${mi.moderator}<br><strong>调节的关系:</strong> ${mi.source} → ${mi.target}<br><strong>关系状态:</strong> ${mi.relationship?.status || 'N/A'}<br><strong>证据类型:</strong> ${mi.relationship?.evidence_type || 'N/A'}<br><strong>效应方向:</strong> ${mi.relationship?.effect_direction || 'N/A'}</div></div>`;
      tooltip.innerHTML = content;
    } else if (e.mediatorInfo) {
      const mi = e.mediatorInfo;
      let content = `<div class=\"tooltip-title\" style=\"color:#6b7280;\">中介变量连线</div>`;
      content += `<div class=\"tooltip-section\"><div class=\"tooltip-content\"><strong>中介变量:</strong> ${mi.mediator}<br><strong>关系:</strong> ${mi.source} → ${mi.target}</div></div>`;
      tooltip.innerHTML = content;
    } else {
      const rel = relationshipsData.find(r => (r.source_construct === e.from && r.target_construct === e.to) || (r.source_construct === e.to && r.target_construct === e.from));
      if (!rel) return;
      const papersById = buildPapersById();
      let content = `<div class=\"tooltip-title\">关系详情</div>`;
      content += `<div class=\"tooltip-section\"><div class=\"tooltip-content\"><strong>从:</strong> ${rel.source_construct}<br><strong>到:</strong> ${rel.target_construct}<br><strong>状态:</strong> ${rel.status || 'N/A'}<br><strong>证据类型:</strong> ${rel.evidence_type || 'N/A'}<br><strong>方向:</strong> ${rel.effect_direction || 'N/A'}<br><strong>因果验证:</strong> ${rel.is_validated_causality ? '是' : '否'}<br><strong>元分析:</strong> ${rel.is_meta_analysis ? '是' : '否'}</div></div>`;

      // Aggregated moderators/mediators and evidence sources
      try {
        const all = Array.isArray(rel.relationship_instances) ? rel.relationship_instances : [];
        const mods = Array.from(new Set(all.flatMap(ri => (ri.moderators || [])).filter(Boolean)));
        const meds = Array.from(new Set(all.flatMap(ri => (ri.mediators || [])).filter(Boolean)));
        if (mods.length || meds.length) {
          content += `<div class=\"tooltip-section\"><div class=\"tooltip-title\">调节/中介</div><div class=\"tooltip-content\">${mods.length ? `<strong>调节变量:</strong> ${mods.join(', ')}` : ''}${meds.length ? `<br><strong>中介变量:</strong> ${meds.join(', ')}` : ''}</div></div>`;
        }
        const sourceIds = Array.from(new Set(all.map(ri => ri.paper_uid).filter(Boolean)));
        if (sourceIds.length) {
          content += `<div class=\"tooltip-section\"><div class=\"tooltip-title\">证据来源</div>`;
          sourceIds.slice(0,3).forEach(pid => {
            const p = papersById[pid];
            if (p) content += `<div class=\"paper-info\">${p.title}<br>${(p.authors || []).join(', ')} (${p.year || 'N/A'})${p.journal ? ` • ${p.journal}` : ''}</div>`;
          });
          if (sourceIds.length > 3) content += `<div style=\"opacity:0.7;font-size:0.8em;\">还有 ${sourceIds.length - 3} 篇来源...</div>`;
          content += `</div>`;
        }
      } catch (_) {}

      // Relationship instances (up to 3)
      if (Array.isArray(rel.relationship_instances) && rel.relationship_instances.length > 0) {
        content += `<div class=\"tooltip-section\"><div class=\"tooltip-title\">关系实例 (${rel.relationship_instances.length}个)</div>`;
        rel.relationship_instances.slice(0,3).forEach(ri => {
          let stats = null; try { stats = ri.statistical_details ? JSON.parse(ri.statistical_details) : null; } catch(_) { stats = ri.statistical_details; }
          content += `<div class=\"stat-info\"><strong>论文:</strong> ${ri.paper_title || 'N/A'}<br><strong>描述:</strong> ${htmlWithMathSafe(ri.description || ri.context_snippet || 'N/A')}`;
          if (stats) {
            if (stats.p_value !== undefined) content += `<br><strong>P值:</strong> ${stats.p_value}`;
            if (stats.beta_coefficient !== undefined) content += `<br><strong>β系数:</strong> ${stats.beta_coefficient}`;
            if (stats.correlation !== undefined) content += `<br><strong>相关系数:</strong> ${stats.correlation}`;
          }
          if (ri.qualitative_finding) content += `<br><strong>定性发现:</strong> ${ri.qualitative_finding}`;
          if (ri.supporting_quote) content += `<br><strong>支持引用:</strong> \"${ri.supporting_quote}\"`;
          if (ri.boundary_conditions) content += `<br><strong>边界条件:</strong> ${ri.boundary_conditions}`;
          if (ri.replication_outcome) content += `<br><strong>复制结果:</strong> ${ri.replication_outcome}`;
          if (ri.theories && ri.theories.length) content += `<br><strong>理论:</strong> ${ri.theories.join(', ')}`;
          content += `</div>`;
        });
        if (rel.relationship_instances.length > 3) content += `<div style=\"opacity:0.7;font-size:0.8em;\">还有 ${rel.relationship_instances.length - 3} 个实例...</div>`;
        content += `</div>`;
      }
      tooltip.innerHTML = content;
    }
    tooltip.style.left = event.pageX + 15 + 'px';
    tooltip.style.top = event.pageY - 15 + 'px';
    tooltip.classList.add('show');
  }, 200);
}

export function bindTooltipEvents() {
  const { network, nodes, edges } = appState;
  if (!network || !nodes || !edges) return;
  ensureTooltip();
  network.on('hoverNode', params => {
    const node = nodes.get(params.node);
    if (node) onHoverNode(node.id, params.event);
  });
  network.on('blurNode', () => hideTooltip());
  network.on('hoverEdge', params => onHoverEdge(params.edge, params.event));
  network.on('blurEdge', () => hideTooltip());
}


