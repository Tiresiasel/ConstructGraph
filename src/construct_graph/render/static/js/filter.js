// filter.js
// Visibility-only filtering logic (hide/show nodes/edges).

import { appState } from './state.js';
import { FIXED_BASE_STYLES } from './constants.js';

// --- Helpers to build edge labels consistent with constructs_network.html ---
function normalizeStatus(status) {
  const s = (status || '').toLowerCase();
  if (s.includes('empirical')) return 'empirical';
  if (s.includes('hypoth')) return 'hypothesis';
  if (s.includes('propos')) return 'proposition';
  return 'unknown';
}

function symbolForInstance(ri) {
  const nonlin = (ri?.non_linear_type || '').toLowerCase();
  if (nonlin.includes('inverted')) return '∩';
  if (nonlin === 'u' || nonlin.includes('u-shape')) return '∪';
  if (nonlin.includes('s')) return 'S';
  const dir = (ri?.effect_direction || '').toLowerCase();
  if (dir === 'positive') return '+';
  if (dir === 'negative') return '−';
  return '·';
}

function isSelectedAndWithinYear(paperUid) {
  const { papersData, selectedPaperIds } = appState;
  const p = papersData.find(x => x.id === paperUid);
  if (!p) return false;
  const inYear = (!p.year) || (p.year <= currentYear());
  return inYear && selectedPaperIds.has(p.id);
}

function computeEdgeLabelForRel(rel) {
  const instances = (rel.relationship_instances || []).filter(ri => isSelectedAndWithinYear(ri.paper_uid));
  if (instances.length === 0) return '';
  // Buckets strictly by ri.status only
  const buckets = { empirical: new Set(), hypothesis: new Set(), proposition: new Set() };
  instances.forEach(ri => {
    const norm = normalizeStatus(ri.status);
    const sym = symbolForInstance(ri);
    if (norm === 'empirical') {
      buckets.empirical.add(sym);
    } else if (norm === 'hypothesis') {
      buckets.hypothesis.add(sym);
    } else if (norm === 'proposition') {
      buckets.proposition.add(sym);
    }
  });
  // Fixed symbol order; keep middle dot '·'
  const order = ['∩','∪','S','+','−','·'];
  const sortJoin = set => order.filter(s => set.has(s)).join('');
  // Merge P into H
  const mergedH = new Set([...(buckets.hypothesis || []), ...(buckets.proposition || [])]);
  const parts = [];
  const hStr = sortJoin(mergedH);
  const eStr = sortJoin(buckets.empirical);
  if (hStr) parts.push('H:' + hStr);
  if (eStr) parts.push('E:' + eStr);
  // If both empty, return empty string (no label)
  return parts.join(' | ');
}

// --- Causality/Correlation classification (Scheme A) ---
function anyInstance(rel, predicate) {
  const instances = (rel?.relationship_instances || []).filter(ri => isSelectedAndWithinYear(ri.paper_uid));
  for (const ri of instances) {
    try { if (predicate(ri)) return true; } catch (_) {}
  }
  return false;
}

function isCausal(rel) {
  // True if top-level causal flag is true OR any selected instance has it true
  if (rel?.is_validated_causality === true) return true;
  if (anyInstance(rel, ri => ri?.is_validated_causality === true)) return true;
  return false;
}

function normalizeTopStatus(s) {
  return String(s || '').trim().toLowerCase();
}

function isEmpiricalStatus(s) {
  const n = normalizeTopStatus(s);
  return n === 'empirical_result' || n === 'empirical' || n === 'result' || n.includes('empirical');
}

function hasEmpiricalEvidence(rel) {
  if (isEmpiricalStatus(rel?.status)) return true;
  if (anyInstance(rel, ri => isEmpiricalStatus(ri?.status))) return true;
  return false;
}

function hasCorrelationStat(rel) {
  try {
    const r = rel?.stats?.correlation;
    return typeof r === 'number' && Number.isFinite(r);
  } catch (_) { return false; }
}

function isCorrelational(rel) {
  // Correlational = not causal, but has empirical evidence OR correlation stat
  if (isCausal(rel)) return false;
  return hasEmpiricalEvidence(rel) || hasCorrelationStat(rel);
}

export function relationshipMatchesSelection(rel) {
  const { papersData, selectedPaperIds } = appState;
  const flat = Array.isArray(rel.paper_ids) ? rel.paper_ids.flat() : [];
  return flat.some(id => {
    const p = papersData.find(x => x.id === id);
    if (!p) return false;
    const inYear = (!p.year) || (p.year <= currentYear());
    return inYear && selectedPaperIds.has(p.id);
  });
}

export function applyFilter() {
  const { nodes, edges, relationshipsData, constructsData, selectedPaperIds, preset } = appState;
  if (!nodes || !edges) return;

  // When nothing selected: hide all
  if (!selectedPaperIds || selectedPaperIds.size === 0) {
    const nUpd = []; nodes.get().forEach(n => { if (!n.hidden) nUpd.push({ id: n.id, hidden: true }); });
    if (nUpd.length) nodes.update(nUpd);
    const eUpd = []; edges.get().forEach(e => { if (!e.hidden) eUpd.push({ id: e.id, hidden: true }); });
    if (eUpd.length) edges.update(eUpd);
    return;
  }

  const edgesToAdd = [];
  const nodeIdsFromEdges = new Set();

  let candidateRelationships = (relationshipsData || []).filter(rel => {
    if (!rel || !rel.source_construct || !rel.target_construct) return false;
    return relationshipMatchesSelection(rel);
  });

  // Apply preset filtering per Scheme A
  const mode = (preset || 'overview');
  if (mode === 'causal') {
    candidateRelationships = candidateRelationships.filter(isCausal);
  } else if (mode === 'correlation') {
    candidateRelationships = candidateRelationships.filter(isCorrelational);
  }

  candidateRelationships.forEach(rel => {
    const fromId = rel.source_construct;
    const toId = rel.target_construct;
    const label = computeEdgeLabelForRel(rel);
    edgesToAdd.push({ 
      from: fromId, 
      to: toId, 
      label,
      color: { color: FIXED_BASE_STYLES.edge.color.color, highlight: FIXED_BASE_STYLES.edge.color.highlight, hover: FIXED_BASE_STYLES.edge.color.hover },
      // Normal state label颜色应偏暗，避免看起来像“高亮”
      font: { color: '#c9d1d9', size: 14, face: 'Times New Roman', strokeWidth: 2, strokeColor: 'rgba(0,0,0,0.35)' },
      width: 1.6,
      arrows: { to: { enabled: true, scaleFactor: 0.5 } },
    });
    nodeIdsFromEdges.add(fromId); nodeIdsFromEdges.add(toId);
    const instances = Array.isArray(rel.relationship_instances) ? rel.relationship_instances : [];
    instances.forEach(ri => {
      (ri.moderators || []).forEach(mod => {
        edgesToAdd.push({ from: mod, to: fromId, dashes: true, moderatorInfo: { moderator: mod, source: fromId, target: toId, relationship: rel } });
        edgesToAdd.push({ from: mod, to: toId, dashes: true, moderatorInfo: { moderator: mod, source: fromId, target: toId, relationship: rel } });
        nodeIdsFromEdges.add(mod);
      });
      (ri.mediators || []).forEach(med => {
        edgesToAdd.push({ from: fromId, to: med, dashes: [3,3], mediatorInfo: { mediator: med, source: fromId, target: toId, relationship: rel } });
        edgesToAdd.push({ from: med, to: toId, dashes: [3,3], mediatorInfo: { mediator: med, source: fromId, target: toId, relationship: rel } });
        nodeIdsFromEdges.add(med);
      });
    });
  });

  // Node visibility
  const wantVisible = new Set(Array.from(nodeIdsFromEdges));
  const nodeUpdates = [];
  nodes.get().forEach(n => {
    const shouldShow = wantVisible.has(n.id);
    if (!!n.hidden === shouldShow) nodeUpdates.push({ id: n.id, hidden: !shouldShow });
  });
  if (nodeUpdates.length) nodes.update(nodeUpdates);

  // Add any missing edges (hidden initially)
  const toAddEdges = [];
  const toUpdateEdges = [];
  edgesToAdd.forEach(e => {
    let id = e.label ? `main__${e.from}__${e.to}` : `e__${e.from}__${e.to}`;
    if (e.moderatorInfo) {
      const k = `${e.moderatorInfo.moderator}__${e.moderatorInfo.source}__${e.moderatorInfo.target}`;
      id = `mod__${e.from}__${e.to}__${k}`;
    }
    if (e.mediatorInfo) {
      const k = `${e.mediatorInfo.mediator}__${e.mediatorInfo.source}__${e.mediatorInfo.target}`;
      id = `med__${e.from}__${e.to}__${k}`;
    }
    const existing = edges.get(id);
    if (!existing) {
      toAddEdges.push(Object.assign({ id, hidden: true }, e));
    } else {
      // Ensure existing prebuilt edge gets the proper label and styles
      const desired = { id, label: e.label, color: e.color, font: e.font, width: e.width, arrows: e.arrows };
      // Only push update if label missing or styles not set
      if (!existing.label || existing.label !== e.label) toUpdateEdges.push(desired);
    }
  });
  if (toAddEdges.length) edges.add(toAddEdges);
  if (toUpdateEdges.length) edges.update(toUpdateEdges);

  // Edge visibility
  const wantVisibleEdges = new Set();
  edgesToAdd.forEach(e => {
    let id = e.label ? `main__${e.from}__${e.to}` : `e__${e.from}__${e.to}`;
    if (e.moderatorInfo) {
      const k = `${e.moderatorInfo.moderator}__${e.moderatorInfo.source}__${e.moderatorInfo.target}`;
      id = `mod__${e.from}__${e.to}__${k}`;
    }
    if (e.mediatorInfo) {
      const k = `${e.mediatorInfo.mediator}__${e.mediatorInfo.source}__${e.mediatorInfo.target}`;
      id = `med__${e.from}__${e.to}__${k}`;
    }
    wantVisibleEdges.add(id);
  });
  const edgeUpdates = [];
  edges.get().forEach(ed => {
    const shouldShow = wantVisibleEdges.has(ed.id);
    if (!!ed.hidden === shouldShow) edgeUpdates.push({ id: ed.id, hidden: !shouldShow });
  });
  if (edgeUpdates.length) edges.update(edgeUpdates);

  // 过滤完成后，刷新“正常效果”的基线，保证背景点击恢复到当前正常状态
  try { window.__captureBaseStyles && window.__captureBaseStyles(); } catch(_) {}
}

// Backward-compatible currentYear helper (reads #year-range if present)
export function currentYear() {
  try {
    const el = document.getElementById('year-range');
    const v = el ? parseInt(el.value, 10) : NaN;
    return Number.isFinite(v) ? v : new Date().getFullYear();
  } catch (e) {
    return new Date().getFullYear();
  }
}


