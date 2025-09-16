// search.js
// Simple search UI for constructs visible on canvas.

import { appState } from './state.js';

export function initSearch() {
  const input = document.getElementById('global-search');
  const results = document.getElementById('search-results');
  if (!input || !results) return;
  let matches = [];
  let idx = -1;
  function perform(q) {
    const { nodes, constructsData } = appState;
    if (!q) { results.style.display = 'none'; return; }
    const ids = nodes.getIds().filter(id => {
      const n = nodes.get(id);
      return n && !n.hidden && (!n.shape || n.shape !== 'diamond');
    });
    matches = ids.filter(id => id.toLowerCase().includes(q.toLowerCase())).map(id => ({ id }));
    results.innerHTML = '';
    if (!matches.length) { results.style.display = 'none'; return; }
    matches.slice(0, 8).forEach(m => {
      const div = document.createElement('div'); div.className = 'search-result';
      const title = document.createElement('div'); title.className = 'search-result-title'; title.textContent = m.id;
      const meta = document.createElement('div'); meta.className = 'search-result-meta';
      const c = constructsData.find(x => x.name === m.id);
      meta.textContent = c ? `${(c.paper_ids||[]).length} 篇论文` : '构型节点';
      div.appendChild(title); div.appendChild(meta);
      div.onclick = () => focusNode(m.id);
      results.appendChild(div);
    });
    results.style.display = 'block';
  }
  function focusNode(id) {
    const { network } = appState;
    if (!network) return;
    network.selectNodes([id]);
    network.focus(id, { scale: 1.5, animation: true });
  }
  input.addEventListener('input', e => perform(e.target.value.trim()));
  input.addEventListener('keydown', e => {
    if (e.key === 'Enter') {
      if (matches.length) { idx = (idx + 1) % matches.length; focusNode(matches[idx].id); results.style.display='none'; input.value = matches[idx].id; }
    } else if (e.key === 'Escape') {
      results.style.display = 'none'; input.blur();
    }
  });
}


