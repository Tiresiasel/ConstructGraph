// ui.js
// Year slider, paper list, layout toggle hookups. Minimal version for interop.

import { appState, setLayoutMode, setPreset } from './state.js';
import { t, applyTranslations } from './i18n.js';
import { setAllNodesToLayout } from './layout.js';
import { applyFilter } from './filter.js';

export function initUI() {
  // Layout buttons
  const btnC = document.getElementById('layout-centrality');
  const btnE = document.getElementById('layout-embedding');
  function applyLayout(mode) {
    setLayoutMode(mode);
    setAllNodesToLayout(appState.layoutMode);
    applyFilter();
  }
  if (btnC) btnC.onclick = () => applyLayout('centrality');
  if (btnE) btnE.onclick = () => applyLayout('embedding');

  // Year slider
  const rangeEl = document.getElementById('year-range');
  const yearLabel = document.getElementById('year-label');
  if (rangeEl && yearLabel) {
    rangeEl.addEventListener('input', function() {
      const format = (val) => {
        const dict = (appState.language || 'zh') === 'en' ? 'en' : 'zh';
        // Fallback via i18n year_colon_value
        try {
          const f = (require('./i18n.js').translations || {})[dict]?.year_colon_value; // not always available
          return f ? f(val) : `${t('year')}: ${val}`;
        } catch (_) {
          return `${t('year')}: ${val}`;
        }
      };
      yearLabel.textContent = format(this.value);
      applyTranslations();
      applyFilter();
    });
  }

  // Paper list (left sidebar)
  const paperListEl = document.getElementById('paper-list');
  const paperSearchEl = document.getElementById('paper-search');
  const selectAllBtn = document.getElementById('select-all');
  const clearAllBtn = document.getElementById('clear-all');
  const papersCountEl = document.getElementById('papers-count');

  function updatePapersCount() {
    if (papersCountEl) {
      const size = appState.selectedPaperIds instanceof Set ? appState.selectedPaperIds.size : 0;
      papersCountEl.textContent = String(size);
    }
  }

  function renderPaperList(term = '') {
    if (!paperListEl) return;
    const normalized = (term || '').trim().toLowerCase();
    paperListEl.innerHTML = '';
    const papers = Array.isArray(appState.papersData) ? appState.papersData : [];
    const selected = appState.selectedPaperIds instanceof Set ? appState.selectedPaperIds : new Set();
    const filtered = !normalized
      ? papers
      : papers.filter(p => {
          const title = (p.title || '').toLowerCase();
          const authors = Array.isArray(p.authors) ? p.authors.join(', ').toLowerCase() : '';
          return title.includes(normalized) || authors.includes(normalized);
        });

    filtered.forEach(p => {
      const wrapper = document.createElement('div');
      wrapper.className = 'paper-item';

      const label = document.createElement('label');
      label.style.display = 'flex';
      label.style.alignItems = 'center';
      label.style.gap = '8px';

      const cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.checked = selected.has(p.id);
      cb.onchange = () => {
        if (!(appState.selectedPaperIds instanceof Set)) appState.selectedPaperIds = new Set();
        if (cb.checked) appState.selectedPaperIds.add(p.id);
        else appState.selectedPaperIds.delete(p.id);
        updatePapersCount();
        applyFilter();
      };

      const title = document.createElement('div');
      title.className = 'paper-title';
      title.textContent = p.title || '(Untitled)';

      const meta = document.createElement('div');
      meta.className = 'paper-meta';
      const year = p.year != null ? String(p.year) : 'N/A';
      const authors = Array.isArray(p.authors) ? p.authors.join(', ') : '';
      meta.style.opacity = '.8';
      meta.style.fontSize = '12px';
      meta.textContent = `${authors}${authors ? ' • ' : ''}${year}`;

      label.appendChild(cb);
      label.appendChild(title);
      wrapper.appendChild(label);
      wrapper.appendChild(meta);
      paperListEl.appendChild(wrapper);
    });
  }

  // Wire actions
  if (paperSearchEl) {
    let debounce = null;
    paperSearchEl.addEventListener('input', e => {
      clearTimeout(debounce);
      debounce = setTimeout(() => renderPaperList(e.target.value), 150);
    });
  }

  if (selectAllBtn) {
    selectAllBtn.onclick = () => {
      const papers = Array.isArray(appState.papersData) ? appState.papersData : [];
      appState.selectedPaperIds = new Set(papers.map(p => p.id));
      renderPaperList(paperSearchEl ? paperSearchEl.value : '');
      updatePapersCount();
      applyFilter();
    };
  }

  if (clearAllBtn) {
    clearAllBtn.onclick = () => {
      appState.selectedPaperIds = new Set();
      renderPaperList(paperSearchEl ? paperSearchEl.value : '');
      updatePapersCount();
      applyFilter();
    };
  }

  // Initial render (when modules are enabled)
  renderPaperList('');
  updatePapersCount();

  // View preset buttons (总览/因果/相关)
  document.querySelectorAll('.preset-btn').forEach(btn => {
    const preset = btn && btn.dataset ? btn.dataset.preset : null;
    if (!preset || preset === 'dense') return; // ignore any legacy dense button if exists
    btn.addEventListener('click', () => {
      // toggle active styles
      document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      // update state and re-apply filter
      setPreset(preset);
      applyFilter();
    });
  });

  // Expose re-render helpers so other modules (e.g., main.js) can sync UI after state changes
  if (typeof window !== 'undefined') {
    window.__ui = {
      renderPaperList: (term = '') => renderPaperList(term),
      updatePapersCount: () => updatePapersCount(),
    };
  }
}


