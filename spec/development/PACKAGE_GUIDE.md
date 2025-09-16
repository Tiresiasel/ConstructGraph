### ConstructGraph Package (Developer Guide)

This package contains the core modules used by scripts and the CLI.

Modules:
- `config.py` — Central configuration (env-driven). Keys: Neo4j, Qdrant, input/output dirs, output file name.
- `db/neo4j.py` — `get_graph()` factory for Neo4j connections.
- `data/fetchers.py` — Cypher queries that fetch constructs/relationships and JSON-serializeable records.
- `models.py` — Lightweight dataclasses for stronger typing and readability.
- `render/` — Jinja2 templates and helpers for HTML generation.
- `cli.py` — Entry point providing `build` and `visualize` commands.

Data contracts (simplified):
- `Construct`: name, description, dimensions, parent_constructs, paper_ids, best_description.
- `Relationship`: source_construct, target_construct, effect_direction, relationship_instances (array).
- `RelationshipInstance`: status, evidence_type, effect_direction, non_linear_type, paper_uid, moderators/mediators.

Workflow:
1) `build` — ingest PDFs from `CONFIG.input_dir`, populate Neo4j, upsert vectors to Qdrant.
2) `visualize` — fetch graph data, compute layouts, render `constructs_network.html.j2` to `CONFIG.output_html`.

Conventions:
- Keep code and comments in English.
- Do not embed secrets in code; use env vars.
- Prefer functional units with clear inputs/outputs over monolithic procedures.


## Frontend modularization (Constructs Network)

The network visualization template `render/templates/constructs_network.html.j2` was split into ES modules served by Flask from `render/static/js`. The page now loads a small module bootstrap and, when modules are active, skips the legacy inline script (guarded by `window.__USE_MODULES`).

Structure:
- `render/static/js/constants.js`: Fixed styles and color palettes used by nodes/edges.
- `render/static/js/state.js`: Central runtime state (data arrays, selected papers, layout mode, vis DataSets/Network references). Provides setters and binds to `window` for interop.
- `render/static/js/api.js`: Fetches `/api/constructs`, `/api/relationships`, `/api/papers` in parallel and returns normalized lists.
- `render/static/js/network.js`: Creates `vis.Network`, datasets, and base options. Binds references back into state.
- `render/static/js/layout.js`: Caches both centrality and embedding coordinates (keyed as `centrality::name` / `embedding::name`), provides `setAllNodesToLayout(mode)` to instantly switch fixed coordinates without physics.
- `render/static/js/filter.js`: Visibility-only filtering. Computes visible main relationships and derived moderator/mediator edges, ensures missing edges exist (hidden by default), and updates `hidden` flags for nodes/edges instead of adding/removing them. If no paper selected, hides everything.
- `render/static/js/ui.js`: Wires layout toggle buttons and the year slider to call layout/filters.
- `render/static/js/main.js`: Entry point. Loads data, sets layouts, creates network, prebuilds all nodes/edges (hidden, fixed), initializes UI, applies layout, then runs the first filter.

Serving static modules:
- Flask app sets `static_folder=render/static` and `static_url_path='/static'` (see `src/server/app.py`).
- The template injects server-computed layout maps and runs `initApp({ embed_pos, central_pos })` from `main.js`.

Migration approach:
- A guard `window.__USE_MODULES = true` is set before loading modules so the legacy inline script can detect and skip. This enables incremental migration: move feature blocks from the inline script into the modules until the inline script becomes empty, then remove it entirely.

Behavioral guarantees:
- First render and every layout switch pin nodes to precomputed positions (no physics), ensuring stability.
- Filtering never adds/removes nodes/edges after bootstrap; it only toggles `hidden`. This preserves positions and prevents flicker/overlaps.
- When no papers are selected, the graph is empty (all hidden).

Extending:
- Keep state mutations inside `state.js` helpers; keep DOM bindings in `ui.js` and network interactions in `network.js`.
- For new filters, add predicate helpers in `filter.js` and only toggle visibility.
- For new layouts, populate prefixed keys in `layout.js` and call `setAllNodesToLayout(newMode)`.

