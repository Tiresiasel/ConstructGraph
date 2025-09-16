# Tests Guide

This document defines the full test plan, directory layout, and how to run tests for ConstructGraph.

## Directory layout

```
ConstructGraph/
  src/
  tests/
    unit/
      test_config.py
      test_fetchers.py
      test_build_graph_utils.py
      test_entity_resolution_rules.py
    integration/
      test_api_constructs.py
      test_api_relationships.py
      test_api_measurements.py
      test_api_definitions.py
      test_api_dimensions_similarity.py
      test_api_papers_authors_theories.py
      test_operations_audit.py
    fixtures/
      neo4j_bootstrap.cypher
      sample_papers.json
      sample_constructs.json
      sample_relationships.json
    e2e/  (optional)
      test_runtime_page.py
    conftest.py  (optional for pytest fixtures)
```

- unit/: Pure logic tests; no external services. Use mocks/stubs.
- integration/: Flask API + Neo4j + (optional) Qdrant end-to-end flows.
- fixtures/: Minimal datasets and seed Cypher to bootstrap a tiny graph.
- e2e/: Optional browser-level/runtime checks.

## What to test

### 1) Data fetching (src/construct_graph/data/fetchers.py)
- fetch_constructs(): fields present; measurements filtered by active; dimensions/parents/definitions/paper_ids included; ordering stable.
- fetch_relationships(): relationship_instances contain status/evidence_type/effect_direction/non_linear_type/moderators/mediators/theories/paper meta; instance active flag present.

Approach: mock py2neo.Graph.run() to return canned rows; assert output shapes and filtering.

### 2) Backend ops (src/build_graph.py)
- update_construct_description(): creates revision; calls vector upsert; returns candidates; optional auto-merge triggers soft-merge.
- _apply_soft_merge(): creates MergeOperation + EdgeRewire; sets drop.active=false; creates ALIAS_OF; no DETACH DELETE.
- rollback_merge(): reverses EdgeRewire; restores drop.active=true; disables ALIAS_OF; marks operation rolled_back.
- update_relationship_instance(): creates RelationshipOperation + RelationshipRevision; props set; role rewires logged; soft-delete/restore flip active.

Approach: stub Graph.run() to capture Cypher; assert critical clauses/params; mock vector_db.upsert.

### 3) API (src/server/app.py)
- Health: GET /api/health → {status:"ok"}.
- Constructs: GET /api/constructs returns {total, items}; supports q/active/page/limit. POST creates; PATCH {name}/description revises+re-embeds+resolves; POST /merge applies soft-merge; POST /rollback-merge rolls back; DELETE soft-deletes.
- Relationships: GET with filters; POST creates (auto_create_construct optional); PATCH updates props/roles; soft-delete/restore/DELETE; POST rollback-operation.
- Dimensions/Similarity: POST/DELETE endpoints behave idempotently.
- Measurements: POST/PATCH/DELETE (soft) with graph links.
- Definitions: POST/PATCH/DELETE (soft) with source links.
- Papers/Authors/Theories: PATCH endpoints update metadata.
- Audit: GET /api/operations returns mixed timeline with pagination.

Approach: use Flask app.test_client() with a test Neo4j; seed fixtures/neo4j_bootstrap.cypher; assert responses and DB effects (MergeOperation/EdgeRewire presence, active flags, rollback correctness).

### 4) Visualization runtime template
- constructs_network.html.j2 loader accepts {total, items}; fallback to embedded JSON; handles empty datasets.

Optional e2e: headless browser to load dist/index.html against a running API; assert nodes/edges appear and filters work.

## How to run

Using pytest (recommended):

```
pip install pytest requests
pytest -q tests/unit
# integration (requires services)
docker compose up -d
pytest -q tests/integration
# all
pytest -q tests
```

Using unittest (alternative):

```
python -m unittest discover -s tests -p 'test_*.py'
```

## Fixtures & env

Environment variables for integration tests:
- NEO4J_URI (default bolt://localhost:7687)
- NEO4J_USER (default neo4j)
- NEO4J_PASSWORD
- QDRANT_HOST (default localhost)
- QDRANT_PORT (default 6333)

Seed script fixtures/neo4j_bootstrap.cypher should:
- Create minimal Paper/CanonicalConstruct/RelationshipInstance
- Ensure active=true and required unique keys

## Quality gates
- No irreversible deletes for merge/update flows; soft delete only.
- Merge creates MergeOperation + EdgeRewire; rollback restores and marks rolled_back.
- List APIs always {total, items} with pagination/filters.
- Idempotent POST/DELETE where applicable (dimensions/similarity links).

## CI (optional)
- Add pytest.ini with markers (unit, integration).
- GitHub Actions: run unit on PR; integration on main with services.
- Coverage via pytest-cov for merge/rollback paths.
