# API Reference

## Overview

This document provides comprehensive API documentation for ConstructGraph. The base URL is `http://<host>:5050` (Docker compose exposes port 5050). All list endpoints support pagination and filtering:

- Pagination parameters: `page` (default 1), `limit` (default 50)
- General filtering: `q` (prefix search), `active` (true/false), see individual endpoint descriptions
- Response format (lists):
```json
{
  "total": 123,
  "items": [ ... ]
}
```
- Error format:
```json
{ "error": "message", "details": {"...": "..."} }
```

## 1) Constructs（构念）

### List constructs (with search/active filtering)
- GET `/api/constructs?active=true&page=1&limit=50&q=alli`
- 200 Response: `{ total, items: [{ name, description, status, canonical_status, dimensions[], parent_constructs[], similar_constructs[], similar_to_constructs[], measurements[], paper_ids[] }] }`

### Create construct (term+definition required; auto-vectorize; optional auto-merge)
- POST `/api/constructs`
- Request:
```json
{
  "term": "emerging technology alliance",
  "definition": "Interorganizational R&D alliance...",
  "paper_uid": "<optional>",
  "measurements": [{"name": "alliance_count", "details": "yearly count"}],
  "auto_merge": true,
  "similarity_threshold": 0.80,
  "merge_confidence_threshold": 0.95
}
```
- 200 Response: `{ term, merged, operation_id?, keep?, drop? }`

### Update construct description (versioned→re-embed→re-resolve→optional auto-merge)
- PATCH `/api/constructs/{name}/description`
- Request:
```json
{ "description": "Updated definition...", "auto_merge": false, "editor": "userA", "reason": "manual" }
```
- 200 Response: `{ revisions:[{revision_id}], embedding:{dimension, revision_id}, candidates:[...], merges:[...] }`

### Soft merge and rollback
- POST `/api/constructs/merge` Request: `{ keep, drop, similarity, confidence, editor?, reason? }`
- POST `/api/constructs/rollback-merge` Request: `{ operation_id }`

## 2) Relationships（关系实例）

### List relationships (filtered by active/status/evidence_type)
- GET `/api/relationships?active=true&status=Empirical_Result&evidence_type=Quantitative&page=1&limit=50`
- 200 Response: `{ total, items: [{ source_construct, target_construct, status, evidence_type, effect_direction, is_validated_causality, relationship_instances:[...] }] }`

### Create relationship (required subject/object/status; optional auto-create constructs)
- POST `/api/relationships`
- Request:
```json
{
  "subject": "shared third party incumbent network",
  "object": "emerging technology alliance",
  "status": "Hypothesized",
  "evidence_type": "Quantitative",
  "effect_direction": "Positive",
  "moderators": ["mabs expertise asymmetry"],
  "mediators": [],
  "controls": [],
  "theories": ["tertius iungens"],
  "paper_uid": "<optional>",
  "auto_create_construct": true
}
```
- 200 Response: `{ ri_uuid }`

### Update relationship (properties + structural rewiring)
- PATCH `/api/relationships/{ri_uuid}`
- Request (partial fields allowed):
```json
{
  "props": { "effect_direction": "Insignificant" },
  "role_changes": { "object": "emergent technology alliance", "add_moderators": ["broker"] },
  "editor": "userA", "reason": "manual fix"
}
```
- 200 Response: `{ operation_id, rewires:[...], updated_fields:[...] }`

### Soft delete/restore/rollback
- POST `/api/relationships/{ri_uuid}/soft-delete` Request: `{ editor?, reason? }`
- POST `/api/relationships/{ri_uuid}/restore` Request: `{ editor?, reason? }`
- POST `/api/relationships/rollback-operation` Request: `{ operation_id }`

## 3) Measurements / Definitions / Dimensions / Similarity / Papers

### Measurements
- POST `/api/measurements` Create/bind to `construct_term` (and optional `relationship_instance` identifier)
- PATCH `/api/measurements/{id}` Versioned update
- DELETE `/api/measurements/{id}` Soft delete

### Definitions
- POST `/api/definitions` Create definition (under term_text+paper constraints)
- PATCH `/api/definitions/{id}` Versioned update
- DELETE `/api/definitions/{id}` Soft delete

### Dimensions
- POST `/api/constructs/{name}/dimensions` body: `{ child }`
- DELETE `/api/constructs/{name}/dimensions/{child}`

### Similarity
- POST `/api/constructs/{a}/similar-to/{b}` body: `{ relationship_type, similarity_score, llm_confidence }`
- DELETE `/api/constructs/{a}/similar-to/{b}`

### Papers
- GET `/api/papers?page=1&limit=50&year=...`
- PATCH `/api/papers/{paper_uid}` Update metadata (authors/journal/year/context etc.)

## 4) Security, Authentication and Idempotency (Recommended)

- Authentication: API-Key/JWT + roles (read-only/edit/audit/admin)
- Idempotency: Write endpoints support `Idempotency-Key` header or request body `client_request_id` to avoid replay
- CORS: Restrict frontend domain whitelist

---

## Stable Data Model and API Contract (Final Version)

This section defines the "long-term stable" data and interface contracts for post-launch. Only "forward-compatible field additions" are allowed hereafter; breaking changes (field renaming/deletion/semantic changes) are strictly prohibited. All deletions use soft delete (`active=false`), all mutable objects provide versioning (Revision) and operation logs (Operation + EdgeRewire), and all writes are rollbackable.

### Core Stability Principles

- Immutable unique keys: `Paper.paper_uid`, `CanonicalConstruct.preferred_name`, `RelationshipInstance.uuid`, `Theory.canonical_id`, `Term.text`, etc.
- Soft delete priority: Any "delete/deactivate/merge" is implemented via `active=false` or `ALIAS_OF`, no physical deletion
- Versioning and audit: Record snapshots and operation logs for "description/property/edge changes", support rollback
- Read default active view: API returns only `active=true` by default; audit view via parameters
- Pagination/filtering required: All list endpoints support `?page`, `?limit` and necessary filter conditions, response returns `total` and `items`

### Entity Models (Nodes)

1) Paper（论文）
- Unique key: `paper_uid`
- Fields: `uuid`, `title`, `doi`, `authors[]`, `publication_year`, `journal`, `research_type`, `research_context`, `is_replication_study`, `filename`, `created_at`, `updated_at`

2) Author（作者）
- Unique key: `full_name`
- Fields: `uuid`, `full_name`

3) Term（术语表示）
- Unique key: `text` (lowercase/remove brackets/normalized)
- Fields: `uuid`, `text`

4) Definition（定义）
- Composite unique: `(text, term_text, paper_uid)`
- Fields: `uuid`, `text`, `context_snippet`, `term_text`, `paper_uid`
- Versioning (optional): `DefinitionRevision(revision_id, snapshot, created_at, editor, reason)`

5) CanonicalConstruct（规范构念）
- Unique key: `preferred_name`
- Fields: `uuid`, `preferred_name`, `status{Provisional|Verified}`, `description`, `canonical_status`, `active`
- Versioning: `ConstructRevision(revision_id, description, created_at, editor, reason)`, `(:CanonicalConstruct)-[:HAS_REVISION]->(:ConstructRevision)`

6) Measurement（测量/操作化）
- Composite unique: `(name, construct_term, paper_uid)`
- Fields: `uuid`, `name`, `description`, `instrument`, `scale_items(json|string|null)`, `scoring_procedure`, `formula`, `reliability`, `validity`, `context_adaptations`, `construct_term`, `paper_uid`, `active` (recommended)
- Versioning (recommended): `MeasurementRevision(revision_id, snapshot, created_at, editor, reason)`

7) Theory（理论）
- Unique key: `canonical_id`
- Fields: `uuid`, `canonical_id`, `name`, `created_at`

8) RelationshipInstance（关系实例）
- Unique key: `uuid`
- Fields: `uuid`, `description`, `context_snippet`, `status{Hypothesized|Empirical_Result}`, `evidence_type{Quantitative|Qualitative|Formal_Model|Simulation|Mixed|null}`, `effect_direction{Positive|Negative|Insignificant|Mixed|null}`, `non_linear_type{U-shaped|Inverted_U-shaped|S-shaped|Other|null}`, `is_validated_causality(bool|null)`, `is_meta_analysis(bool|null)`, `qualitative_finding`, `supporting_quote`, `boundary_conditions`, `replication_outcome{Successful|Failed|Mixed|null}`, `active`
- Versioning: `RelationshipRevision(revision_id, snapshot, created_at, editor, reason)`

9) Operations/Audit (Entity)
- `MergeOperation(id, created_at, initiator, reason, similarity_score, llm_confidence, status{applied|rolled_back})`
- `RelationshipOperation(id, type{update|rewire|split|merge|soft_delete|restore}, created_at, editor, reason, status{applied|rolled_back})`
- `EdgeRewire(edge_type, ri_uuid?, term_text?, measurement_name?, prev_target_id, new_target_id, at)` and `-[:OF_OPERATION]->(Operation)`

### Relationship Models (Edges)

- `(:Paper)-[:AUTHORED_BY]->(:Author)`
- `(:Paper)-[:USES_TERM]->(:Term)`
- `(:Term)-[:HAS_DEFINITION]->(:Definition)`
- `(:Term)-[:IS_REPRESENTATION_OF]->(:CanonicalConstruct)`
- `(:Definition)-[:DEFINED_IN]->(:Paper)`
- `(:CanonicalConstruct)-[:USES_MEASUREMENT]->(:Measurement)`
- `(:Measurement)-[:MEASURED_IN]->(:Paper)`
- `(:CanonicalConstruct)-[:HAS_DIMENSION]->(:CanonicalConstruct)`
- `(:Paper)-[:ESTABLISHES]->(:RelationshipInstance)`
- `(:RelationshipInstance)-[:HAS_SUBJECT]->(:CanonicalConstruct)`
- `(:RelationshipInstance)-[:HAS_OBJECT]->(:CanonicalConstruct)`
- `(:RelationshipInstance)-[:HAS_MODERATOR]->(:CanonicalConstruct)`
- `(:RelationshipInstance)-[:HAS_MEDIATOR]->(:CanonicalConstruct)`
- `(:RelationshipInstance)-[:HAS_CONTROL]->(:CanonicalConstruct)`
- `(:RelationshipInstance)-[:APPLIES_THEORY]->(:Theory)`
- `(:CanonicalConstruct)-[:IS_SIMILAR_TO{similarity_score,llm_confidence,relationship_type}]->(:CanonicalConstruct)` (bidirectional)
- `(:CanonicalConstruct)-[:ALIAS_OF{active,merge_operation_id,created_at}]->(:CanonicalConstruct)` (soft merge alias)

### Constraints and Indexes (Required)

- Unique constraints: `Paper(paper_uid)`, `Author(full_name)`, `Term(text)`, `CanonicalConstruct(preferred_name)`, `Theory(canonical_id)`, `RelationshipInstance(uuid)`, `MergeOperation(id)`, `RelationshipOperation(id)`, `ConstructRevision(revision_id)`, `RelationshipRevision(revision_id)`, `MeasurementRevision(revision_id)`
- Composite unique (application layer or `MERGE` guarantee): `Definition(text,term_text,paper_uid)`, `Measurement(name,construct_term,paper_uid)`
- Indexes: `active`, `preferred_name`, `publication_year`, `status`, `evidence_type`

### Soft Delete/Versioning/Rollback (Unified Convention)

- Soft delete: Set business nodes `active=false` (or `ALIAS_OF` marking), queries filter by default; audit view via `?includeInactive=true`
- Versioning: Update "description/properties" by first creating *Revision*, then updating main node properties; rollback by taking latest *Revision* to restore
- Rollback: Reverse edges according to `Operation`-associated `EdgeRewire`; mark `status=rolled_back`, record `rolled_back_at`

### Vector Database (Qdrant)

- Construct collection: vector=definition embedding (1024, COSINE), payload=`{term, definition, paper, revision_id?}`
- Theory collection: vector=theory name(+context) embedding, payload=`{canonical_id, name, context}`
- Append-only: Historical vectors preserved, visibility controlled by `revision_id`/aliases; no need to destroy old data

### API (Flask) — Read/Write and Audit (Supports pagination and filtering)

General: `?page=1&limit=50`, `?q=` prefix search (construct name/term/title), `?active=` filter active/inactive, response: `{ total, items: [...] }`.

1) Health/Basic
- GET `/api/health`

2) Constructs
- GET `/api/constructs` (`q`, `active`)
- PATCH `/api/constructs/{name}/description` (versioned→re-embed→re-resolve→optional auto-merge)
- POST `/api/constructs/merge` (soft merge)
- POST `/api/constructs/rollback-merge`
- POST `/api/constructs` (create term+definition, vectorize, merge by threshold)
- DELETE `/api/constructs/{name}` (soft delete)

3) Relationships (RelationshipInstance)
- GET `/api/relationships` (`active`, `status`, `evidence_type`)
- PATCH `/api/relationships/{ri_uuid}` (props and role_changes: `subject/object`, add/remove `moderators/mediators/controls`)
- POST `/api/relationships/{ri_uuid}/soft-delete`
- POST `/api/relationships/{ri_uuid}/restore`
- POST `/api/relationships/rollback-operation`
- POST `/api/relationships` (create relationship; optional `auto_create_construct`)
- DELETE `/api/relationships/{ri_uuid}` (soft delete)

4) Measurements
- POST `/api/measurements` (bind to `construct_term` and optional `paper_uid`; can bind `relationship_instance` identifier)
- PATCH `/api/measurements/{id}` (versioned update)
- DELETE `/api/measurements/{id}` (soft delete)

5) Definitions
- POST `/api/definitions`
- PATCH `/api/definitions/{id}` (versioned update)
- DELETE `/api/definitions/{id}` (soft delete)

6) Dimensions/Similarity
- POST `/api/constructs/{name}/dimensions` (`parent->child`)
- DELETE `/api/constructs/{name}/dimensions/{child}`
- POST `/api/constructs/{a}/similar-to/{b}` (`relationship_type`, `similarity_score`, `llm_confidence`)
- DELETE `/api/constructs/{a}/similar-to/{b}`

7) Papers/Authors/Theories
- GET `/api/papers` (`year` etc. filtering)
- PATCH `/api/papers/{paper_uid}` (update metadata; deduplication strategy separately defined)
- PATCH `/api/authors/{id}` (can find by `full_name`)
- PATCH `/api/theories/{canonical_id}` (update `name` etc.; support merge to `canonical_id`)

8) Audit/Rollback
- GET `/api/operations?type={merge|relationship}&page=&limit=` (audit timeline)
- POST `/api/operations/{id}/rollback`

9) Security and Idempotency (Recommended)
- Authentication: API Key/JWT + RBAC (read-only/edit/audit/admin)
- Idempotency: Write endpoints support `Idempotency-Key` header or `client_request_id` field, duplicate submissions don't repeat execution

### Evolution Strategy (Don't Break Old Data)

- New requirements implemented as "add optional fields/new relationships/new endpoints", don't change old field definitions and semantics
- New features need to keep old clients working (default values, backward compatibility)
- Data migration prioritizes "add new nodes/edges" rather than modify or delete old nodes
