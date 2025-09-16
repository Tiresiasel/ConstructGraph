<div align="right">English | <a href="README.zh.md">中文</a></div>

# ConstructGraph

ConstructGraph builds and visualizes a research knowledge graph from academic PDFs, fully containerized. Drop PDFs into `data/input/`, start Docker, and open a local URL. The system auto-detects new files and updates the graph in real time.

## Architecture (Docker-first)

- `api` (Flask):
  - Serves the interactive page at `http://localhost:5050` (static `dist/index.html`).
  - Exposes REST endpoints under `/api/*` (constructs, relationships, papers, audit, etc.).
  - Runs a background poller that scans `/app/data/input` for new PDFs and ingests them.
- `neo4j` (internal only): Graph database for constructs and relationships.
- `qdrant` (internal only): Vector database for semantic similarity and entity resolution.

Only port 5050 (Flask) is exposed to the host. Databases are on the internal Docker network (no host port bindings) to avoid conflicts.

## Prerequisites

- Docker + Docker Compose
- An `.env` file providing credentials and the OpenAI API key

## Quick start

1) Prepare environment
```bash
cp .env.example .env
# Edit .env and set at least: OPENAI_API_KEY, NEO4J_PASSWORD (used inside the network)
```

2) Add PDFs to the host folder
```
data/
  input/
    your-paper-1.pdf
    your-paper-2.pdf
```

3) Start services
```bash
docker compose up -d
```

4) Open the app
```
http://localhost:5050
```

- The page loads data via the API in real time.
- Add new PDFs to `data/input/` at any time; the container’s poller (default every 5 seconds) ingests them automatically.

## How ingestion works (no host-side moves)

- The host path `./data/input` is bind‑mounted read-only at `/app/data/input` in the `api` container.
- The poller scans for `*.pdf`, computes the file content SHA‑256, and checks Neo4j for `(:IngestedFile {sha256})`.
  - If already seen, the file is skipped (no duplicate processing).
  - If new, it is processed: extraction → graph upsert → entity resolution (Qdrant) → mark as `IngestedFile`.
- No files are moved or deleted on the host. To force reprocessing a file, change its content or clear the `IngestedFile` record.

Tuning via environment variables (set in compose):
- `POLL_ENABLED=true|false` (default true)
- `POLL_INTERVAL=5` (seconds)
- `INPUT_DIR=/app/data/input` (container path)
- `OUTPUT_DIR=/app/dist` (container path serving index.html)

## API overview

- Base: `http://localhost:5050`
- Health: `GET /api/health`
- Read: `GET /api/constructs`, `GET /api/relationships`, `GET /api/papers` (pagination via `?page=&limit=`)
- Construct ops: update description, soft‑merge / rollback, create, soft‑delete
- Relationship ops: create, update (props/role rewires), soft‑delete/restore, rollback
- Dimensions & Similarity: add/remove child dimension; add/remove `IS_SIMILAR_TO`
- Measurements & Definitions: POST/PATCH/DELETE (soft)
- Audit: `GET /api/operations` returns merge/relationship operation history

The root path `/` serves `dist/index.html`. The page fetches data from `/api/*` at runtime.

## Configuration (.env)

```bash
# Database (used inside the Docker network)
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=12345678

QDRANT_HOST=qdrant
QDRANT_PORT=6333
QDRANT_COLLECTION=construct_definitions

# Paths inside the container
INPUT_DIR=/app/data/input
OUTPUT_DIR=/app/dist

# OpenAI
OPENAI_API_KEY=your_openai_api_key_here
```

You can also adjust `POLL_ENABLED` and `POLL_INTERVAL` in `docker-compose.yml`.

## Project layout

```
ConstructGraph/
  data/
    input/           # Put your PDFs here (host). Container sees /app/data/input (read‑only)
  dist/              # Static assets served by Flask (index.html)
  src/
    construct_graph/
      config.py
      db/neo4j.py
      data/fetchers.py
      models.py
      render/
        templates/constructs_network.html.j2
        page.py
      cli.py
    build_graph.py
    server/app.py     # Flask app (API + static index + background poller)
  docker-compose.yml  # All services (DBs internal-only, API exposed)
```

## Notes

- Only `api` exposes port `5050` to the host; `neo4j` and `qdrant` are internal.
- The visualization runs entirely from the container; there is no need to copy the HTML to the host.
- Qdrant client may warn about minor version differences; functionality is unaffected in this setup.

## Local development (optional)

If you prefer running Python locally, you can still use `./setup.sh` and point to the Docker databases. This is optional; production usage is Docker‑first.



