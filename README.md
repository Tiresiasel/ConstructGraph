# ConstructGraph

ConstructGraph builds and visualizes a research knowledge graph from academic PDFs, fully containerized. Drop PDFs into `data/input/`, start Docker, and open a local URL. The system auto-detects new files and updates the graph in real time.

## Quick Start

1. **Prepare environment**
   ```bash
   cp .env.example .env
   # Edit .env and set at least: OPENAI_API_KEY, NEO4J_PASSWORD
   ```

2. **Add PDFs to the host folder**
   ```
   data/
     input/
       your-paper-1.pdf
       your-paper-2.pdf
   ```

3. **Start services**
   ```bash
   docker compose up -d
   ```

4. **Open the app**
   ```
   http://localhost:5050
   ```

## Architecture

- `api` (Flask): Serves the interactive page and exposes REST endpoints
- `neo4j` (internal only): Graph database for constructs and relationships  
- `qdrant` (internal only): Vector database for semantic similarity and entity resolution

Only port 5050 (Flask) is exposed to the host. Databases are on the internal Docker network.

## Documentation

All project specifications and guides are organized in the `spec/` directory:

### 📋 Technical Specifications
- [**Technical Specification**](spec/technical/SPECIFICATION.md) - Complete system architecture and technical details

### 🛠️ Development Guides  
- [**Developer Guide**](spec/development/DEVELOPER_GUIDE.md) - Codebase architecture and development workflows
- [**Package Guide**](spec/development/PACKAGE_GUIDE.md) - Core package modules and usage
- [**Prompt Management**](spec/development/PROMPT_MANAGEMENT.md) - Centralized prompt configuration system

### 🧪 Testing
- [**Test Guide**](spec/testing/TEST_GUIDE.md) - Testing strategy and implementation

### 🚀 Deployment & Operations
- [**Docker Development**](spec/deployment/DOCKER_DEVELOPMENT.md) - Docker development environment setup
- [**README (English)**](spec/deployment/README_EN.md) - English deployment guide
- [**README (中文)**](spec/deployment/README_ZH.md) - Chinese deployment guide

### 🔌 API Reference
- [**API Reference**](spec/api/API_REFERENCE.md) - Complete REST API documentation

### 🔬 Experimental
- [**Prompt Lab Guide**](spec/experimental/PROMPT_LAB_GUIDE.md) - Experimental prompt testing framework
- [**Model Analysis Report**](spec/experimental/MODEL_ANALYSIS_REPORT.md) - AI model consistency analysis

## How It Works

- The host path `./data/input` is bind-mounted read-only at `/app/data/input` in the `api` container
- The poller scans for `*.pdf`, computes file content SHA-256, and checks Neo4j for `(:IngestedFile {sha256})`
  - If already seen, the file is skipped (no duplicate processing)
  - If new, it is processed: extraction → graph upsert → entity resolution (Qdrant) → mark as `IngestedFile`
- No files are moved or deleted on the host. To force reprocessing a file, change its content or clear the `IngestedFile` record

## Configuration

See [spec/deployment/README_EN.md](spec/deployment/README_EN.md) for detailed configuration options.

## Project Structure

```
ConstructGraph/
├── spec/                    # All specifications and documentation
│   ├── technical/          # Technical specifications
│   ├── development/        # Development guides
│   ├── testing/           # Testing documentation
│   ├── deployment/        # Deployment and operations
│   ├── api/              # API reference
│   └── experimental/     # Experimental features
├── src/                   # Source code
├── data/input/           # PDF input directory
├── dist/                 # Static assets served by Flask
└── docker-compose.yml    # Docker services configuration
```

## Contributing

Please refer to the [Developer Guide](spec/development/DEVELOPER_GUIDE.md) for contribution guidelines and development setup.

## License

See [LICENSE](LICENSE) file for details.
