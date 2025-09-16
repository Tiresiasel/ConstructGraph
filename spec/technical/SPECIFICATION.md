# ConstructGraph - Technical Specification

## Project Overview

ConstructGraph is a comprehensive academic knowledge graph system that automatically extracts, processes, and visualizes research constructs and their relationships from academic PDFs. The system is fully containerized and provides both a REST API and an interactive web interface for exploring research knowledge networks.

## Architecture Overview

### System Components

1. **API Service (Flask)** - Main application server
2. **Neo4j Database** - Graph database for constructs and relationships
3. **Qdrant Vector Database** - Semantic similarity and entity resolution
4. **Frontend Web Interface** - Interactive visualization and management UI
5. **Background Processing** - Automated PDF ingestion and processing

### Technology Stack

**Backend:**
- Python 3.11+
- Flask (Web framework)
- Neo4j (Graph database)
- Qdrant (Vector database)
- OpenAI API (LLM processing)
- PyPDF (PDF processing)
- Sentence Transformers (Embeddings)

**Frontend:**
- Vanilla JavaScript (ES6 modules)
- Vis.js Network (Graph visualization)
- MathJax (Mathematical formula rendering)
- Jinja2 (Template engine)
- CSS3 (Styling)

**Infrastructure:**
- Docker & Docker Compose
- Multi-container architecture
- Volume mounting for data persistence

## Backend Specification

### 1. Core Data Models

#### Construct Model
```python
@dataclass
class Construct:
    name: str
    description: Optional[str] = None
    status: Optional[str] = None
    canonical_status: Optional[str] = None
    definitions: list = field(default_factory=list)
    measurements: list = field(default_factory=list)
    dimensions: List[str] = field(default_factory=list)
    parent_constructs: List[str] = field(default_factory=list)
    similar_constructs: list = field(default_factory=list)
    similar_to_constructs: list = field(default_factory=list)
    paper_ids: List[str] = field(default_factory=list)
    best_description: Optional[str] = None
```

#### Relationship Model
```python
@dataclass
class Relationship:
    source_construct: str
    target_construct: str
    status: Optional[str] = None
    evidence_type: Optional[str] = None
    effect_direction: Optional[str] = None
    is_validated_causality: Optional[bool] = None
    relationship_instances: List[RelationshipInstance] = field(default_factory=list)
    paper_ids: List[str] = field(default_factory=list)
```

#### Paper Model
```python
@dataclass
class Paper:
    id: str
    title: str
    authors: Optional[str] = None
    year: Optional[int] = None
    journal: Optional[str] = None
    research_type: Optional[str] = None
    research_context: Optional[str] = None
    is_replication_study: Optional[bool] = None
```

### 2. Database Schema

#### Neo4j Node Types
- **CanonicalConstruct**: Main construct entities
- **Paper**: Academic papers
- **Author**: Paper authors
- **Theory**: Theoretical frameworks
- **Measurement**: Construct measurement instruments
- **Definition**: Construct definitions
- **RelationshipInstance**: Specific relationship instances
- **IngestedFile**: File processing tracking

#### Neo4j Relationship Types
- **HAS_RELATIONSHIP**: Between constructs
- **HAS_DIMENSION**: Hierarchical construct relationships
- **IS_SIMILAR_TO**: Similarity relationships
- **USES_MEASUREMENT**: Construct-measurement links
- **HAS_DEFINITION**: Construct-definition links
- **DEFINED_IN**: Definition-paper links
- **MEASURED_IN**: Measurement-paper links

### 3. API Endpoints

#### Health & System
- `GET /api/health` - System health check
- `POST /api/cleanup/duplicates` - Clean duplicate papers/constructs
- `POST /api/cleanup/reset-ingestion` - Reset ingestion status

#### Constructs Management
- `GET /api/constructs` - List constructs (with pagination, filtering)
- `PATCH /api/constructs/<name>/description` - Update construct description
- `POST /api/constructs` - Create new construct
- `DELETE /api/constructs/<name>` - Soft delete construct
- `POST /api/constructs/merge` - Merge constructs
- `POST /api/constructs/rollback-merge` - Rollback merge operation

#### Relationships Management
- `GET /api/relationships` - List relationships (with pagination, filtering)
- `POST /api/relationships` - Create relationship
- `PATCH /api/relationships/<ri_uuid>` - Update relationship
- `DELETE /api/relationships/<ri_uuid>` - Soft delete relationship
- `POST /api/relationships/<ri_uuid>/restore` - Restore relationship
- `POST /api/relationships/rollback-operation` - Rollback relationship operation

#### Papers Management
- `GET /api/papers` - List papers (with pagination, year filtering)
- `PATCH /api/papers/<paper_uid>` - Update paper metadata

#### Dimensions & Similarity
- `POST /api/constructs/<name>/dimensions` - Add dimension relationship
- `DELETE /api/constructs/<name>/dimensions/<child>` - Remove dimension
- `POST /api/constructs/<a>/similar-to/<b>` - Add similarity relationship
- `DELETE /api/constructs/<a>/similar-to/<b>` - Remove similarity

#### Measurements & Definitions
- `POST /api/measurements` - Create measurement
- `PATCH /api/measurements/<uuid>` - Update measurement
- `DELETE /api/measurements/<uuid>` - Soft delete measurement
- `POST /api/definitions` - Create definition
- `PATCH /api/definitions/<uuid>` - Update definition
- `DELETE /api/definitions/<uuid>` - Soft delete definition

#### Audit & Operations
- `GET /api/operations` - List all operations (merge, relationship changes)

### 4. PDF Processing Pipeline

#### Input Processing
1. **File Detection**: Background poller scans `data/input/` directory
2. **Deduplication**: SHA-256 hash checking against `IngestedFile` records
3. **Concurrent Processing**: Up to 4 PDFs processed simultaneously
4. **Status Tracking**: In-progress, succeeded, failed states

#### Extraction Process
1. **PDF Text Extraction**: Using PyPDF library
2. **LLM Processing**: OpenAI API for construct/relationship extraction
3. **Entity Resolution**: Qdrant vector similarity for duplicate detection
4. **Graph Population**: Neo4j database updates
5. **Status Recording**: Success/failure tracking

#### Configuration
- `POLL_ENABLED`: Enable/disable background polling
- `POLL_INTERVAL`: Polling frequency (default: 5 seconds)
- `MAX_CONCURRENT_PDFS`: Concurrent processing limit (default: 4)
- `POLL_STALE_MINUTES`: Stale processing timeout (default: 10 minutes)

### 5. Vector Database Integration

#### Qdrant Collections
- **construct_definitions**: Construct definition embeddings
- **theories**: Theoretical framework embeddings

#### Similarity Operations
- Semantic similarity calculation for construct merging
- Entity resolution for duplicate detection
- Auto-merge functionality with configurable thresholds

## Frontend Specification

### 1. User Interface Architecture

#### Layout Structure
- **Global Toolbar**: Search, filters, view presets, layout controls
- **Sidebar**: Paper list with selection controls
- **Main Canvas**: Interactive network visualization
- **HUD Overlays**: Contextual information displays

#### Responsive Design
- Mobile-friendly responsive layout
- Collapsible sidebar for smaller screens
- Touch-friendly controls for mobile devices

### 2. Visualization Components

#### Network Visualization
- **Engine**: Vis.js Network library
- **Layouts**: 
  - Centrality-based layout (force-directed)
  - Semantic embedding layout (precomputed positions)
- **Interactive Features**:
  - Zoom and pan
  - Node selection and highlighting
  - Edge filtering and styling
  - Real-time updates

#### Node Styling
- **Constructs**: Color-coded by status and type
- **Papers**: Distinct styling for paper nodes
- **Size**: Based on centrality or importance metrics
- **Labels**: Dynamic label visibility based on zoom level

#### Edge Styling
- **Relationship Types**: Different colors and styles
- **Effect Direction**: Arrow direction and thickness
- **Evidence Strength**: Line opacity and style
- **Non-linear Types**: Special styling for S/U/Inverted-U curves

### 3. JavaScript Module Architecture

#### Core Modules
- **main.js**: Application initialization and coordination
- **network.js**: Vis.js network management
- **api.js**: REST API communication
- **state.js**: Application state management
- **search.js**: Search functionality
- **filter.js**: Data filtering logic
- **layout.js**: Layout computation and switching
- **interactions.js**: User interaction handling
- **tooltips.js**: Contextual information display
- **ui.js**: UI component management
- **labels.js**: Dynamic label management
- **highlight.js**: Selection and highlighting
- **constants.js**: Application constants

#### Module Communication
- Event-driven architecture
- Centralized state management
- API abstraction layer
- Error handling and loading states

### 4. User Experience Features

#### Search & Discovery
- **Global Search**: Real-time construct search with autocomplete
- **Paper Search**: Filter papers by title, author, year
- **Advanced Filters**: Relationship type, evidence strength, year range
- **View Presets**: Predefined filter combinations

#### Interaction Patterns
- **Selection**: Single and multi-select nodes
- **Highlighting**: Related node highlighting
- **Tooltips**: Contextual information on hover
- **Modal Dialogs**: Detailed information and editing

#### Data Management
- **Real-time Updates**: Live data refresh from API
- **Optimistic Updates**: Immediate UI feedback
- **Error Handling**: Graceful error recovery
- **Loading States**: Progress indicators

### 5. Styling & Theming

#### Design System
- **Color Palette**: Dark theme with academic styling
- **Typography**: Times New Roman for academic feel
- **Spacing**: Consistent spacing scale
- **Components**: Reusable UI components

#### CSS Architecture
- **Modular CSS**: Component-based styling
- **Responsive Design**: Mobile-first approach
- **Accessibility**: WCAG compliance considerations
- **Performance**: Optimized for smooth animations

## Deployment Specification

### 1. Docker Configuration

#### Multi-Container Architecture
```yaml
services:
  neo4j:
    image: neo4j:5.15-community
    environment:
      - NEO4J_AUTH=${NEO4J_USER:-neo4j}/${NEO4J_PASSWORD:-12345678}
      - NEO4J_PLUGINS=["apoc"]
    volumes:
      - neo4j_data:/data
      - neo4j_logs:/logs
    networks:
      - constructgraph-network

  qdrant:
    image: qdrant/qdrant:latest
    volumes:
      - qdrant_data:/qdrant/storage
    networks:
      - constructgraph-network

  api:
    build: .
    environment:
      - NEO4J_URI=bolt://neo4j:7687
      - QDRANT_HOST=qdrant
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    ports:
      - "5050:5050"
    volumes:
      - ./data/input:/app/data/input:ro
    networks:
      - constructgraph-network
```

#### Container Specifications
- **API Container**: Python 3.11-slim base image
- **Database Containers**: Official Neo4j and Qdrant images
- **Volume Mounting**: Persistent data storage
- **Network Isolation**: Internal Docker network

### 2. Environment Configuration

#### Required Environment Variables
```bash
# Database Configuration
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=12345678

# Vector Database
QDRANT_HOST=qdrant
QDRANT_PORT=6333
QDRANT_COLLECTION=construct_definitions

# OpenAI API
OPENAI_API_KEY=your_openai_api_key_here

# Application Settings
POLL_INTERVAL=5
MAX_CONCURRENT_PDFS=4
POLL_STALE_MINUTES=10
```

#### Optional Configuration
- `POLL_ENABLED`: Enable/disable background polling
- `FLASK_HOST`: API host binding (default: 0.0.0.0)
- `FLASK_PORT`: API port (default: 5050)
- `RESET_DATABASE`: Reset database on startup

### 3. Data Persistence

#### Volume Mounts
- **neo4j_data**: Graph database storage
- **qdrant_data**: Vector database storage
- **hf_cache**: Hugging Face model cache
- **data/input**: Read-only PDF input directory

#### Backup Strategy
- Database snapshots via Neo4j backup tools
- Qdrant collection exports
- Configuration file versioning

### 4. Security Considerations

#### Network Security
- Internal Docker network isolation
- No external database port exposure
- API-only external access

#### Data Security
- Environment variable management
- API key protection
- Input validation and sanitization

## Performance Specifications

### 1. Scalability

#### Concurrent Processing
- **PDF Processing**: Up to 4 concurrent PDFs
- **API Requests**: Configurable concurrent limits
- **Database Connections**: Connection pooling

#### Resource Requirements
- **Memory**: Minimum 4GB RAM recommended
- **Storage**: Variable based on PDF corpus size
- **CPU**: Multi-core recommended for concurrent processing

### 2. Optimization

#### Database Optimization
- **Indexing**: Strategic Neo4j indexes
- **Query Optimization**: Efficient Cypher queries
- **Connection Pooling**: Reused database connections

#### Frontend Optimization
- **Lazy Loading**: On-demand data loading
- **Caching**: Browser and API response caching
- **Bundle Optimization**: Minified JavaScript/CSS

## Development & Maintenance

### 1. Development Workflow

#### Local Development
- Docker Compose for local development
- Hot reloading for frontend changes
- Environment variable configuration

#### Testing Strategy
- Unit tests for core functionality
- Integration tests for API endpoints
- End-to-end tests for user workflows

### 2. Monitoring & Logging

#### Application Logging
- Structured logging with timestamps
- Error tracking and reporting
- Performance metrics collection

#### Health Monitoring
- Database connectivity checks
- API endpoint health monitoring
- Background process status tracking

### 3. Maintenance Tasks

#### Regular Maintenance
- Database cleanup and optimization
- Log rotation and cleanup
- Security updates and patches

#### Data Management
- Duplicate detection and cleanup
- Data quality monitoring
- Backup verification

## Future Enhancements

### 1. Planned Features
- Advanced analytics and insights
- Export functionality (PDF, CSV, JSON)
- User authentication and authorization
- Collaborative editing capabilities

### 2. Scalability Improvements
- Horizontal scaling support
- Microservices architecture
- Cloud deployment options

### 3. Integration Opportunities
- External database connectors
- Additional LLM providers
- Research database integrations

---

This specification provides a comprehensive overview of the ConstructGraph system, covering both technical implementation details and user-facing features. The modular architecture allows for future enhancements while maintaining system stability and performance.
