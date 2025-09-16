"""Central configuration for ConstructGraph.

Avoids global constants scattered across scripts. Import from this module.
"""

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Neo4jConfig:
    uri: str = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    user: str = os.getenv('NEO4J_USER', 'neo4j')
    password: str = os.getenv('NEO4J_PASSWORD', '123456')


@dataclass(frozen=True)
class QdrantConfig:
    host: str = os.getenv('QDRANT_HOST', 'localhost')
    port: int = int(os.getenv('QDRANT_PORT', '6333'))
    collection: str = os.getenv('QDRANT_COLLECTION', 'construct_definitions')


@dataclass(frozen=True)
class AppConfig:
    output_html: str = os.getenv('OUTPUT_HTML_FILE', 'dist/index.html')
    input_dir: str = os.getenv('INPUT_DIR', 'data/input')
    output_dir: str = os.getenv('OUTPUT_DIR', 'dist')
    neo4j: Neo4jConfig = Neo4jConfig()
    qdrant: QdrantConfig = QdrantConfig()


CONFIG = AppConfig()


