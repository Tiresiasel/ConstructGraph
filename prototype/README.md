# Enhanced Academic Knowledge Graph System

This system implements the Technical Blueprint for an Academic Knowledge Graph Prototype with advanced features including entity resolution, vector database integration, and support for diverse research methodologies.

## Features

### 🚀 Enhanced LLM Extraction
- **Multi-methodology support**: Quantitative, Qualitative, Conceptual, Review, Meta-Analysis, Mixed-Methods
- **Relationship-first approach**: Identifies constructs through relationship claims
- **Comprehensive coverage**: Captures definitions, dimensions, boundary conditions, and replication outcomes
- **Advanced relationship characterization**: Evidence type, causality validation, meta-analysis identification
- **Clean construct names**: Automatically removes abbreviations in parentheses (e.g., "organizational commitment" not "organizational commitment (OC)")

### 🔍 Intelligent Entity Resolution
- **Two-stage workflow**: Vector similarity search + LLM semantic adjudication
- **Similarity relationship creation**: Preserves constructs and links them with semantic relationships
- **Canonical vs. variant constructs**: Primary constructs and their variant forms
- **Qdrant vector database**: Efficient semantic search using embeddings
- **Provisional to verified workflow**: Ensures data quality and consistency

### 🗄️ Enhanced Graph Schema
- **Relationship reification**: Treats relationships as first-class entities
- **Construct dimensions**: Hierarchical relationships between constructs
- **Rich metadata**: Research type, replication status, boundary conditions
- **Citation tracking**: Paper-to-paper relationships

## Prerequisites

### 1. Neo4j Database
```bash
# Install Neo4j Desktop or use Docker
docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest
```

### 2. Qdrant Vector Database
```bash
# Using Docker (as mentioned in your setup)
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

### 3. Python Dependencies
```bash
pip install -r requirements.txt
```

### 4. OpenAI API Key
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Configuration

Update the configuration in `build_graph.py`:

```python
# Neo4j configuration
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"  # Change to your password

# Qdrant configuration
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_COLLECTION_NAME = "construct_definitions"

# PDF folder path
PDF_FOLDER = Path("/path/to/your/papers")
```

## Usage

### 1. Build the Knowledge Graph

```bash
cd scripts
python build_graph.py
```

This will:
1. **Extract text** from PDF files
2. **Process with LLM** using the enhanced blueprint schema
3. **Populate Neo4j** with provisional constructs
4. **Run entity resolution** to merge similar constructs
5. **Store embeddings** in Qdrant for future searches

### 2. Visualize the Network

```bash
python visualize_graph.py
```

This generates an interactive HTML visualization with:
- **3-panel layout**: Papers, Network, Details
- **Advanced filtering**: Research type, evidence type, causality
- **Rich tooltips**: Definitions, dimensions, relationships
- **Timeline controls**: Year-based filtering and animation

### 3. Prompt Lab (快速人工对比 PDF 与 Prompt 输出)

用于快速测试不同的 Prompt 与输入 PDF，并将输入/输出都保存在本地便于人工比对与迭代。

目录结构：

```
scripts/prompt_lab/
  inputs/     # 你要测试的 PDF 可以放这里（也可用绝对路径）
  prompts/    # Prompt 模板（示例：sample_prompt.txt）
  outputs/    # 每次运行会生成 run_时间戳 的输出文件夹
  tmp/        # 中间文件，例如抽取的 PDF 文本
  configs/    # 可选：模型与输出配置示例
  run_prompt_lab.py  # CLI 工具
```

安装依赖：

```bash
pip install -r scripts/requirements.txt
```

运行示例：

```bash
python scripts/prompt_lab/run_prompt_lab.py \
  --pdf "scripts/sample/Strategic Management Journal - 2001 - Park - Guanxi and organizational dynamics  organizational networking in Chinese firms.pdf" \
  --prompt scripts/prompt_lab/prompts/sample_prompt.txt \
  --model gpt-5 \
  --json-output
```

运行后会在 `scripts/prompt_lab/outputs/run_YYYYmmdd_HHMMSS/` 下生成：
- `prompt_template.txt`: 本次使用的 Prompt 模板
- `pdf_text.txt`: 从 PDF 抽取的纯文本
- `user_prompt_final.txt`: 实际发送给模型的 User 消息
- `output.json` 或 `output.txt`: 模型主输出（优先保存为 JSON）
- `response_raw.json`: 完整原始响应（调试用）
- `meta.json`: 本次运行的元信息（模型、时间戳等）

提示：
- 如果你的 Prompt 模板包含 `<PAPER_TEXT>` 占位符，将被替换为 PDF 文本；否则系统会直接把 PDF 文本追加到模板后面。
- 设置 `OPENAI_API_KEY` 环境变量后再运行。

## Data Flow

```
PDF Papers → LLM Extraction → Provisional Graph → Entity Resolution → Final Graph
     ↓              ↓              ↓                    ↓              ↓
  Text Data    Structured    Neo4j + Qdrant      LLM + Vector    Unified
                Knowledge      Storage            Similarity      Knowledge
```

## Schema Overview

### Core Nodes
- **Paper**: Academic publications with metadata
- **CanonicalConstruct**: Normalized academic concepts
- **Term**: Paper-specific vocabulary
- **Definition**: Construct definitions with context
- **RelationshipInstance**: Reified relationship claims
- **Theory**: Theoretical foundations
- **Measurement**: Operationalization methods

### Key Relationships
- **ESTABLISHES**: Paper → RelationshipInstance
- **HAS_SUBJECT/OBJECT**: RelationshipInstance → CanonicalConstruct
- **IS_REPRESENTATION_OF**: Term → CanonicalConstruct
- **HAS_DIMENSION**: Parent → Child constructs
- **APPLIES_THEORY**: RelationshipInstance → Theory

## Advanced Queries

### Quantitative Studies with Agency Theory
```cypher
MATCH (theory:Theory {name: 'Agency Theory'})
MATCH (paper:Paper {research_type: 'Quantitative'})
MATCH (paper)-[:ESTABLISHES]->(ri:RelationshipInstance)-[:APPLIES_THEORY]->(theory)
WHERE ri.evidence_type = 'Quantitative' AND ri.effect_direction = 'Negative'
RETURN paper.title, ri.statistical_details
```

### Qualitative Findings
```cypher
MATCH (ri:RelationshipInstance)
WHERE ri.evidence_type = 'Qualitative'
RETURN ri.qualitative_finding, ri.supporting_quote
```

### Construct Dimensions
```cypher
MATCH (parent:CanonicalConstruct)-[:HAS_DIMENSION]->(dim:CanonicalConstruct)
WHERE parent.preferred_name = 'Organizational Justice'
RETURN collect(dim.preferred_name) as dimensions
```

### Similar Constructs (NEW)
```cypher
MATCH (main:CanonicalConstruct {preferred_name: 'Organizational Commitment'})
MATCH (main)-[r:IS_SIMILAR_TO]->(similar:CanonicalConstruct)
WHERE r.similarity_score > 0.8
RETURN similar.preferred_name, r.similarity_score, r.llm_confidence
ORDER BY r.similarity_score DESC
```

### Canonical vs. Variant Constructs (NEW)
```cypher
MATCH (primary:CanonicalConstruct {canonical_status: 'primary'})
OPTIONAL MATCH (primary)-[r:IS_SIMILAR_TO]->(variant:CanonicalConstruct {canonical_status: 'variant'})
RETURN primary.preferred_name, 
       collect({variant: variant.preferred_name, similarity: r.similarity_score}) AS variants
```

## Troubleshooting

### Common Issues

1. **Neo4j Connection Error**
   - Check if Neo4j is running
   - Verify password in configuration
   - Ensure port 7687 is accessible

2. **Qdrant Connection Error**
   - Check if Qdrant Docker container is running
   - Verify ports 6333 and 6334 are accessible
   - Check collection creation permissions

3. **OpenAI API Error**
   - Verify API key is set correctly
   - Check API quota and billing
   - Ensure model name is correct (gpt-5o-mini)

4. **Memory Issues**
   - Reduce PDF text length in LLM prompt
   - Process papers in smaller batches
   - Monitor system memory usage

### Performance Tips

1. **Vector Database**: Use SSD storage for Qdrant
2. **Neo4j**: Configure appropriate memory settings
3. **LLM Processing**: Process papers sequentially to avoid rate limits
4. **Embeddings**: Cache sentence transformer model

## Extending the System

### Adding New Research Types
1. Update the LLM prompt schema
2. Add new properties to RelationshipInstance nodes
3. Extend visualization filters

### Custom Entity Resolution
1. Modify similarity threshold
2. Customize LLM adjudication prompt
3. Add domain-specific rules

### New Visualization Features
1. Add new filter categories
2. Implement custom layouts
3. Add export functionality

## Contributing

1. Follow the blueprint schema strictly
2. Test with diverse paper types
3. Maintain backward compatibility
4. Document new features

## License

This project implements the Technical Blueprint for Academic Knowledge Graph Prototype.
