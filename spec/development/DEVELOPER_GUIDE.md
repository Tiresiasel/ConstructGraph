# Developer Guide (src)

This document describes the codebase architecture, data model, and workflows that power ConstructGraph.

## High-level architecture
- Ingestion (build): parse PDFs, extract constructs/relationships, upsert to Neo4j; store embeddings and support entity resolution in Qdrant.
- API (Flask): expose realtime read/write endpoints for constructs and relationships（CORS enabled）。
- Visualization (runtime): the template fetches constructs/relationships/papers from the API at runtime (fallback to embedded data if API unavailable) and renders an interactive HTML via vis-network.
- Prompt Lab (optional): experiment with prompts and analyze extraction quality.

## Directory layout
```
src/
  construct_graph/
    config.py            # Central configuration (env-driven)
    db/neo4j.py          # Neo4j connection factory
    data/fetchers.py     # Cypher queries & normalization helpers
    models.py            # Typed dataclasses (Construct/Relationship/...)
    render/
      templates/constructs_network.html.j2  # Jinja2 template
      page.py            # Rendering helpers
    cli.py               # CLI entry: build / visualize

  build_graph.py         # Ingestion pipeline (to be further modularized)
  visualize_graph.py     # Legacy script using the modules above
  prompt_lab/            # Prompt experimentation utilities
```

## Data model (simplified)
- Construct: `name`, `description`, `dimensions[]`, `parent_constructs[]`, `paper_ids[]`, `best_description`.
- Relationship: `source_construct`, `target_construct`, `effect_direction`, `relationship_instances[]`.
- RelationshipInstance: `status`, `evidence_type`, `effect_direction`, `non_linear_type`, `theories[]`, `moderators[]`, `mediators[]`, `paper_uid`, `paper_year`.
- Paper (list for UI filters): `id`, `title`, `authors`, `year`, `journal`, `research_type`, `research_context`.

## 数据库与数据模型（详细规范）

本节系统性定义了系统中“存储的所有信息”的类别、字段、取值域、它们之间的关系与含义、软删除/版本化/审计策略，以及变量合并（实体解析）。该规范面向实现与扩展，作为上线后的稳定版（仅做前向兼容的增量，避免破坏式变更）。

### 图数据库（Neo4j）实体与属性

- Paper（论文）
  - 唯一性与键：`paper_uid`（由标题+年份规范化后哈希生成）
  - 属性：
    - `uuid`: string（系统生成）
    - `paper_uid`: string（唯一）
    - `title`: string
    - `doi`: string（去除 http(s):// 前缀的紧凑形式）
    - `authors`: array<string>
    - `publication_year`: number | null
    - `journal`: string | null
    - `research_type`: enum{"Quantitative","Qualitative","Conceptual","Review","Meta-Analysis","Mixed-Methods"} | null
    - `research_context`: string | null
    - `is_replication_study`: boolean（默认 false）
    - `filename`: string（来源文件名）
    - `created_at`: ISO datetime
    - `updated_at`: ISO datetime

- Author（作者）
  - 唯一性与键：`full_name`
  - 属性：
    - `uuid`: string（系统生成）
    - `full_name`: string

- Term（术语表示）
  - 唯一性与键：`text`（小写、去除括注、规范化）
  - 属性：
    - `uuid`: string
    - `text`: string（术语的“表面形态”，如同义词、别名等，1:N 映射到 CanonicalConstruct）

- Definition（定义）
  - 唯一性与键：三元组(`text`, `term_text`, `paper_uid`)
  - 属性：
    - `uuid`: string
    - `text`: string | null（完整理论定义，允许为空以占位）
    - `context_snippet`: string | null（上下文片段）
    - `term_text`: string（对应 Term.text）
    - `paper_uid`: string（对应 Paper.paper_uid）

- CanonicalConstruct（规范构念）
  - 唯一性与键：`preferred_name`（小写、最小充分语义名称）
  - 属性：
    - `uuid`: string
    - `preferred_name`: string
    - `status`: enum{"Provisional","Verified"}（导入时缺省为 Provisional；解析后置为 Verified）
    - `description`: string | null（最佳可用定义/长描述）
    - `canonical_status`: string | null（如 "primary" 等标识，可用于后续标注）

- Measurement（测量/操作化）
  - 唯一性与键：三元组(`name`, `construct_term`, `paper_uid`)
  - 属性：
    - `uuid`: string
    - `name`: string（操作化标签/量表名/指数名）
    - `description`: string（how-to 细节）
    - `instrument`: string | null
    - `scale_items`: string | json-string | null（当为数组或对象时以 json 字符串持久化）
    - `scoring_procedure`: string | null
    - `formula`: string | null（数式以 $$...$$ 格式）
    - `reliability`: string | null
    - `validity`: string | null
    - `context_adaptations`: string | null
    - `construct_term`: string（对应 CanonicalConstruct.preferred_name）
    - `paper_uid`: string

- Theory（理论）
  - 唯一性与键：`canonical_id`（由显示名规范化生成的 snake_case）
  - 属性：
    - `uuid`: string
    - `canonical_id`: string（唯一）
    - `name`: string（显示名，保留最长/最具信息量）
    - `created_at`: datetime

- RelationshipInstance（关系实例）
  - 唯一性与键：`uuid`
  - 属性：
    - `uuid`: string
    - `description`: string（自动生成的关系描述）
    - `context_snippet`: string | null
    - `status`: enum{"Hypothesized","Empirical_Result"}
    - `evidence_type`: enum{"Quantitative","Qualitative","Formal_Model","Simulation","Mixed"} | null
    - `effect_direction`: enum{"Positive","Negative","Insignificant","Mixed"} | null
    - `non_linear_type`: enum{"U-shaped","Inverted_U-shaped","S-shaped","Other"} | null
    - `is_validated_causality`: boolean（是否满足可信对因识别）
    - `is_meta_analysis`: boolean
    - `qualitative_finding`: string | null
    - `supporting_quote`: string | null
    - `boundary_conditions`: string | null
    - `replication_outcome`: enum{"Successful","Failed","Mixed"} | null

### 图数据库（Neo4j）关系类型、基数与语义

- (Paper)-[:AUTHORED_BY]->(Author)
  - 多对多：一文多作，作者可复用。

- (Paper)-[:USES_TERM]->(Term)
  - Paper 使用的术语（溯源）。

- (Term)-[:HAS_DEFINITION]->(Definition)
  - 同一术语在同一论文内可有一条主定义（去重规则见 Definition 键）。

- (Term)-[:IS_REPRESENTATION_OF]->(CanonicalConstruct)
  - 多对一：多个术语表示聚合到同一规范构念。

- (Definition)-[:DEFINED_IN]->(Paper)
  - 定义的来源论文。

- (CanonicalConstruct)-[:USES_MEASUREMENT]->(Measurement)
  - 构念关联的操作化方案。

- (Measurement)-[:MEASURED_IN]->(Paper)
  - 操作化方案的来源论文。

- (parent:CanonicalConstruct)-[:HAS_DIMENSION]->(dim:CanonicalConstruct)
  - 维度/层级关系：父 → 子维度。

- (Paper)-[:ESTABLISHES]->(RelationshipInstance)
  - 该论文建立（提出/检验）的一条关系实例。

- (RelationshipInstance)-[:HAS_SUBJECT]->(CanonicalConstruct)
- (RelationshipInstance)-[:HAS_OBJECT]->(CanonicalConstruct)
  - 每个关系实例恰有一个 subject 与一个 object（都指向规范构念）。

- (RelationshipInstance)-[:APPLIES_THEORY]->(Theory)
  - 关系实例明确援引/基于的理论。

- (RelationshipInstance)-[:HAS_MODERATOR]->(CanonicalConstruct)
- (RelationshipInstance)-[:HAS_MEDIATOR]->(CanonicalConstruct)
- (RelationshipInstance)-[:HAS_CONTROL]->(CanonicalConstruct)
  - 关系实例中扮演调节/中介/理论上重要控制变量的构念。

- (a:CanonicalConstruct)-[:IS_SIMILAR_TO {similarity_score, llm_confidence, relationship_type}]->(b:CanonicalConstruct)
  - 语义相似但未合并的双向链接；`relationship_type` 常用值："synonym"。

### 值域与取值规范（关键字段）

- RelationshipInstance.status: {"Hypothesized","Empirical_Result"}
- RelationshipInstance.evidence_type: {"Quantitative","Qualitative","Formal_Model","Simulation","Mixed"} | null
- RelationshipInstance.effect_direction: {"Positive","Negative","Insignificant","Mixed"} | null
- RelationshipInstance.non_linear_type: {"U-shaped","Inverted_U-shaped","S-shaped","Other"} | null
- RelationshipInstance.replication_outcome: {"Successful","Failed","Mixed"} | null
- CanonicalConstruct.status: {"Provisional","Verified"}
- Paper.research_type: {"Quantitative","Qualitative","Conceptual","Review","Meta-Analysis","Mixed-Methods"}
- Measurement.formula: string | null（数式必须在 $$...$$ 内）

说明：未列出的 string/boolean/number 字段遵循直观含义；数组字段均为去重聚合后的语义列表。

### 向量数据库（Qdrant）集合与向量规范

- Constructs 集合（1024 维，COSINE 距离）
  - 向量：构念定义文本嵌入
  - Payload：
    - `term`: string（规范化后的构念名）
    - `definition`: string（完整定义）
    - `paper`: string（来源文件名或标识）

- Theories 集合（1024 维，COSINE 距离）
  - 向量：理论名称 + 可选上下文的嵌入
  - Payload：
    - `canonical_id`: string（理论唯一键）
    - `name`: string（理论显示名）
    - `context`: string（用于判别的上下文片段）

### 工作流细化与约束

1) 导入/抽取（Ingestion）
   - 依据结构化提取结果创建/合并节点与关系。
   - 去重键：Paper(paper_uid)、Author(full_name)、Term(text)、Definition(text,term_text,paper_uid)、CanonicalConstruct(preferred_name)、Measurement(name,construct_term,paper_uid)、Theory(canonical_id)、RelationshipInstance(uuid)。
   - 测量（Measurement）仅对“出现在关系中的变量（subject/object/调节/中介/控制）”进行创建与关联。

2) 关系刻画（RelationshipInstance）
   - 每条实例固定一对 subject/object；可选调节/中介/控制；可选理论；字段遵循上文值域。

3) 定义/量表规范
   - 定义与量表的公式与统计符号必须使用 $$...$$ 数学格式；量表条目可序列化为 JSON 字符串。

### 变量合并（实体解析）策略（重点，支持可逆回滚）

目标：对“语义相同或高度相近”的构念统一到同一 CanonicalConstruct，同时最大化保留信息。

步骤（双阶段）：
1) 候选检索（向量相似度门槛）
   - 在 Constructs 向量集合中以新定义的嵌入检索候选。
   - 门槛 SIMILARITY_THRESHOLD = 0.80（得分低于该值的候选忽略）。

2) 候选裁决（LLM 裁决 + 置信度）
   - 对得分通过门槛的候选，使用 LLM 对“是否为同一概念”进行裁决，得到 (is_same, confidence)。
   - 合并阈值 MERGE_CONFIDENCE = 0.95。

合并与保留策略：
- 若 is_same 且 confidence ≥ 0.95（软合并，可逆）：
  - 选择更短的名称作为 canonical `preferred_name`（shorter_name），另一侧记为 drop（别名）。
  - 记录操作：创建 `MergeOperation` 节点，并逐条记录 `EdgeRewire`（每条被重接线的边都会被记录）。
  - 关系重接线（rewire）：
    - (ri)-[:HAS_SUBJECT|HAS_OBJECT|HAS_MODERATOR|HAS_MEDIATOR|HAS_CONTROL]->(drop)
      → MERGE 到 (keep)，并删除旧边；同时写入 `EdgeRewire` 日志。
    - (t:Term)-[:IS_REPRESENTATION_OF]->(drop) → 迁移至 keep，并写日志。
    - (drop)-[:USES_MEASUREMENT]->(m) → 迁移到 keep，并写日志。
  - 标记 drop：`drop.active=false`，并新增 `(:CanonicalConstruct {drop})-[:ALIAS_OF {active:true, merge_operation_id}]->(:CanonicalConstruct {keep})`。
  - 描述迁移：若 keep.description 为空或更短，则覆盖为 drop.description。
  - 不删除任何节点，保持回滚可能；`keep.status` 可置为 `Verified`。

- 若 is_same 且 confidence < 0.95（仅相似）：
  - 保留二者，创建双向 `:IS_SIMILAR_TO`，并记录 `{similarity_score, llm_confidence, relationship_type: 'synonym'}`。

- 若未找到任何相似候选：
  - 将该新构念的 `status` 由 `Provisional` 置为 `Verified`，并可将 `canonical_status` 置为 `primary`。

规范化与清洗规则（输入侧）：
- 构念/理论名移除括号中的缩写或附注（包括 () 与 []）。
- 统一小写；去除冗余停用词；常见“X of Y”化为“Y X”等一致化处理。
- 测量名去括注；公式必须为 $$...$$。

风险与边界：
- 合并操作可逆：通过 `MergeOperation` + `EdgeRewire` 的完整日志可回放恢复。
- 低置信度使用 `IS_SIMILAR_TO`，避免误合并；人工确认后再执行合并。
- 合并仅影响 CanonicalConstruct 节点与其关联边，不修改 Definition 与 Measurement 的来源溯源（仍保留 Paper 连接）。

### 版本化与回滚

- ConstructRevision（新）
  - 记录构念 `description` 的每次修改：`revision_id`, `description`, `created_at`, `editor`, `reason`。
  - `(:CanonicalConstruct)-[:HAS_REVISION]->(:ConstructRevision)`；当前生效描述保留在 `CanonicalConstruct.description`。

- MergeOperation / EdgeRewire（新）
  - `MergeOperation`：记录一次软合并的元数据（`id`, `created_at`, `initiator`, `reason`, `similarity_score`, `llm_confidence`, `status:{applied,rolled_back}`）。
  - `EdgeRewire`：逐条记录本次合并中修改的边（`edge_type`, `from_node_id`, `to_node_id`, `prev_target_id`, `new_target_id`, 关联 `ri_uuid` 或 measurement 键），并与 `MergeOperation` 关联。

- 回滚 `rollback_merge(merge_operation_id)`：
  - 读取 `EdgeRewire` 清单逐条反向恢复；
  - 还原 `drop.active=true`，并将 `ALIAS_OF.active=false`；
  - 标记 `MergeOperation.status=rolled_back`。

### 后端接口（仅后端，不直接暴露前端）

- `update_construct_description(construct_id, new_description, options) -> ChangeSet`
  - 版本化描述 → 重嵌入 → 重解析 →（可选）自动合并/刷新相似链接。

- `reembed_construct(construct_id, options) -> EmbeddingRecord`
  - 使用当前描述生成新向量并 upsert 到 Qdrant（按 `construct_id + revision_id` 唯一）。

- `re_resolve_construct(construct_id, options) -> ResolutionReport`
  - 基于相似度门槛（默认 0.80）与裁决模型返回候选、分数与建议。

- `propose_merge(keep_id, drop_id, rationale) -> MergeProposal`
  - 准备一条软合并提案，包含候选对、相似度/置信度与理由。

- `apply_merge(merge_proposal_id) -> MergeOperation`
  - 执行软合并：建 `MergeOperation`、`EdgeRewire`，重接线，标记 alias。

- `rollback_merge(merge_operation_id) -> RollbackReport`
  - 基于 `EdgeRewire` 逐条反向恢复。

- `recompute_similarity_links(construct_id, options) -> SimilarityUpdateReport`
  - 仅刷新 `IS_SIMILAR_TO` 边，不做合并。

## Workflows
### Build (ingestion)
1. Load PDFs from `CONFIG.input_dir` (default `data/input`).
2. Extract text → detect constructs and relationships.
3. Upsert nodes/edges to Neo4j; write embeddings to Qdrant for entity resolution.
4. Track processed files in `processed_files.json` to allow incremental runs.

### Visualize (runtime fetch)
1. Frontend loads and calls API endpoints at runtime to get `constructs`/`relationships`/`papers`.
2. Compute layouts in the client (existing logic preserved) and render via vis-network.
3. If API is unreachable, the page falls back to embedded JSON baked in during render.

## Config & environment
- Environment variables (see root README): `NEO4J_*`, `QDRANT_*`, `INPUT_DIR`, `OUTPUT_DIR`, `OUTPUT_HTML_FILE`.
- All configuration is read via `construct_graph.config.CONFIG`.
 - API configuration (runtime): `FLASK_HOST` (default `0.0.0.0`), `FLASK_PORT` (default `5050`). Frontend will use `window.API_BASE` or `localStorage.API_BASE` or fallback to `//<host>:5050`.

## API (Flask) overview

- Health
  - `GET /api/health`

- Read
  - `GET /api/constructs`（默认只返回 `active=true` 的规范构念）
  - `GET /api/relationships`（默认只返回 `ri.active=true` 的关系实例）
  - `GET /api/papers`

- Construct ops
  - `PATCH /api/constructs/{name}/description`（版本化描述→重嵌入→重解析→可选自动合并；可回滚）
  - `POST /api/constructs/merge`（软合并，可回滚）
  - `POST /api/constructs/rollback-merge`（回滚某次合并）
  - `POST /api/constructs`（新增构念：必填 `term`、`definition`；可选 `paper_uid`、`measurements[]`；自动写入向量库并按阈值尝试软合并）

- Relationship ops
  - `PATCH /api/relationships/{ri_uuid}`（更新关系属性与两端/参与变量；审计+可回滚）
  - `POST /api/relationships/{ri_uuid}/soft-delete`、`POST /api/relationships/{ri_uuid}/restore`
  - `POST /api/relationships/rollback-operation`
  - `POST /api/relationships`（新增关系：必填 `subject`、`object`、`status`；可选 `evidence_type`、`effect_direction`、`non_linear_type`、`moderators[]`、`mediators[]`、`controls[]`、`theories[]`、`paper_uid`）

前端运行时数据获取：
- 模板 `constructs_network.html.j2` 中通过 `loadDataRuntime()` 依次请求 `/api/constructs`、`/api/relationships`、`/api/papers`，加载失败则自动回退到页面内嵌 JSON。
- 可通过浏览器控制台设置 `localStorage.setItem('API_BASE','http://your-host:5050')` 来指定 API 地址。

## Run

- Local (dev):
  - Backend API: `python src/server/app.py`（默认为 `0.0.0.0:5050`）
  - Neo4j/Qdrant: 可使用 docker-compose（见下）或本地服务
  - 打开生成的 `dist/index.html`（页面会请求 API 获取数据）

- Docker Compose:
  - `docker compose up -d` 将启动 `neo4j`、`qdrant`、`api`（Flask）等服务；API 暴露在 `:5050`
  - 通过环境变量可覆盖连接参数（compose 已为 `api` 注入默认 `NEO4J_URI=bolt://neo4j:7687`、`QDRANT_HOST=qdrant` 等）。

## Coding conventions
- All comments and docs in English.
- Modules follow clear responsibilities; avoid monolithic functions.
- Prefer pure functions with typed inputs/outputs.

## Roadmap for further modularization
- Extract `build_graph.py` into `construct_graph/build/` (readers, parsers, resolvers, writers).
- Add `layout/` module for positioning strategies.
- Add `utils/` for serialization, logging setup, and common helpers.

## Testing (suggested)
- Unit-test prompt parsers, Cypher builders, and layout functions.
- Snapshot-test template rendering with small datasets.

## API Documentation

For detailed API reference, see [API_REFERENCE.md](../api/API_REFERENCE.md) in the spec/api directory.

