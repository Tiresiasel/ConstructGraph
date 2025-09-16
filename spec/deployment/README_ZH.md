<div align="right"><a href="README.md">English</a> | 中文</div>

# ConstructGraph

ConstructGraph 全面容器化：将 PDF 放入宿主机 `data/input/`，启动 Docker，打开浏览器即可访问本地 URL。系统会实时检测新增文件并更新图谱与页面。

## 架构（Docker 优先）

- `api`（Flask）
  - 根路径 `http://localhost:5050` 提供交互式页面（静态 `dist/index.html`）。
  - 暴露 `/api/*` REST 接口（构念、关系、论文、审计等）。
  - 后台轮询 `/app/data/input` 自动摄取新 PDF。
- `neo4j`（仅内部网络）
- `qdrant`（仅内部网络）

对外只暴露 `5050` 端口，数据库不暴露到宿主机，避免冲突。

## 先决条件

- Docker + Docker Compose
- `.env`（OpenAI API Key、数据库口令等）

## 快速开始

1）准备环境
```bash
cp .env.example .env
# 编辑 .env，至少设置 OPENAI_API_KEY、NEO4J_PASSWORD
```

2）放置 PDF
```
data/
  input/
    论文1.pdf
    论文2.pdf
```

3）启动
```bash
docker compose up -d
```

4）访问
```
http://localhost:5050
```

- 页面实时从 API 获取数据。
- 之后将新 PDF 放入 `data/input/`，容器内轮询（默认 5 秒）会自动摄取并更新图谱。

## 摄取机制（不改宿主机文件）

- `./data/input` 以只读方式挂载到容器 `/app/data/input`。
- 轮询扫描 `*.pdf`，计算内容 SHA‑256，并在 Neo4j 查 `(:IngestedFile {sha256})`：
  - 已存在 → 跳过
  - 新文件 → 提取 → 入库 → 实体解析（Qdrant）→ 记录 `IngestedFile`
- 不移动/删除宿主机文件。若要强制重跑，可修改文件内容或清理对应 `IngestedFile` 记录。

可调参数（compose 中）：
- `POLL_ENABLED=true|false`（默认 true）
- `POLL_INTERVAL=5`（秒）
- `INPUT_DIR=/app/data/input`（容器路径）
- `OUTPUT_DIR=/app/dist`（容器路径，服务 index.html）

## API 概览

- 基础：`http://localhost:5050`
- 健康：`GET /api/health`
- 读取：`GET /api/constructs`、`GET /api/relationships`、`GET /api/papers`（分页：`?page=&limit=`）
- 构念：更新描述、软合并/回滚、创建、软删除
- 关系：创建、更新（属性/角色重接线）、软删/恢复、回滚
- 维度/相似：增加/删除子维度；增加/删除 `IS_SIMILAR_TO`
- 测量/定义：POST/PATCH/DELETE（软删除）
- 审计：`GET /api/operations` 操作时间线

根路径 `/` 直接服务 `dist/index.html`，页面在运行时调用 API。

## 配置（.env）

```bash
# 数据库（容器内网络使用）
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=12345678

QDRANT_HOST=qdrant
QDRANT_PORT=6333
QDRANT_COLLECTION=construct_definitions

# 容器内路径
INPUT_DIR=/app/data/input
OUTPUT_DIR=/app/dist

# OpenAI
OPENAI_API_KEY=your_openai_api_key_here
```

也可在 `docker-compose.yml` 中调整 `POLL_ENABLED`、`POLL_INTERVAL`。

## 目录结构

```
ConstructGraph/
  data/
    input/           # 宿主机 PDF 目录（容器内只读）
  dist/              # 静态页面（index.html）由 Flask 提供
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
    server/app.py     # Flask（API + 静态页 + 后台轮询）
  docker-compose.yml  # 所有服务（DB 内部化，API 暴露 5050）
```

## 说明

- 仅 `api` 暴露 `5050`；`neo4j`、`qdrant` 不暴露到宿主机。
- 前端页面完全在容器内运行；无需将 HTML 拷回宿主机。
- Qdrant 客户端可能提示版本差异警告；不影响功能。

## 本地开发（可选）

如需本地 Python 开发，可使用 `./setup.sh`，同时连接 Docker 内数据库。生产使用推荐 Docker 优先。


