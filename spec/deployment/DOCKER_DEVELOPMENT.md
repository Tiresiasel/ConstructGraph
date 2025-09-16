# Docker 开发环境设置

本项目提供了两种Docker运行模式：开发模式和生产模式。

## 🚀 快速开始

### 1. 设置环境变量
```bash
export OPENAI_API_KEY="your-openai-api-key-here"
```

### 2. 选择运行模式

#### 开发模式（推荐用于开发）
```bash
./dev.sh
```
**特点：**
- ✅ 实时监控代码变化
- ✅ 自动重新加载
- ✅ 暴露数据库端口（方便调试）
- ✅ Flask调试模式
- ✅ 热重载

#### 生产模式
```bash
./prod.sh
```
**特点：**
- ✅ 性能优化
- ✅ 数据库端口内部化
- ✅ 适合生产部署
- ❌ 代码修改需要手动重启

## 🔧 开发模式详解

### 启动命令
```bash
docker compose -f docker-compose.dev.yml up -d
```

### 关键特性
1. **实时代码同步**
   - `./src` 目录挂载为读写模式
   - 本地修改立即同步到容器
   - Flask自动检测文件变化

2. **Flask开发服务器**
   - 启用 `--reload` 和 `--debug` 模式
   - 代码变化自动重新加载
   - 详细的错误信息和调试界面

3. **数据库端口暴露**
   - Neo4j: http://localhost:7474
   - Qdrant: http://localhost:6333
   - 方便直接访问和调试

### 开发工作流
1. 启动开发模式：`./dev.sh`
2. 修改 `src/` 目录下的代码
3. 保存文件后，Flask自动重新加载
4. 刷新浏览器查看变化

## 🏭 生产模式详解

### 启动命令
```bash
docker compose up -d
```

### 关键特性
1. **性能优化**
   - 使用生产级Flask服务器
   - 数据库端口内部化
   - 优化的容器配置

2. **安全性**
   - 数据库端口不暴露到主机
   - 只暴露必要的API端口

## 📁 文件结构

```
ConstructGraph/
├── docker-compose.yml          # 生产模式配置
├── docker-compose.dev.yml      # 开发模式配置
├── dev.sh                      # 开发模式启动脚本
├── prod.sh                     # 生产模式启动脚本
├── src/                        # 源代码目录
│   ├── server/
│   │   └── app.py             # Flask应用
│   └── construct_graph/
└── data/
    └── input/                  # PDF输入目录
```

## 🔍 常用命令

### 开发模式
```bash
# 启动开发环境
./dev.sh

# 查看实时日志
docker logs -f construct-graph-api-1

# 停止开发环境
docker compose -f docker-compose.dev.yml down

# 重启API服务
docker compose -f docker-compose.dev.yml restart api
```

### 生产模式
```bash
# 启动生产环境
./prod.sh

# 查看日志
docker compose logs -f api

# 停止生产环境
docker compose down

# 重启服务
docker compose restart api
```

### 通用命令
```bash
# 查看所有容器状态
docker compose ps

# 查看容器资源使用
docker stats

# 进入API容器
docker exec -it construct-graph-api-1 bash

# 查看容器文件系统
docker exec construct-graph-api-1 ls -la /app/src
```

## 🐛 故障排除

### 常见问题

1. **端口冲突**
   ```bash
   # 检查端口占用
   lsof -i :5050
   
   # 停止冲突的服务
   docker compose down
   ```

2. **代码修改不生效**
   ```bash
   # 检查开发模式是否启动
   docker compose -f docker-compose.dev.yml ps
   
   # 手动重启API服务
   docker compose -f docker-compose.dev.yml restart api
   ```

3. **数据库连接失败**
   ```bash
   # 检查数据库状态
   docker compose ps neo4j qdrant
   
   # 查看数据库日志
   docker compose logs neo4j
   docker compose logs qdrant
   ```

### 日志查看
```bash
# 实时查看API日志
docker logs -f construct-graph-api-1

# 查看特定时间段的日志
docker logs --since="2025-08-26T10:00:00" construct-graph-api-1

# 查看错误日志
docker logs construct-graph-api-1 2>&1 | grep ERROR
```

## 💡 最佳实践

1. **开发时使用开发模式**
   - 实时代码同步
   - 快速迭代
   - 详细调试信息

2. **部署时使用生产模式**
   - 性能优化
   - 安全性提升
   - 稳定运行

3. **定期清理**
   ```bash
   # 清理未使用的镜像和容器
   docker system prune -f
   
   # 清理未使用的卷
   docker volume prune -f
   ```

4. **环境变量管理**
   - 使用 `.env` 文件管理敏感信息
   - 不要将API密钥提交到版本控制
   - 为不同环境设置不同的配置

## 🔄 模式切换

### 从开发模式切换到生产模式
```bash
# 停止开发模式
docker compose -f docker-compose.dev.yml down

# 启动生产模式
./prod.sh
```

### 从生产模式切换到开发模式
```bash
# 停止生产模式
docker compose down

# 启动开发模式
./dev.sh
```

## 📚 更多信息

- [Docker Compose 官方文档](https://docs.docker.com/compose/)
- [Flask 开发服务器](https://flask.palletsprojects.com/en/2.3.x/server/)
- [Neo4j Docker 指南](https://neo4j.com/docs/operations-manual/current/docker/)
- [Qdrant Docker 指南](https://qdrant.tech/documentation/guides/installation/)
