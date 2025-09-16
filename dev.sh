#!/bin/bash

# 开发模式启动脚本 - 实时监控代码变化
echo "🚀 启动开发模式..."

# 检查环境变量
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ 错误: 请设置 OPENAI_API_KEY 环境变量"
    echo "   例如: export OPENAI_API_KEY='your-api-key-here'"
    exit 1
fi

# 停止现有的生产容器
echo "🛑 停止生产容器..."
docker compose down

# 启动开发模式
echo "🔧 启动开发模式容器..."
docker compose -f docker-compose.dev.yml up -d neo4j qdrant

# 等待数据库启动
echo "⏳ 等待数据库启动..."
sleep 10

# 启动API服务（开发模式）
echo "🚀 启动API服务（开发模式）..."
docker compose -f docker-compose.dev.yml up -d api

echo ""
echo "✅ 开发模式启动完成！"
echo ""
echo "📱 访问地址:"
echo "   - 前端页面: http://localhost:5050"
echo "   - Neo4j浏览器: http://localhost:7474"
echo "   - Qdrant管理: http://localhost:6333"
echo ""
echo "🔍 实时监控:"
echo "   - 代码变化会自动重新加载"
echo "   - 查看日志: docker logs -f construct-graph-api-1"
echo "   - 停止服务: docker compose -f docker-compose.dev.yml down"
echo ""
echo "💡 提示: 修改 src/ 目录下的任何文件都会自动重新加载！"
