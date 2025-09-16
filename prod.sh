#!/bin/bash

# 生产模式启动脚本
echo "🚀 启动生产模式..."

# 检查环境变量
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ 错误: 请设置 OPENAI_API_KEY 环境变量"
    echo "   例如: export OPENAI_API_KEY='your-api-key-here'"
    exit 1
fi

# 停止开发模式容器
echo "🛑 停止开发容器..."
docker compose -f docker-compose.dev.yml down

# 启动生产模式
echo "🔧 启动生产模式容器..."
docker compose up -d

echo ""
echo "✅ 生产模式启动完成！"
echo ""
echo "📱 访问地址:"
echo "   - 前端页面: http://localhost:5050"
echo ""
echo "🔍 管理命令:"
echo "   - 查看日志: docker compose logs -f api"
echo "   - 停止服务: docker compose down"
echo "   - 重启服务: docker compose restart api"
echo ""
echo "💡 提示: 生产模式下代码不会自动重新加载，需要手动重启容器"
