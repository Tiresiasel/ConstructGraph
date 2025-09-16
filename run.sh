#!/bin/bash

# Local run script for ConstructGraph
# This script runs the Python application locally while connecting to Docker database services

echo "🚀 Starting ConstructGraph locally..."

# Check if .env file exists - this is required for configuration
if [ ! -f ".env" ]; then
    echo "❌ .env file not found!"
    echo ""
    echo "📋 Please configure your environment first:"
    echo "   1. Copy the example file: cp .env.example .env"
    echo "   2. Edit .env with your actual values"
    echo "   3. Run this script again"
    echo ""
    echo "💡 Required configuration:"
    echo "   - Database credentials (Neo4j, Qdrant)"
    echo "   - OpenAI API key"
    echo "   - Input/output folder paths"
    echo ""
    exit 1
fi

# Load environment variables from .env file
echo "📋 Loading configuration from .env file..."
export $(cat .env | grep -v '^#' | xargs)
source .env

# Set Python path
export PYTHONPATH="${PWD}/src"

echo "🔧 Environment configuration loaded:"
echo "   PYTHONPATH: $PYTHONPATH"
echo "   NEO4J_URI: ${NEO4J_URI:-'NOT SET'}"
echo "   NEO4J_USER: ${NEO4J_USER:-'NOT SET'}"
echo "   QDRANT_HOST: ${QDRANT_HOST:-'NOT SET'}"
echo "   OPENAI_API_KEY: ${OPENAI_API_KEY:-'NOT SET'}"
echo ""

# Validate required environment variables
if [ -z "$NEO4J_URI" ] || [ -z "$NEO4J_USER" ] || [ -z "$NEO4J_PASSWORD" ]; then
    echo "❌ Missing required database configuration!"
    echo "   Please check your .env file"
    exit 1
fi

if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  Warning: OPENAI_API_KEY not set"
    echo "   AI-powered features may not work"
    echo ""
fi

# Check if Python dependencies are installed
if ! python -c "import neo4j, qdrant_client" 2>/dev/null; then
    echo "📦 Installing Python dependencies..."
    pip install -r src/requirements.txt
fi

# Check if Docker services are running
echo "📊 Checking Docker database services..."

# Check if Neo4j is accessible
if ! nc -z localhost 7687 2>/dev/null; then
    echo "❌ Neo4j service is not accessible on port 7687"
    echo "   Starting Docker services..."
    docker compose up -d
    echo "⏳ Waiting for services to be ready..."
    sleep 15
    
    # Verify services are up
    if ! nc -z localhost 7687 2>/dev/null; then
        echo "❌ Failed to start Neo4j service"
        echo "   Please check Docker logs: docker compose logs neo4j"
        exit 1
    fi
    
    if ! nc -z localhost 6333 2>/dev/null; then
        echo "❌ Failed to start Qdrant service"
        echo "   Please check Docker logs: docker compose logs qdrant"
        exit 1
    fi
    
    echo "✅ Docker services are now running"
else
    echo "✅ Docker services are already running"
fi

# Run the application
echo "🎯 Running ConstructGraph..."
python -m construct_graph.cli build
python -m construct_graph.cli visualize -o dist/index.html

echo "✅ Done! Check dist/index.html for output."
