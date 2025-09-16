#!/bin/bash

# Setup local Python virtual environment for ConstructGraph development

echo "🐍 Setting up local Python environment..."

# Check if Python 3.11+ is available
python_version=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
if [[ "$python_version" < "3.11" ]]; then
    echo "❌ Python 3.11+ is required. Current version: $python_version"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📁 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📦 Installing Python dependencies..."
pip install -r src/requirements.txt

echo "✅ Local environment setup complete!"
echo ""
echo "To activate the environment manually:"
echo "   source venv/bin/activate"
echo ""
echo "To run the application:"
echo "   ./run_local.sh"
echo ""
echo "Or manually:"
echo "   export PYTHONPATH=\${PWD}/src"
echo "   python -m construct_graph.cli visualize -o dist/index.html"
