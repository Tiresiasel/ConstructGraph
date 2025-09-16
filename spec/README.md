# ConstructGraph Specifications

This directory contains all project specifications, documentation, and guides organized by category.

## 📁 Directory Structure

### 📋 Technical Specifications (`technical/`)
Core technical documentation and system specifications.

- **[SPECIFICATION.md](technical/SPECIFICATION.md)** - Complete technical specification covering:
  - System architecture and components
  - Backend and frontend specifications  
  - Database schemas and data models
  - API endpoints and workflows
  - Deployment and performance specifications

### 🛠️ Development Guides (`development/`)
Developer-focused documentation for codebase understanding and contribution.

- **[DEVELOPER_GUIDE.md](development/DEVELOPER_GUIDE.md)** - Main developer guide covering:
  - High-level architecture overview
  - Data models and workflows
  - Configuration and environment setup
  - Coding conventions and best practices

- **[PACKAGE_GUIDE.md](development/PACKAGE_GUIDE.md)** - Core package documentation covering:
  - Package modules and responsibilities
  - Data contracts and models
  - Frontend modularization details
  - Usage examples and conventions

- **[PROMPT_MANAGEMENT.md](development/PROMPT_MANAGEMENT.md)** - Prompt system documentation covering:
  - Centralized prompt management
  - Prompt loading and fallback mechanisms
  - Best practices for prompt design
  - Integration guidelines

### 🧪 Testing (`testing/`)
Testing strategy, implementation, and quality assurance.

- **[TEST_GUIDE.md](testing/TEST_GUIDE.md)** - Comprehensive testing guide covering:
  - Test directory structure and organization
  - Unit, integration, and e2e testing strategies
  - Test fixtures and environment setup
  - Quality gates and CI/CD considerations

### 🚀 Deployment & Operations (`deployment/`)
Deployment, operations, and infrastructure documentation.

- **[DOCKER_DEVELOPMENT.md](deployment/DOCKER_DEVELOPMENT.md)** - Docker development environment guide covering:
  - Development vs production modes
  - Docker Compose configurations
  - Development workflow and best practices
  - Troubleshooting and maintenance

- **[README_EN.md](deployment/README_EN.md)** - English deployment guide covering:
  - Quick start instructions
  - Architecture overview
  - Configuration options
  - Usage examples

- **[README_ZH.md](deployment/README_ZH.md)** - Chinese deployment guide (中文部署指南)

### 🔌 API Reference (`api/`)
Complete API documentation and reference.

- **[API_REFERENCE.md](api/API_REFERENCE.md)** - Comprehensive API documentation covering:
  - All REST endpoints with examples
  - Request/response schemas
  - Authentication and security
  - Data models and constraints
  - Error handling and status codes

### 🔬 Experimental (`experimental/`)
Experimental features, research, and analysis.

- **[PROMPT_LAB_GUIDE.md](experimental/PROMPT_LAB_GUIDE.md)** - Prompt experimentation framework covering:
  - Prompt Lab structure and usage
  - Experimentation workflows
  - Analysis tools and metrics
  - Best practices for prompt iteration

- **[MODEL_ANALYSIS_REPORT.md](experimental/MODEL_ANALYSIS_REPORT.md)** - AI model consistency analysis report covering:
  - Cross-model output comparison
  - Consistency scoring methodology
  - Recommendations for improvement
  - Performance analysis results

## 🎯 How to Use This Documentation

### For New Contributors
1. Start with [DEVELOPER_GUIDE.md](development/DEVELOPER_GUIDE.md) for architecture overview
2. Read [PACKAGE_GUIDE.md](development/PACKAGE_GUIDE.md) for code structure
3. Follow [TEST_GUIDE.md](testing/TEST_GUIDE.md) for testing practices
4. Use [API_REFERENCE.md](api/API_REFERENCE.md) for API integration

### For System Administrators
1. Begin with [README_EN.md](deployment/README_EN.md) for deployment
2. Reference [DOCKER_DEVELOPMENT.md](deployment/DOCKER_DEVELOPMENT.md) for operations
3. Check [SPECIFICATION.md](technical/SPECIFICATION.md) for system requirements

### For API Users
1. Start with [API_REFERENCE.md](api/API_REFERENCE.md) for complete API docs
2. Reference [SPECIFICATION.md](technical/SPECIFICATION.md) for data models
3. Use [DEVELOPER_GUIDE.md](development/DEVELOPER_GUIDE.md) for integration examples

### For Researchers
1. Explore [PROMPT_LAB_GUIDE.md](experimental/PROMPT_LAB_GUIDE.md) for experimentation
2. Review [MODEL_ANALYSIS_REPORT.md](experimental/MODEL_ANALYSIS_REPORT.md) for analysis results
3. Check [SPECIFICATION.md](technical/SPECIFICATION.md) for technical details

## 📝 Document Maintenance

- **Technical specifications** are stable and should only be updated for major architectural changes
- **Development guides** should be updated as the codebase evolves
- **API reference** must be kept in sync with actual API implementation
- **Deployment guides** should reflect current infrastructure and procedures
- **Experimental docs** are updated as research progresses

## 🔄 Contributing to Documentation

When contributing to documentation:

1. **Update the relevant spec file** in the appropriate category
2. **Update this README** if adding new documents or changing structure
3. **Cross-reference related documents** where appropriate
4. **Test all links and examples** before submitting
5. **Follow the established naming conventions** and directory structure

## 📚 Additional Resources

- [Main Project README](../README.md) - Project overview and quick start
- [Source Code](../src/) - Implementation details
- [Tests](../tests/) - Test implementations and examples
- [Docker Configuration](../docker-compose.yml) - Service definitions
