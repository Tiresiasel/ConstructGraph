# Centralized Prompt Management System

## Overview

This directory contains centralized prompt configurations for all scripts in the project. Instead of hardcoding prompts in individual script files, all scripts now read from these centralized files, ensuring consistency and easier maintenance.

## File Structure

```
scripts/prompts/
├── README.md           # This file
├── extraction.txt      # Main extraction prompt for LLM-based knowledge extraction
└── [future prompts]    # Additional prompt files can be added here
```

## How It Works

### 1. Centralized Prompt Storage
- All prompts are stored in text files in this directory
- Each prompt file contains the complete prompt content
- Scripts dynamically load prompts at runtime

### 2. Automatic Loading
Scripts automatically load prompts using this pattern:
```python
# Read the centralized prompt from extraction.txt
prompt_file_path = Path(__file__).parent / "prompts" / "extraction.txt"
try:
    with open(prompt_file_path, 'r', encoding='utf-8') as f:
        prompt_content = f.read()
    print(f"Successfully loaded prompt from: {prompt_file_path}")
except FileNotFoundError:
    print(f"Warning: Prompt file not found at {prompt_file_path}")
    # Fallback to default prompt
```

### 3. Fallback Mechanism
- If a prompt file is not found, scripts fall back to a minimal default prompt
- This ensures scripts continue to work even if prompt files are missing

## Current Prompts

### extraction.txt
- **Purpose**: Main prompt for extracting structured knowledge from academic papers
- **Used by**: `scripts/build_graph.py`
- **Content**: Comprehensive extraction instructions including:
  - Role and goal definition
  - Core directives
  - Multi-step extraction process
  - Scope policies
  - JSON schema requirements
  - Controlled vocabulary rules

## Benefits

### ✅ **Consistency**
- All scripts use the exact same prompt content
- No more discrepancies between different script versions
- Centralized quality control

### ✅ **Maintainability**
- Update prompts in one place
- Changes automatically propagate to all scripts
- Version control for prompt evolution

### ✅ **Collaboration**
- Team members can easily review and modify prompts
- Clear separation between prompt logic and script logic
- Easy to track prompt changes in git

### ✅ **Flexibility**
- Easy to create prompt variants for different use cases
- Simple to A/B test different prompt versions
- Quick prompt iteration without touching code

## Usage

### For Developers
1. **Modify prompts**: Edit the `.txt` files in this directory
2. **Test changes**: Run your scripts to see prompt updates in action
3. **Version control**: Commit prompt changes separately from code changes

### For Users
1. **Customize prompts**: Modify the `.txt` files to suit your specific needs
2. **Share prompts**: Copy prompt files to other projects
3. **Backup prompts**: Keep copies of working prompt versions

## Adding New Prompts

1. Create a new `.txt` file in this directory
2. Add your prompt content
3. Update the relevant script to load from the new file
4. Test the integration
5. Document the new prompt in this README

## Best Practices

### Prompt Design
- Keep prompts focused and specific
- Use clear, unambiguous language
- Include examples where helpful
- Test prompts with various inputs

### File Management
- Use descriptive filenames
- Include version information in comments
- Document any special formatting requirements
- Keep prompts under version control

### Integration
- Always include fallback mechanisms
- Log when prompts are loaded successfully
- Handle encoding issues gracefully
- Test with missing prompt files

## Troubleshooting

### Common Issues

1. **Prompt not loading**
   - Check file path in script
   - Verify file exists and is readable
   - Check file encoding (should be UTF-8)

2. **Encoding errors**
   - Ensure files are saved as UTF-8
   - Check for special characters
   - Use proper encoding in file operations

3. **Fallback not working**
   - Verify fallback logic in script
   - Check error handling
   - Test with intentionally missing files

### Debug Steps

1. Check script logs for loading messages
2. Verify file paths are correct
3. Test file reading manually
4. Check file permissions
5. Verify encoding settings

## Future Enhancements

- [ ] Prompt versioning system
- [ ] Prompt validation tools
- [ ] Prompt performance metrics
- [ ] A/B testing framework
- [ ] Prompt template system
- [ ] Multi-language support

## Contributing

When contributing to prompts:
1. Follow existing formatting conventions
2. Test changes thoroughly
3. Update documentation
4. Consider backward compatibility
5. Get feedback from team members

---

**Note**: This system replaces the previous hardcoded prompt approach. All scripts have been updated to use this centralized system.
