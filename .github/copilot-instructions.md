# Copilot Instructions for TherasyncPipeline Project

## General Guidelines

### Language and Documentation
- **ALL code documentation MUST be in English** (docstrings, comments, variable names, function names)
- **ALL commit messages MUST be in English**
- Use clear, descriptive naming conventions following PEP 8
- Every function/class must have a comprehensive docstring

### Code Quality Standards
- **Maximum file length: 200 lines** (excluding docstrings and comments)
- If a file exceeds 200 lines, split it into multiple modules
- Follow SOLID principles and keep functions focused on single responsibilities
- Use type hints for all function signatures

### Testing Requirements
- **ALL functions MUST have corresponding unit tests**
- Tests must be located in the `tests/` directory
- Use pytest framework
- Aim for >80% code coverage
- Test edge cases and error conditions

### File Operations - CRITICAL RULES
⚠️ **NEVER create, modify, or delete ANY file without explicit user approval**
- Always ask before creating new files
- Always ask before modifying existing files
- Always ask before deleting files
- Show the user what changes you intend to make first

### Logging Standards
- Use Python's `logging` module (not print statements)
- Log files must be stored in `log/` directory
- Use appropriate log levels:
  - DEBUG: Detailed diagnostic information
  - INFO: General information about execution
  - WARNING: Warnings about potential issues
  - ERROR: Error messages for failures
  - CRITICAL: Critical failures requiring immediate attention
- Include timestamps and module names in logs
- Rotate logs to prevent excessive file sizes

## Project Structure

```
TherasyncPipeline/
├── src/           # Source code modules
├── scripts/       # Executable scripts
├── tests/         # Unit and integration tests
├── docs/          # Documentation
├── config/        # Configuration files
├── notebooks/     # Jupyter notebooks for analysis and visualization
├── data/          # Input data and derivatives - gitignored
├── log/           # Log files - gitignored
├── pyproject.toml # Poetry dependencies
└── README.md      # Project documentation
```

## Workflow
1. Always start by understanding the requirements
2. Plan the architecture before coding
3. Write tests before or alongside implementation (TDD encouraged)
4. Document as you code
5. Ask for approval before making file changes
6. Commit small, logical changes with descriptive messages

## Python Style Guide
- Follow PEP 8
- Use Black formatter (line length: 88)
- Use isort for import sorting
- Use pylint/flake8 for linting
- Maximum function complexity: 10 (cyclomatic complexity)

## Error Handling
- Use specific exception types
- Always log errors with full context
- Provide helpful error messages to users
- Fail gracefully with meaningful feedback

## Performance Guidelines
- Profile code when processing large datasets
- Use generators for memory efficiency
- Parallelize when beneficial (multiprocessing/threading)
- Cache expensive computations when appropriate

## Security
- Never commit sensitive data
- Sanitize user inputs
- Use secure random for any cryptographic needs
- Keep dependencies updated

## Version Control
- Commit frequently with meaningful messages
- Use conventional commit format: `type(scope): description`
  - Types: feat, fix, docs, style, refactor, test, chore
- Keep commits atomic and focused
- Branch for features/experiments

### Sprint Workflow
- **Each sprint gets a dedicated branch**: `sprint-N/description`
  - Example: `sprint-1/config-and-utilities`
- Create branch from `master` before starting sprint work
- Make all sprint commits on the sprint branch
- Request user approval before merging to master
- Delete sprint branch after successful merge
- Never push directly to master without approval

Remember: Quality over speed. Ask questions when requirements are unclear.