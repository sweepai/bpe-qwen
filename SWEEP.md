# SWEEP.md - Development Guidelines and Commands

## Core Development Philosophy

### Fail Fast, Not Fail Safe
- **Let everything fail fast rather than fail safe**
- Don't wrap code in try/except blocks unless absolutely necessary for error handling
- Tests should fail immediately when something is wrong, not skip or hide errors
- Avoid defensive programming that masks real issues
- If a dependency is missing or a setup is wrong, let it crash with a clear error message

## Testing Guidelines

### Test Commands
```bash
# Run specific test file
python -m pytest tests/regression/test_auto_linear_tokenizer.py -v

# Run specific test class
python -m pytest tests/regression/test_auto_linear_tokenizer.py::TestAutoLinearTokenizerBasics -v

# Run specific test method
python -m pytest tests/regression/test_auto_linear_tokenizer.py::TestAutoLinearTokenizerBasics::test_can_create_tokenizer_instance -v

# Run parallelization tests
python -m pytest tests/regression/test_auto_linear_tokenizer.py::TestParallelization -v

# Run all tests
python -m pytest tests/ -v
```

### Test Writing Principles
- No unnecessary try/except blocks in tests
- No sys.path.append - use proper imports
- Let tests fail with clear error messages
- Don't skip tests unless the environment genuinely doesn't support them
- Test real functionality, not mocked behavior when possible

## Code Style Preferences

### Import Style
- Use direct imports from the package structure
- Avoid sys.path manipulation
- Prefer absolute imports over relative imports

### Error Handling
- Only catch exceptions when you can meaningfully handle them
- Let unexpected errors bubble up with full stack traces
- Use specific exception types, not bare `except:` clauses

## Project Structure

### Key Directories
- `python/bpe_qwen/` - Main Python package
- `tests/` - Test files
- `tests/regression/` - Regression tests for specific components
- `data/` - Tokenizer data files

### Important Files
- `python/bpe_qwen/auto_linear_tokenizer.py` - AutoLinearTokenizer implementation
- `tests/regression/test_auto_linear_tokenizer.py` - End-to-end tests for AutoLinearTokenizer

## Development Dependencies

### Required for Development
```toml
[project.optional-dependencies]
dev = [
    "transformers",
    "datasets",
    "huggingface-hub",
]
```

Install with: `uv pip install -e ".[dev]"`

## Common Issues and Solutions

### Import Errors
- Make sure you're running from the project root
- Ensure the package is installed in development mode: `uv pip install -e .`
- Don't use sys.path.append - fix the import structure instead

### Test Failures
- Let them fail! Don't wrap in try/except
- Check that required data files exist in `data/` directory
- Ensure tokenizer dependencies are properly installed