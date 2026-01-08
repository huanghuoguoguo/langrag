# Contributing to LangRAG

Thank you for your interest in contributing to LangRAG! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Development Workflow](#development-workflow)
- [Code Style](#code-style)
- [Testing](#testing)
- [Commit Guidelines](#commit-guidelines)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)

## Development Setup

### Prerequisites

- Python 3.11 or higher
- [uv](https://github.com/astral-sh/uv) - Fast Python package manager

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/huanghuoguoguo/langrag.git
   cd langrag
   ```

2. Install dependencies with uv:
   ```bash
   uv sync --all-extras
   ```

3. Copy environment configuration:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. Verify installation:
   ```bash
   uv run pytest tests/smoke/ -v
   ```

## Project Structure

```
langrag/
├── src/langrag/          # Main source code
│   ├── config/           # Configuration management
│   ├── core/             # Core abstractions
│   ├── datasource/       # Data sources and vector DBs
│   ├── entities/         # Domain entities
│   ├── index_processor/  # Indexing strategies
│   ├── llm/              # LLM integrations
│   ├── retrieval/        # Retrieval workflow
│   └── utils/            # Utilities
├── web/                  # Web UI and API
├── tests/                # Test suites
│   ├── smoke/            # Quick sanity checks
│   ├── unit/             # Unit tests
│   ├── integration/      # Integration tests
│   └── e2e/              # End-to-end tests
└── main.py               # Application entry point
```

## Development Workflow

1. **Create a branch** from `main`:
   ```bash
   git checkout main
   git pull origin main
   git checkout -b <type>/<description>
   ```

2. **Make your changes** following the code style guidelines.

3. **Run tests** to ensure nothing is broken:
   ```bash
   ./run_tests.sh unit
   ```

4. **Run linting and type checks**:
   ```bash
   uv run ruff check .
   uv run ruff format .
   uv run mypy src/
   ```

5. **Commit your changes** following the commit guidelines.

6. **Push and create a Pull Request**.

### Branch Naming Convention

| Prefix      | Purpose                |
|-------------|------------------------|
| `feat/`     | New features           |
| `fix/`      | Bug fixes              |
| `docs/`     | Documentation          |
| `refactor/` | Code refactoring       |
| `test/`     | Test additions/changes |
| `chore/`    | Maintenance tasks      |

Examples:
- `feat/add-semantic-cache`
- `fix/duckdb-connection-leak`
- `docs/update-api-reference`

## Code Style

This project uses **Ruff** for linting and formatting, and **MyPy** for type checking.

### Ruff Configuration

- Line length: 100 characters
- Target Python version: 3.11
- Enabled rules: pycodestyle, pyflakes, isort, flake8-bugbear, and more

### Formatting

```bash
# Check for issues
uv run ruff check .

# Auto-fix issues
uv run ruff check --fix .

# Format code
uv run ruff format .
```

### Type Hints

- Use type hints for function signatures
- Run MyPy to check types:
  ```bash
  uv run mypy src/
  ```

### General Guidelines

- Write clear, self-documenting code
- Keep functions focused and small
- Use meaningful variable and function names
- Add docstrings for public APIs
- Prefer composition over inheritance

## Testing

### Running Tests

```bash
# All tests with coverage
./run_tests.sh all

# Specific test suites
./run_tests.sh smoke        # Quick sanity checks
./run_tests.sh unit         # Unit tests with coverage
./run_tests.sh integration  # Integration tests
./run_tests.sh e2e          # End-to-end tests

# Run specific test file
uv run pytest tests/unit/test_example.py -v
```

### Writing Tests

- Place tests in the appropriate directory (`smoke/`, `unit/`, `integration/`, `e2e/`)
- Use descriptive test names: `test_should_return_error_when_input_is_invalid`
- Use pytest fixtures for common setup
- Aim for high coverage on core modules

### Test Markers

```python
import pytest

@pytest.mark.smoke
def test_basic_functionality():
    """Quick sanity check."""
    pass

@pytest.mark.asyncio
async def test_async_operation():
    """Test async code."""
    pass
```

## Commit Guidelines

We follow [Conventional Commits](https://www.conventionalcommits.org/) specification.

### Format

```
<type>: <description>

[optional body]

[optional footer]
```

### Types

| Type       | Description                          |
|------------|--------------------------------------|
| `feat`     | New feature                          |
| `fix`      | Bug fix                              |
| `docs`     | Documentation changes                |
| `style`    | Code style (formatting, no logic)    |
| `refactor` | Code refactoring                     |
| `test`     | Adding or updating tests             |
| `chore`    | Maintenance tasks                    |
| `perf`     | Performance improvements             |

### Examples

```bash
feat: add OpenTelemetry tracing support
fix: resolve DuckDB connection leak on shutdown
docs: update README with FTS clarification
refactor: split rag_kernel into smaller modules
test: add unit tests for retrieval workflow
```

## Pull Request Process

### Before Submitting

1. Ensure all tests pass locally
2. Run linting and fix any issues
3. Update documentation if needed
4. Rebase on latest `main` if necessary

### PR Template

```markdown
## Summary
- Brief description of changes

## Changes
- List of specific modifications

## Test Plan
- How the changes were tested
- Any manual testing steps

## Related Issues
- Fixes #123
```

### Review Process

1. Submit PR against `main` branch
2. Ensure CI checks pass
3. Address reviewer feedback
4. Squash commits if requested
5. Maintainer will merge after approval

## Issue Reporting

### Bug Reports

Please include:
- Clear description of the issue
- Steps to reproduce
- Expected vs actual behavior
- Environment details (Python version, OS, etc.)
- Relevant logs or error messages

### Feature Requests

Please include:
- Clear description of the feature
- Use case / motivation
- Proposed implementation (if any)

---

## Questions?

If you have questions, feel free to:
- Open a [GitHub Issue](https://github.com/huanghuoguoguo/langrag/issues)
- Check existing issues and discussions

Thank you for contributing!
