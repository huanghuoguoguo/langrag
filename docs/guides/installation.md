# Installation

## Requirements

- Python 3.11 or higher
- pip or uv package manager

## Basic Installation

Install LangRAG using pip:

```bash
pip install langrag
```

Or using uv:

```bash
uv add langrag
```

## Optional Dependencies

LangRAG has several optional dependency groups for specific features:

### Document Parsers

For extended document format support:

```bash
pip install langrag[parsers]
```

This includes:

- `pypdf` - PDF document parsing
- `python-docx` - Word document parsing
- `markdown` - Markdown parsing
- `beautifulsoup4` - HTML parsing
- `chardet` - Encoding detection

### Reranker Support

For reranking capabilities:

```bash
pip install langrag[reranker]
```

### Observability

For OpenTelemetry tracing:

```bash
pip install langrag[observability]
```

### All Features

Install everything:

```bash
pip install langrag[all]
```

## Development Installation

For contributing to LangRAG:

```bash
# Clone the repository
git clone https://github.com/huanghuoguoguo/langrag.git
cd langrag

# Install with development dependencies
uv sync

# Or with pip
pip install -e ".[dev]"
```

## Verify Installation

```python
import langrag
print(langrag.__version__)
```

## Next Steps

- [Quick Start Guide](quickstart.md)
- [Core Concepts](concepts.md)
