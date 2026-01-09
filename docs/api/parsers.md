# Parsers

Document parsers for extracting text from various file formats.

## BaseParser

Abstract base class for all parsers.

```python
from langrag import BaseParser

class CustomParser(BaseParser):
    def parse(self, file_path: str) -> str:
        # Your parsing implementation
        pass

    def supported_extensions(self) -> list[str]:
        return [".custom"]
```

### Methods

| Method | Description |
|--------|-------------|
| `parse(file_path)` | Parse file and return plain text |
| `supported_extensions()` | Return list of supported file extensions |

## SimpleTextParser

Parser for plain text files.

```python
from langrag import SimpleTextParser

parser = SimpleTextParser(encoding="utf-8")
content = parser.parse("document.txt")
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `encoding` | str | "utf-8" | File encoding |

### Supported Extensions

- `.txt`
- `.text`

## Other Parsers

LangRAG supports additional parsers through optional dependencies:

| Parser | Formats | Install |
|--------|---------|---------|
| PDFParser | `.pdf` | `pip install langrag[parsers]` |
| DocxParser | `.docx` | `pip install langrag[parsers]` |
| MarkdownParser | `.md` | `pip install langrag[parsers]` |
| HTMLParser | `.html` | `pip install langrag[parsers]` |
