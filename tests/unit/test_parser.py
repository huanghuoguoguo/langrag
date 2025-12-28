import pytest
from pathlib import Path
from langrag.entities.document import Document
from langrag.index_processor.extractor import SimpleTextParser

def test_simple_text_parser(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("Hello World", encoding="utf-8")
    
    parser = SimpleTextParser()
    docs = parser.parse(f)
    
    assert len(docs) == 1
    assert docs[0].page_content == "Hello World"
    assert docs[0].metadata["filename"] == "test.txt"
    assert str(docs[0].metadata["source"]) == str(f.absolute())

def test_parser_file_not_found():
    parser = SimpleTextParser()
    with pytest.raises(FileNotFoundError):
        parser.parse("non_existent_file.txt")

def test_parse_utf8_encoding(tmp_path):
    f = tmp_path / "utf8.txt"
    f.write_text("你好，世界", encoding="utf-8")
    
    parser = SimpleTextParser(encoding="utf-8")
    docs = parser.parse(f)
    
    assert docs[0].page_content == "你好，世界"
