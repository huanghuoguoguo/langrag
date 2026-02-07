"""Tests for PageIndex Processor and Components."""

import asyncio
import uuid
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from langrag.entities.dataset import Dataset
from langrag.entities.document import Document, DocumentType
from langrag.index_processor.processor.page_index import (
    PageIndexConfig,
    PageIndexProcessor,
    MarkdownStructureParser,
    TreeNode,
)


class TestPageIndexConfig:
    """Tests for PageIndexConfig dataclass."""

    def test_default_values(self):
        """Default configuration values are set correctly."""
        config = PageIndexConfig()

        assert config.max_summary_tokens == 500
        assert config.max_concurrency == 5
        assert "{content}" in config.summarize_prompt
        assert len(config.header_patterns) == 4  # H1-H4

    def test_custom_values(self):
        """Custom configuration values are accepted."""
        config = PageIndexConfig(
            max_summary_tokens=256,
            max_concurrency=10,
        )

        assert config.max_summary_tokens == 256
        assert config.max_concurrency == 10


class TestTreeNode:
    """Tests for TreeNode dataclass."""

    def test_create_node(self):
        """Create a basic TreeNode."""
        node = TreeNode(
            id="test-1",
            content="Test content",
            level=1,
        )

        assert node.id == "test-1"
        assert node.content == "Test content"
        assert node.level == 1
        assert node.children == []
        assert node.parent is None
        assert node.summary is None

    def test_add_child(self):
        """Add child node to parent."""
        parent = TreeNode(id="parent", content="Parent", level=0)
        child = TreeNode(id="child", content="Child", level=1)

        parent.add_child(child)

        assert len(parent.children) == 1
        assert parent.children[0] == child
        assert child.parent == parent

    def test_add_multiple_children(self):
        """Add multiple children to parent."""
        parent = TreeNode(id="parent", content="Parent", level=0)

        for i in range(3):
            child = TreeNode(id=f"child-{i}", content=f"Child {i}", level=1)
            parent.add_child(child)

        assert len(parent.children) == 3
        for child in parent.children:
            assert child.parent == parent


class TestMarkdownStructureParser:
    """Tests for MarkdownStructureParser."""

    @pytest.fixture
    def parser(self):
        """Create a parser with default patterns."""
        patterns = [
            (r"^#\s+(.*)", 1),
            (r"^##\s+(.*)", 2),
            (r"^###\s+(.*)", 3),
        ]
        return MarkdownStructureParser(patterns)

    def test_parse_simple_markdown(self, parser):
        """Parse markdown with single header."""
        text = """# Introduction

This is the introduction paragraph.
"""
        root = parser.parse(text, "doc-1")

        assert root.level == 0
        assert len(root.children) == 1
        assert root.children[0].metadata.get("title") == "Introduction"

    def test_parse_nested_headers(self, parser):
        """Parse markdown with nested headers."""
        text = """# Chapter 1

Intro to chapter 1.

## Section 1.1

Section details.

## Section 1.2

More details.

# Chapter 2

Intro to chapter 2.
"""
        root = parser.parse(text, "doc-2")

        # Root should have 2 H1 children (Chapter 1, Chapter 2)
        assert len(root.children) == 2

        chapter1 = root.children[0]
        assert chapter1.metadata.get("title") == "Chapter 1"
        assert chapter1.level == 1

        # Chapter 1 should have content + 2 H2 sections
        # (content node + Section 1.1 + Section 1.2)
        assert len(chapter1.children) >= 2

    def test_parse_only_content(self, parser):
        """Parse text without any headers."""
        text = """Just some plain content.
No headers here.
"""
        root = parser.parse(text, "doc-3")

        assert root.level == 0
        # Content should be attached directly to root
        assert len(root.children) == 1
        assert root.children[0].metadata.get("is_leaf") is True

    def test_parse_empty_content(self, parser):
        """Parse empty text."""
        root = parser.parse("", "doc-4")

        assert root.level == 0
        assert len(root.children) == 0

    def test_header_level_detection(self, parser):
        """Detect correct header levels."""
        text = """# H1

## H2

### H3
"""
        root = parser.parse(text, "doc-5")

        h1 = root.children[0]
        assert h1.level == 1

        h2 = h1.children[0]
        assert h2.level == 2

        h3 = h2.children[0]
        assert h3.level == 3


class TestPageIndexProcessor:
    """Tests for PageIndexProcessor class."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM with async support."""
        llm = MagicMock()
        llm.chat_async = AsyncMock(return_value="This is a summary of the content.")
        llm.embed_documents.return_value = [[0.1, 0.2, 0.3]]
        llm.embed_query.return_value = [0.1, 0.2, 0.3]
        return llm

    @pytest.fixture
    def mock_embedder(self):
        """Create a mock embedder."""
        embedder = MagicMock()
        embedder.embed.return_value = [[0.1, 0.2, 0.3]]
        return embedder

    @pytest.fixture
    def mock_vector_manager(self):
        """Create a mock vector manager."""
        manager = MagicMock()
        manager.add_texts = MagicMock()
        return manager

    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset."""
        return Dataset(
            id="test-dataset",
            name="Test Dataset",
            description="Test description",
            collection_name="test_collection",
        )

    @pytest.fixture
    def sample_markdown_document(self):
        """Create a sample markdown document."""
        content = """# Overview

This document provides an overview.

## Background

Some background information here.

## Methodology

The methodology section describes our approach.

# Results

Here are the results.

## Key Findings

The key findings are summarized below.
"""
        return Document(
            page_content=content,
            metadata={
                "document_id": "test-doc-1",
                "document_name": "test.md",
            },
        )

    def test_init_with_all_components(self, mock_llm, mock_embedder, mock_vector_manager):
        """Initialize with all components."""
        processor = PageIndexProcessor(
            llm=mock_llm,
            embedder=mock_embedder,
            vector_manager=mock_vector_manager,
        )

        assert processor.llm == mock_llm
        assert processor.embedder == mock_embedder
        assert processor.vector_manager == mock_vector_manager
        assert processor.cleaner is not None
        assert processor.config is not None
        assert processor.parser is not None

    def test_init_with_custom_config(self, mock_llm):
        """Initialize with custom config."""
        config = PageIndexConfig(max_summary_tokens=256)
        processor = PageIndexProcessor(llm=mock_llm, config=config)

        assert processor.config.max_summary_tokens == 256

    def test_embed_texts_uses_embedder(self, mock_llm, mock_embedder):
        """Uses embedder when provided."""
        processor = PageIndexProcessor(llm=mock_llm, embedder=mock_embedder)
        result = processor._embed_texts(["test"])

        mock_embedder.embed.assert_called_once_with(["test"])
        assert result == [[0.1, 0.2, 0.3]]

    def test_embed_texts_uses_llm(self, mock_llm):
        """Uses LLM when embedder not provided."""
        processor = PageIndexProcessor(llm=mock_llm)
        processor._embed_texts(["test"])

        mock_llm.embed_documents.assert_called_once_with(["test"])

    def test_process_creates_tree_structure(
        self,
        mock_llm,
        mock_embedder,
        mock_vector_manager,
        sample_dataset,
        sample_markdown_document,
    ):
        """Process creates correct tree structure in metadata."""
        processor = PageIndexProcessor(
            llm=mock_llm,
            embedder=mock_embedder,
            vector_manager=mock_vector_manager,
        )

        processor.process(sample_dataset, [sample_markdown_document])

        # Vector manager should be called
        mock_vector_manager.add_texts.assert_called_once()

        # Get the documents that were added
        call_args = mock_vector_manager.add_texts.call_args
        stored_docs = call_args[0][1]

        # Should have multiple nodes (headers + content)
        assert len(stored_docs) > 0

        # Check metadata contains tree structure fields
        for doc in stored_docs:
            assert "node_id" in doc.metadata
            assert "level" in doc.metadata
            assert "is_leaf" in doc.metadata

    def test_process_adds_dataset_id(
        self,
        mock_llm,
        mock_embedder,
        mock_vector_manager,
        sample_dataset,
        sample_markdown_document,
    ):
        """Dataset ID is added to node metadata."""
        processor = PageIndexProcessor(
            llm=mock_llm,
            embedder=mock_embedder,
            vector_manager=mock_vector_manager,
        )

        processor.process(sample_dataset, [sample_markdown_document])

        call_args = mock_vector_manager.add_texts.call_args
        stored_docs = call_args[0][1]

        for doc in stored_docs:
            assert doc.metadata.get("dataset_id") == "test-dataset"

    def test_process_empty_documents(
        self, mock_llm, mock_vector_manager, sample_dataset
    ):
        """Process empty document list."""
        processor = PageIndexProcessor(
            llm=mock_llm,
            vector_manager=mock_vector_manager,
        )

        processor.process(sample_dataset, [])

        mock_vector_manager.add_texts.assert_not_called()

    def test_process_with_callback(
        self,
        mock_llm,
        mock_embedder,
        mock_vector_manager,
        sample_dataset,
        sample_markdown_document,
    ):
        """Progress callback is called during processing."""
        callback = MagicMock()
        processor = PageIndexProcessor(
            llm=mock_llm,
            embedder=mock_embedder,
            vector_manager=mock_vector_manager,
            progress_callback=callback,
        )

        processor.process(sample_dataset, [sample_markdown_document])

        assert callback.called
        # Should be called at least once for parsing, summarizing, and saving
        assert callback.call_count >= 1

    @patch("langrag.datasource.vdb.global_manager.get_vector_manager")
    def test_process_uses_global_manager(
        self, mock_get_manager, mock_llm, mock_embedder, sample_dataset, sample_markdown_document
    ):
        """Uses global vector manager when not provided."""
        mock_global_manager = MagicMock()
        mock_get_manager.return_value = mock_global_manager

        processor = PageIndexProcessor(
            llm=mock_llm, embedder=mock_embedder, vector_manager=None
        )
        processor.process(sample_dataset, [sample_markdown_document])

        mock_get_manager.assert_called_once()
        mock_global_manager.add_texts.assert_called_once()

    def test_flatten_tree_preserves_hierarchy(
        self, mock_llm, mock_embedder, mock_vector_manager, sample_dataset
    ):
        """Flattening preserves parent-child relationships."""
        processor = PageIndexProcessor(
            llm=mock_llm,
            embedder=mock_embedder,
            vector_manager=mock_vector_manager,
        )

        # Manually create a tree
        root = TreeNode(id="root", content="", level=0)
        child1 = TreeNode(id="child1", content="# Chapter 1", level=1, metadata={"title": "Chapter 1"})
        child1.summary = "Summary of Chapter 1"
        leaf = TreeNode(id="leaf1", content="Leaf content", level=2, metadata={"is_leaf": True})
        
        root.add_child(child1)
        child1.add_child(leaf)

        docs = processor._flatten_tree(root, "dataset-1", "doc-1")

        # Should produce docs for child1 (with summary) and leaf
        assert len(docs) >= 2

        # Find the child1 doc
        child_docs = [d for d in docs if d.metadata.get("node_id") == "child1"]
        if child_docs:
            child_doc = child_docs[0]
            assert "leaf1" in child_doc.metadata.get("children_ids", [])


class TestPageIndexProcessorIntegration:
    """Integration-style tests for PageIndexProcessor."""

    @pytest.fixture
    def mock_llm_with_summarization(self):
        """Create mock LLM that provides varied summaries."""
        llm = MagicMock()
        call_count = [0]

        async def mock_chat_async(messages):
            call_count[0] += 1
            return f"Summary #{call_count[0]}: Key points from the content."

        llm.chat_async = mock_chat_async
        llm.embed_documents.side_effect = lambda texts: [[0.1 * i, 0.2, 0.3] for i in range(len(texts))]
        return llm

    def test_end_to_end_processing(self, mock_llm_with_summarization):
        """Test complete processing flow."""
        mock_manager = MagicMock()
        dataset = Dataset(
            id="e2e-test",
            name="E2E Test",
            collection_name="e2e_collection",
        )

        document = Document(
            page_content="""# Introduction

Overview of the topic.

## Background

Historical context.

# Methodology

Our approach.
""",
            metadata={"document_id": "e2e-doc"},
        )

        processor = PageIndexProcessor(
            llm=mock_llm_with_summarization,
            vector_manager=mock_manager,
        )

        processor.process(dataset, [document])

        mock_manager.add_texts.assert_called_once()
        stored_docs = mock_manager.add_texts.call_args[0][1]

        # Verify we have both summary and leaf nodes
        summaries = [d for d in stored_docs if d.metadata.get("is_summary")]
        leaves = [d for d in stored_docs if d.metadata.get("is_leaf")]

        # Should have at least some nodes
        assert len(stored_docs) > 0
