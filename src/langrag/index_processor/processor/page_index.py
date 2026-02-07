"""
PageIndex Processor - Hierarchical Tree-Structured Indexing.

This processor implements the "Page Index" (or Tree Index) strategy, which:
1. Parses documents into a hierarchical tree structure based on headers (H1, H2, etc.).
2. Generates summaries for each node in the tree using an LLM.
3. Indexes both original content leaf nodes and summary nodes into the vector store.
4. Preserves structural relationships (parent/child) in metadata to enable agentic navigation.
"""

import asyncio
import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from langrag.entities.dataset import Dataset
from langrag.entities.document import Document, DocumentType
from langrag.index_processor.cleaner.cleaner import Cleaner
from langrag.index_processor.processor.base import BaseIndexProcessor
from langrag.llm.base import BaseLLM

logger = logging.getLogger(__name__)


# ==================== Prompt Templates ====================

DEFAULT_SUMMARIZE_PROMPT = """You are a document structure analyzer. Summarize the content below.

Requirements:
1. Keep the summary concise (2-4 sentences).
2. Preserve key entities, numbers, and terminology.
3. If the content is in Chinese, respond in Chinese; otherwise respond in English.
4. Focus on WHAT information is covered, not how.

Content:
{content}

Summary:"""

DEFAULT_SUMMARIZE_PROMPT_ZH = """你是一个文档结构分析助手。请对以下内容进行摘要总结。

要求：
1. 摘要简洁，2-4句话即可。
2. 保留关键实体、数字和专业术语。
3. 聚焦于"涵盖了什么信息"，而非"如何阐述"。

内容：
{content}

摘要："""


@dataclass
class PageIndexConfig:
    """Configuration for PageIndex processing."""

    summarize_prompt: str = field(default_factory=lambda: DEFAULT_SUMMARIZE_PROMPT)
    """Prompt template for summarizing nodes. Use {content} placeholder."""
    
    summarize_prompt_zh: str = field(default_factory=lambda: DEFAULT_SUMMARIZE_PROMPT_ZH)
    """Chinese prompt template for summarizing nodes."""
    
    auto_detect_language: bool = True
    """Automatically detect content language and use appropriate prompt."""

    max_summary_tokens: int = 500
    """Maximum tokens for summary generation."""

    max_concurrency: int = 5
    """Max concurrent LLM calls for summarization."""
    
    min_content_length_for_summary: int = 500
    """Only generate LLM summary if content exceeds this length."""

    header_patterns: List[Tuple[str, int]] = field(default_factory=lambda: [
        (r"^#\s+(.*)", 1),
        (r"^##\s+(.*)", 2),
        (r"^###\s+(.*)", 3),
        (r"^####\s+(.*)", 4),
        # Chinese Markdown variants (full-width)
        (r"^#\u3000+(.*)", 1),
        (r"^##\u3000+(.*)", 2),
    ])
    """Regex patterns to identify headers and their levels."""


@dataclass
class TreeNode:
    """Internal representation of a document node."""
    id: str
    content: str  # Original text content
    level: int    # 0=Root, 1=H1, 2=H2, ...
    children: List["TreeNode"] = field(default_factory=list)
    parent: Optional["TreeNode"] = None
    summary: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_child(self, node: "TreeNode"):
        node.parent = self
        self.children.append(node)


class MarkdownStructureParser:
    """Parses markdown text into a hierarchical tree of TreeNodes."""

    def __init__(self, header_patterns: List[Tuple[str, int]]):
        self.header_patterns = [
            (re.compile(pattern), level) for pattern, level in header_patterns
        ]

    def parse(self, text: str, doc_id: str) -> TreeNode:
        """Parse text into a tree structure."""
        lines = text.split("\n")
        root = TreeNode(id=f"{doc_id}_root", content="", level=0, metadata={"title": "Root"})
        
        # Stack to keep track of current path in tree: [root, h1, h2, ...]
        # invariant: stack[-1] is the parent of the current node being processed
        stack: List[TreeNode] = [root]
        
        current_content_lines: List[str] = []
        
        for line in lines:
            header_level = self._get_header_level(line)
            
            if header_level is not None:
                # Flush previous content to the current node (top of stack)
                if current_content_lines:
                    self._append_content_node(stack[-1], current_content_lines, doc_id)
                    current_content_lines = []
                
                # Pop stack until we find the parent (level < header_level)
                while len(stack) > 1 and stack[-1].level >= header_level:
                    stack.pop()
                
                # Create new header node
                new_node = TreeNode(
                    id=str(uuid.uuid4()),
                    content=line, # Header line itself
                    level=header_level,
                    metadata={"title": line.strip().lstrip("#").strip()}
                )
                stack[-1].add_child(new_node)
                stack.append(new_node)
            else:
                current_content_lines.append(line)
        
        # Flush remaining content
        if current_content_lines:
            self._append_content_node(stack[-1], current_content_lines, doc_id)
            
        return root

    def _get_header_level(self, line: str) -> Optional[int]:
        """Check if line matches any header pattern."""
        for pattern, level in self.header_patterns:
            if pattern.match(line):
                return level
        return None

    def _append_content_node(self, parent: TreeNode, lines: List[str], doc_id: str):
        """Append a leaf content node to the parent."""
        content = "\n".join(lines).strip()
        if not content:
            return
            
        leaf = TreeNode(
            id=str(uuid.uuid4()),
            content=content,
            level=parent.level + 1, # Leaf is conceptually one level deeper
            metadata={"is_leaf": True}
        )
        parent.add_child(leaf)


class PageIndexProcessor(BaseIndexProcessor):
    """
    PageIndex Processor.
    
    Builds a tree-structured index:
    - Parses document into a hierarchy (H1 -> H2 -> Content).
    - Summarizes each structural node (H1, H2) based on its children.
    - Stores both Summaries and Leaf Content in VectorStore.
    """

    def __init__(
        self,
        llm: BaseLLM,
        embedder: Any = None,
        vector_manager: Any = None,
        cleaner: Cleaner | None = None,
        config: PageIndexConfig | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ):
        self.llm = llm
        self.embedder = embedder
        self.vector_manager = vector_manager
        self.cleaner = cleaner or Cleaner()
        self.config = config or PageIndexConfig()
        self.progress_callback = progress_callback
        self.parser = MarkdownStructureParser(self.config.header_patterns)

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        if self.embedder is not None:
            return self.embedder.embed(texts)
        return self.llm.embed_documents(texts)

    async def _summarize_node(self, node: TreeNode) -> str:
        """
        Recursively summarize a node.
        Summary is based on:
        1. Node's own content (header title)
        2. Summaries of its children (if internal node)
        3. Content of its children (if leaf node)
        """
        if node.summary:
            return node.summary

        # Base case: Leaf node (just content)
        if not node.children:
            # For leaf nodes, the summary is just a truncated version of content?
            # Or we can summarize the content if it's too long.
            # For now, let's say leaf summary = content itself (or generated)
            if len(node.content) > 1000:
                 prompt = self.config.summarize_prompt.format(content=node.content[:3000])
                 try:
                    summary = await self.llm.chat_async([{"role": "user", "content": prompt}])
                    node.summary = summary
                 except Exception as e:
                     logger.warning(f"Failed to summarize leaf node: {e}")
                     node.summary = node.content[:500]
            else:
                node.summary = node.content
            return node.summary

        # Recursive case: Internal node
        # First gather children summaries
        child_summaries = []
        for child in node.children:
            s = await self._summarize_node(child)
            child_summaries.append(s)
        
        combined_content = f"Section: {node.metadata.get('title', '')}\n\n" + "\n---\n".join(child_summaries)
        
        try:
            prompt = self.config.summarize_prompt.format(content=combined_content[:6000]) # Limit context
            summary = await self.llm.chat_async([{"role": "user", "content": prompt}])
            node.summary = summary
        except Exception as e:
            logger.warning(f"Failed to summarize internal node: {e}")
            node.summary = combined_content[:500] + "..."
            
        return node.summary

    def _flatten_tree(self, node: TreeNode, dataset_id: str, doc_id: str) -> List[Document]:
        """Convert tree to list of Documents for VDB."""
        docs = []
        
        # Self (except root might be empty)
        if node.content or node.summary:
            # We index the SUMMARY for internal nodes, and CONTENT for leaf nodes
            # Ideally both, but for PageIndex we want "Index of Summaries".
            
            # Text to be embedded
            text_to_embed = node.summary if node.children else node.content
            
            # Metadata
            metadata = {
                "dataset_id": dataset_id,
                "document_id": doc_id,
                "node_id": node.id,
                "parent_id": node.parent.id if node.parent else None,
                "children_ids": [c.id for c in node.children],
                "level": node.level,
                "is_leaf": not bool(node.children),
                "is_summary": bool(node.children),
                "title": node.metadata.get("title", ""),
                "summary": node.summary
            }
            
            doc = Document(
                page_content=text_to_embed,
                metadata=metadata,
                type=DocumentType.CHUNK
            )
            docs.append(doc)
            
        for child in node.children:
            docs.extend(self._flatten_tree(child, dataset_id, doc_id))
            
        return docs

    def process(self, dataset: Dataset, documents: List[Document], **kwargs) -> None:
        """Process documents into PageIndex tree."""
        all_final_docs = []
        
        for doc in documents:
            if self.progress_callback:
                self.progress_callback(f"Parsing structure for {doc.metadata.get('document_name', 'doc')}")
                
            # 1. Parse
            root = self.parser.parse(doc.page_content, doc.metadata.get("document_id", str(uuid.uuid4())))
            
            # 2. Summarize (Async loop)
            # We run this synchronously here for simplicity of the interface, 
            # but internally it calls async LLM methods wrapped in loop if needed,
            # or we assume process() is run in thread. Use asyncio.run for now.
            if self.progress_callback:
                self.progress_callback(f"Generating summaries for {doc.metadata.get('document_name', 'doc')}")
            
            try:
                asyncio.run(self._summarize_node(root))
            except Exception as e:
                logger.error(f"Failed to summarize tree: {e}")
                # Continue best effort
            
            # 3. Flatten
            tree_docs = self._flatten_tree(root, dataset.id, doc.metadata.get("document_id"))
            
            # 4. Embed
            texts = [d.page_content for d in tree_docs]
            if texts:
                embeddings = self._embed_texts(texts)
                for i, d in enumerate(tree_docs):
                    d.vector = embeddings[i]
            
            all_final_docs.extend(tree_docs)

        # 5. Save
        if not all_final_docs:
            return

        manager = self.vector_manager
        if manager is None:
            # Fallback to global manager
            from langrag.datasource.vdb.global_manager import get_vector_manager
            manager = get_vector_manager()

        if self.progress_callback:
            self.progress_callback(f"Saving {len(all_final_docs)} nodes to vector store")
            
        manager.add_texts(dataset, all_final_docs)
