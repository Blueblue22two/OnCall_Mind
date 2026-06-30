"""Document splitting with legacy and section-aware parent-child strategies."""

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from loguru import logger

from app.config import config


class DocumentSplitterService:
    """文档分割服务 - 使用 LangChain 的分割器"""

    def __init__(self):
        """初始化文档分割服务"""
        self.chunk_size = config.chunk_max_size
        self.chunk_overlap = config.chunk_overlap

        # Markdown 标题分割器 (只按一级和二级标题分割，减少分片数)
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2"),
                # 不再按三级标题分割，避免过度碎片化
            ],
            strip_headers=False,  # 保留标题在内容中
        )

        self.section_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
            strip_headers=False,
        )

        # 递归字符分割器 (用于二次分割，使用更大的chunk_size)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size * 2,  # 加倍chunk_size，减少分片数
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

        logger.info(
            f"文档分割服务初始化完成, chunk_size={self.chunk_size}, "
            f"secondary_chunk_size={self.chunk_size * 2}, "
            f"overlap={self.chunk_overlap}"
        )

    def split_markdown(
        self,
        content: str,
        file_path: str = "",
        strategy: str | None = None,
        include_section_prefix: bool | None = None,
    ) -> List[Document]:
        """
        分割 Markdown 文档 (两阶段分割 + 合并小片段)

        Args:
            content: Markdown 内容
            file_path: 文件路径 (用于元数据)

        Returns:
            List[Document]: 文档分片列表
        """
        if not content or not content.strip():
            logger.warning(f"Markdown 文档内容为空: {file_path}")
            return []

        try:
            active_strategy = strategy or config.rag_chunk_strategy
            include_prefix = (
                config.rag_include_section_prefix
                if include_section_prefix is None
                else include_section_prefix
            )
            if active_strategy == "section_child":
                return self._split_markdown_parent_child(
                    content, file_path, include_section_prefix=include_prefix
                )

            # Legacy strategy: split by H1/H2, then by size.
            md_docs = self.markdown_splitter.split_text(content)

            # 第二阶段: 按大小进一步分割
            docs_after_split = self.text_splitter.split_documents(md_docs)

            # 第三阶段: 合并太小的分片 (< 300字符)
            final_docs = self._merge_small_chunks(docs_after_split, min_size=300)

            # 添加文件路径元数据
            for doc in final_docs:
                self._enrich_metadata(
                    doc,
                    file_path,
                    parent_content=doc.page_content,
                    include_section_prefix=include_prefix,
                )

            logger.info(f"Markdown 分割完成: {file_path} -> {len(final_docs)} 个分片")
            return final_docs

        except Exception as e:
            logger.error(f"Markdown 分割失败: {file_path}, 错误: {e}")
            raise

    @staticmethod
    def _section_type(h2: str) -> str:
        mapping = {
            "告警名称": "trigger",
            "排查步骤": "procedure",
            "常用命令": "commands",
            "相关工具命令": "commands",
            "常见原因分析": "causes",
            "紧急处理措施": "remediation",
            "紧急处理流程": "remediation",
            "验证步骤": "verification",
        }
        return mapping.get(h2.strip(), "general")

    @staticmethod
    def _stable_id(*parts: str) -> str:
        raw = "||".join(parts)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]

    def _enrich_metadata(
        self,
        doc: Document,
        file_path: str,
        parent_content: str,
        ordinal: int = 0,
        include_section_prefix: bool | None = None,
    ) -> None:
        metadata = doc.metadata
        h1 = str(metadata.get("h1", ""))
        h2 = str(metadata.get("h2", ""))
        h3 = str(metadata.get("h3", ""))
        file_name = Path(file_path).name
        section_path = " > ".join(part for part in (h1, h2, h3) if part)
        parent_id = self._stable_id(file_path, h1, h2)
        content_hash = hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest()
        metadata.update(
            {
                "_source": file_path,
                "_extension": Path(file_path).suffix or ".md",
                "_file_name": file_name,
                "chunk_id": self._stable_id(file_path, section_path, str(ordinal), content_hash),
                "parent_id": parent_id,
                "section_path": section_path,
                "section_type": self._section_type(h2),
                "chunk_ordinal": ordinal,
                "content_hash": content_hash,
                "_parent_content": parent_content[: config.rag_parent_context_max_chars],
            }
        )
        use_prefix = (
            config.rag_include_section_prefix
            if include_section_prefix is None
            else include_section_prefix
        )
        if use_prefix and section_path:
            doc.page_content = f"文档: {file_name}\n章节: {section_path}\n\n{doc.page_content}"

    def _split_markdown_parent_child(
        self,
        content: str,
        file_path: str,
        include_section_prefix: bool | None = None,
    ) -> List[Document]:
        """Retrieve H3/size-bounded children while retaining their H2 parent."""
        section_docs = self.section_splitter.split_text(content)
        grouped: dict[tuple[str, str], list[Document]] = defaultdict(list)
        for doc in section_docs:
            key = (str(doc.metadata.get("h1", "")), str(doc.metadata.get("h2", "")))
            grouped[key].append(doc)

        children: list[Document] = []
        ordinal_by_parent: dict[tuple[str, str], int] = defaultdict(int)
        for parent_key, siblings in grouped.items():
            parent_content = "\n\n".join(doc.page_content for doc in siblings)
            split_siblings: list[Document] = []
            for sibling in siblings:
                split_siblings.extend(self.child_splitter.split_documents([sibling]))
            split_siblings = self._merge_small_chunks(split_siblings, min_size=180)
            for child in split_siblings:
                ordinal = ordinal_by_parent[parent_key]
                ordinal_by_parent[parent_key] += 1
                self._enrich_metadata(
                    child,
                    file_path,
                    parent_content,
                    ordinal,
                    include_section_prefix=include_section_prefix,
                )
                children.append(child)

        logger.info(
            f"Section-aware Markdown 分割完成: {file_path} -> {len(children)} 个 child"
        )
        return children

    def split_text(self, content: str, file_path: str = "") -> List[Document]:
        """
        分割普通文本文档

        Args:
            content: 文本内容
            file_path: 文件路径 (用于元数据)

        Returns:
            List[Document]: 文档分片列表
        """
        if not content or not content.strip():
            logger.warning(f"文本文档内容为空: {file_path}")
            return []

        try:
            # 直接使用递归字符分割器
            docs = self.text_splitter.create_documents(
                texts=[content],
                metadatas=[
                    {
                        "_source": file_path,
                        "_extension": Path(file_path).suffix,
                        "_file_name": Path(file_path).name,
                    }
                ],
            )

            logger.info(f"文本分割完成: {file_path} -> {len(docs)} 个分片")
            return docs

        except Exception as e:
            logger.error(f"文本分割失败: {file_path}, 错误: {e}")
            raise

    def split_document(
        self,
        content: str,
        file_path: str = "",
        strategy: str | None = None,
        include_section_prefix: bool | None = None,
    ) -> List[Document]:
        """
        智能分割文档 (根据文件类型选择分割器)

        Args:
            content: 文档内容
            file_path: 文件路径

        Returns:
            List[Document]: 文档分片列表
        """
        if file_path.endswith(".md"):
            return self.split_markdown(
                content,
                file_path,
                strategy=strategy,
                include_section_prefix=include_section_prefix,
            )
        else:
            return self.split_text(content, file_path)

    def _merge_small_chunks(
        self, documents: List[Document], min_size: int = 300
    ) -> List[Document]:
        """
        合并太小的分片

        Args:
            documents: 文档列表
            min_size: 最小分片大小 (字符数)

        Returns:
            List[Document]: 合并后的文档列表
        """
        if not documents:
            return []

        merged_docs = []
        current_doc = None

        for doc in documents:
            doc_size = len(doc.page_content)

            if current_doc is None:
                # 第一个文档
                current_doc = doc
            elif (
                doc_size < min_size
                and len(current_doc.page_content) < self.chunk_size * 2
                and self._same_parent(current_doc, doc)
            ):
                # 当前文档太小且合并后不会太大，则合并
                current_doc.page_content += "\n\n" + doc.page_content
                # 保留主文档的元数据
            else:
                # 保存当前文档，开始新文档
                merged_docs.append(current_doc)
                current_doc = doc

        # 添加最后一个文档
        if current_doc is not None:
            merged_docs.append(current_doc)

        return merged_docs

    @staticmethod
    def _same_parent(left: Document, right: Document) -> bool:
        return all(
            left.metadata.get(key, "") == right.metadata.get(key, "")
            for key in ("h1", "h2", "h3")
        )


# 全局单例
document_splitter_service = DocumentSplitterService()
