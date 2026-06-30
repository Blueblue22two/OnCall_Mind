"""Build an isolated Enhanced RAG collection for a chunking ablation."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--collection", help="Enhanced collection name")
    target.add_argument("--basic", action="store_true", help="Rebuild the fixed Basic collection")
    parser.add_argument("--strategy", choices=["legacy", "section_child"], required=True)
    parser.add_argument("--include-section-prefix", action="store_true")
    parser.add_argument("--docs-dir", default="aiops-docs")
    args = parser.parse_args()

    # Settings is instantiated during app imports, so configure the variant first.
    if args.collection:
        os.environ["ENHANCED_COLLECTION_NAME"] = args.collection
    os.environ["RAG_CHUNK_STRATEGY"] = args.strategy
    os.environ["RAG_INCLUDE_SECTION_PREFIX"] = str(args.include_section_prefix).lower()

    from app.services.document_splitter_service import document_splitter_service
    if args.basic:
        from app.services.vector_store_manager import vector_store_manager

        store_manager = vector_store_manager
        target_name = "biz"
    else:
        from app.core.milvus_client import milvus_manager
        from app.services.enhanced_vector_store_manager import enhanced_vector_store_manager

        # Enhanced manager is intentionally lazy and does not connect on import.
        # Connect after the collection environment override so the requested
        # versioned A/B collection is created and loaded.
        milvus_manager.connect()
        store_manager = enhanced_vector_store_manager
        target_name = args.collection

    docs_dir = Path(args.docs_dir).resolve()
    files = sorted(docs_dir.glob("*.md"))
    if not files:
        raise SystemExit(f"No Markdown documents found in {docs_dir}")

    total_chunks = 0
    for path in files:
        source = path.as_posix()
        chunks = document_splitter_service.split_document(
            path.read_text(encoding="utf-8"),
            source,
            strategy=args.strategy,
            include_section_prefix=args.include_section_prefix,
        )
        store_manager.delete_by_source(source)
        store_manager.add_documents(chunks)
        total_chunks += len(chunks)
        print(f"{path.name}: {len(chunks)} chunks")

    print(
        f"Indexed {len(files)} files / {total_chunks} chunks into {target_name} "
        f"({args.strategy}, prefix={args.include_section_prefix})"
    )


if __name__ == "__main__":
    main()
