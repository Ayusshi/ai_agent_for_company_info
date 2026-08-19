from pathlib import Path
import json

from embeddings.embedder import create_embeddings
from ingestion.chunker import chunk_text
from ingestion.pdf_loader import load_pdf
from retrieval.vector_store import VectorStore


class KnowledgeBase:

    def __init__(
        self,
        documents_dir: str,
        index_dir: str = "data/index",
        retrieval_k: int = 3,
        retrieval_threshold: float = 0.5,
    ):

        self.documents_dir = Path(documents_dir)
        self.index_dir = Path(index_dir)

        self.retrieval_k = retrieval_k
        self.retrieval_threshold = retrieval_threshold

        self.chunks = []
        self.vector_store = None

        self.index_path = (
            self.index_dir / "faiss.index"
        )

        self.chunks_path = (
            self.index_dir / "chunks.json"
        )

        self._load_or_build()

    # =========================================================
    # Load existing KB or build a new one
    # =========================================================

    def _load_or_build(self):

        if (
            self.index_path.exists()
            and self.chunks_path.exists()
        ):

            print(
                "Loading existing knowledge base..."
            )

            self._load()

        else:

            print(
                "No persistent knowledge base found."
            )

            print(
                "Building knowledge base..."
            )

            self._build()

    # =========================================================
    # Build knowledge base
    # =========================================================

    def _build(self):

        pdf_files = list(
            self.documents_dir.glob("*.pdf")
        )

        if not pdf_files:

            raise ValueError(
                f"No PDF documents found in "
                f"{self.documents_dir}"
            )

        all_chunks = []

        # -----------------------------------------------------
        # Load and chunk documents
        # -----------------------------------------------------

        for pdf_path in pdf_files:

            print(
                f"Loading document: "
                f"{pdf_path.name}"
            )

            pages = load_pdf(
                str(pdf_path)
            )

            chunks = chunk_text(
                pages
            )

            all_chunks.extend(
                chunks
            )

        self.chunks = all_chunks

        # -----------------------------------------------------
        # Create embeddings
        # -----------------------------------------------------

        chunk_texts = [
            chunk["text"]
            for chunk in self.chunks
        ]

        embeddings = create_embeddings(
            chunk_texts
        )

        # -----------------------------------------------------
        # Create FAISS store
        # -----------------------------------------------------

        self.vector_store = VectorStore(
            dimension=embeddings.shape[1]
        )

        self.vector_store.add_embeddings(
            embeddings
        )

        # -----------------------------------------------------
        # Persist
        # -----------------------------------------------------

        self._save()

        print(
            f"Knowledge base built successfully "
            f"with {len(self.chunks)} chunks."
        )

    # =========================================================
    # Save
    # =========================================================

    def _save(self):

        self.index_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.vector_store.save(
            self.index_path
        )

        with open(
            self.chunks_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                self.chunks,
                file,
                ensure_ascii=False,
                indent=2,
            )

        print(
            f"Knowledge base saved to "
            f"{self.index_dir}"
        )

    # =========================================================
    # Load
    # =========================================================

    def _load(self):

        self.vector_store = (
            VectorStore.load(
                self.index_path
            )
        )

        with open(
            self.chunks_path,
            "r",
            encoding="utf-8",
        ) as file:

            self.chunks = json.load(
                file
            )

        print(
            f"Knowledge base loaded "
            f"with {len(self.chunks)} chunks."
        )

    # =========================================================
    # Search
    # =========================================================

    def search(
        self,
        query: str
    ):

        query_embedding = create_embeddings(
            [query]
        )

        scores, indices = (
            self.vector_store.search(
                query_embedding,
                k=self.retrieval_k,
                threshold=self.retrieval_threshold,
            )
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0]
        ):

            # -------------------------------------------------
            # Safety check
            # -------------------------------------------------

            if index < 0:
                continue

            chunk = self.chunks[index]

            results.append(
                {
                    "score": float(score),
                    "chunk_id": chunk["chunk_id"],
                    "source": chunk["source"],
                    "page": chunk["page"],
                    "text": chunk["text"],
                }
            )

        return results