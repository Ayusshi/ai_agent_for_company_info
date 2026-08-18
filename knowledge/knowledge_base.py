from pathlib import Path

from embeddings.embedder import create_embeddings
from ingestion.chunker import chunk_text
from ingestion.pdf_loader import load_pdf
from retrieval.vector_store import VectorStore


class KnowledgeBase:

    def __init__(
        self,
        documents_dir: str,
        retrieval_k: int = 3,
        retrieval_threshold: float = 0.5,
    ):
        self.documents_dir = Path(documents_dir)

        self.retrieval_k = retrieval_k
        self.retrieval_threshold = retrieval_threshold

        self.chunks = []
        self.vector_store = None

        self._build()

    # -------------------------------------------------
    # Build knowledge base
    # -------------------------------------------------

    def _build(self):

        pdf_files = list(
            self.documents_dir.glob("*.pdf")
        )

        if not pdf_files:
            raise ValueError(
                f"No PDF documents found in {self.documents_dir}"
            )

        all_chunks = []

        # ---------------------------------------------
        # Load and chunk every document
        # ---------------------------------------------

        for pdf_path in pdf_files:

            print(f"Loading document: {pdf_path.name}")

            pages = load_pdf(str(pdf_path))

            chunks = chunk_text(pages)

            all_chunks.extend(chunks)

        self.chunks = all_chunks

        # ---------------------------------------------
        # Create embeddings for all chunks
        # ---------------------------------------------

        chunk_texts = [
            chunk["text"]
            for chunk in self.chunks
        ]

        embeddings = create_embeddings(chunk_texts)

        # ---------------------------------------------
        # Create vector store
        # ---------------------------------------------

        self.vector_store = VectorStore(
            dimension=embeddings.shape[1]
        )

        # ---------------------------------------------
        # Add embeddings to FAISS
        # ---------------------------------------------

        self.vector_store.add_embeddings(
            embeddings
        )

        print(
            f"Knowledge base built successfully "
            f"with {len(self.chunks)} chunks."
        )

    # -------------------------------------------------
    # Search knowledge base
    # -------------------------------------------------

    def search(self, query: str):

        query_embedding = create_embeddings(
            [query]
        )

        scores, indices = self.vector_store.search(
            query_embedding,
            k=self.retrieval_k,
            threshold=self.retrieval_threshold,
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0]
        ):

            chunk = self.chunks[index]

            results.append({
                "score": float(score),
                "chunk_id": chunk["chunk_id"],
                "source": chunk["source"],
                "page": chunk["page"],
                "text": chunk["text"],
            })

        return results