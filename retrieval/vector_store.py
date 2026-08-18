import faiss
import numpy as np


class VectorStore:

    def __init__(self, dimension=None):
        self.index = (
            faiss.IndexFlatIP(dimension)
            if dimension is not None
            else None
        )

    def add_embeddings(self, embeddings):

        embeddings = np.asarray(
            embeddings
        ).astype("float32")

        self.index.add(embeddings)

    def search(
        self,
        query_embedding,
        k=3,
        threshold=None
    ):

        query_embedding = np.asarray(
            query_embedding
        ).astype("float32")

        scores, indices = self.index.search(
            query_embedding,
            k
        )

        if threshold is not None:

            mask = scores[0] >= threshold

            scores = scores[:, mask]
            indices = indices[:, mask]

        return scores, indices

    def save(self, path):

        faiss.write_index(
            self.index,
            str(path)
        )

    @classmethod
    def load(cls, path):

        store = cls()

        store.index = faiss.read_index(
            str(path)
        )

        return store