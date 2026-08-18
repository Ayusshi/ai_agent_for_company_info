import faiss
import numpy as np


class VectorStore:

    def __init__(self, dimension):
        self.index = faiss.IndexFlatIP(dimension)

    def add_embeddings(self, embeddings):
        embeddings = np.asarray(embeddings).astype("float32")
        self.index.add(embeddings)

    def search(self, query_embedding, k=3, threshold=None):

     query_embedding = np.asarray(query_embedding).astype("float32")

     scores, indices = self.index.search(
        query_embedding,
        k
      )

     if threshold is not None:
        mask = scores[0] >= threshold
        scores = scores[:, mask]
        indices = indices[:, mask]

     return scores, indices