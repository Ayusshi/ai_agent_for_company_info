from knowledge.knowledge_base import KnowledgeBase


knowledge_base = KnowledgeBase(
    documents_dir="data/documents",
    retrieval_k=3,
    retrieval_threshold=0.5,
)


query = "How many days of annual leave do employees receive?"

results = knowledge_base.search(query)


print("\nRetrieved Results:")
print("-" * 60)

for rank, result in enumerate(results, start=1):

    print(f"\nRank: {rank}")
    print(f"Score: {result['score']:.4f}")
    print(f"Source: {result['source']}")
    print(f"Page: {result['page']}")
    print(f"Chunk ID: {result['chunk_id']}")
    print(f"Text: {result['text']}")