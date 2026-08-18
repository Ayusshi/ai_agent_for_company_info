from knowledge.knowledge_base import KnowledgeBase
from tools.tool_definition import SearchKnowledgeBaseInput


# -------------------------------------------------
# Knowledge Base
# -------------------------------------------------

knowledge_base = KnowledgeBase(
    documents_dir="data/documents",
    retrieval_k=3,
    retrieval_threshold=0.35,
)


# -------------------------------------------------
# Tool implementations
# -------------------------------------------------

def search_knowledge_base(query: str):

    results = knowledge_base.search(query)

    if not results:

        return {
            "found": False,
            "message": (
                "No relevant information was found "
                "in the company knowledge base."
            ),
        }

    formatted_results = []

    for result in results:

        formatted_results.append({
            "source": result["source"],
            "page": result["page"],
            "content": result["text"],
        })

    return {
        "found": True,
        "results": formatted_results,
    }


# -------------------------------------------------
# Tool Registry
# -------------------------------------------------

TOOL_REGISTRY = {

    "search_knowledge_base": {
        "function": search_knowledge_base,
        "schema": SearchKnowledgeBaseInput,
    },

}