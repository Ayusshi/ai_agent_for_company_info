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
            "context": "",
            "sources": [],
            "message": (
                "No relevant information was found "
                "in the company knowledge base."
            ),
        }

    context_parts = []
    sources = []

    for result in results:

        context_parts.append(
            f"SOURCE: {result['source']}\n"
            f"PAGE: {result['page']}\n\n"
            f"{result['text']}"
        )

        sources.append({
            "source": result["source"],
            "page": result["page"],
        })

    context = "\n\n---\n\n".join(context_parts)

    return {
        "found": True,
        "context": context,
        "sources": sources,
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