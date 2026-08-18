from pydantic import BaseModel


# -------------------------------------------------
# Tool input schemas
# -------------------------------------------------

class SearchKnowledgeBaseInput(BaseModel):
    query: str


# -------------------------------------------------
# Tool definitions sent to the LLM
# -------------------------------------------------

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "Search the company knowledge base for information "
                "contained in the company's internal documents. "
                "Use this tool when the user asks about company "
                "policies, procedures, employee information, "
                "security, expenses, leave, remote work, or other "
                "information that may be present in the company documents."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "The question or information to search "
                            "for in the company knowledge base."
                        ),
                    }
                },
                "required": ["query"],
            },
        },
    }
]