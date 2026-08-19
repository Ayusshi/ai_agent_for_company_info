from pathlib import Path
from typing import TypedDict

from pydantic import BaseModel

from langchain_ollama import ChatOllama

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)

from langchain_core.tools import tool

from langgraph.graph import (
    StateGraph,
    START,
    END,
    MessagesState,
)

from langgraph.prebuilt import (
    ToolNode,
    tools_condition,
)

from langgraph.checkpoint.memory import (
    InMemorySaver,
)

from knowledge.knowledge_base import KnowledgeBase


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"


# ============================================================
# Knowledge Base
# ============================================================

knowledge_base = KnowledgeBase(
    documents_dir=str(DOCUMENTS_DIR),
    index_dir=str(
        PROJECT_ROOT / "data" / "index"
    ),
    retrieval_k=3,
    retrieval_threshold=0.5,
)


# ============================================================
# LLM
# ============================================================

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0,
)


# ============================================================
# Tool Input Schema
# ============================================================

class SearchKnowledgeBaseInput(BaseModel):
    query: str


# ============================================================
# RAG Tool
# ============================================================

@tool(args_schema=SearchKnowledgeBaseInput)
def search_knowledge_base(query: str) -> str:
    """
    Search the company knowledge base for information
    contained in internal company documents.

    Use this tool when the user asks about company
    policies, procedures, employee information,
    security, expenses, leave, remote work, or other
    company-specific information.
    """

    results = knowledge_base.search(query)

    if not results:
        return (
            "No relevant information was found in the "
            "company knowledge base."
        )

    formatted_results = []

    for result in results:

        formatted_results.append(
            (
                f"Source: {result['source']}\n"
                f"Page: {result['page']}\n"
                f"Score: {result['score']:.3f}\n"
                f"Content:\n{result['text']}"
            )
        )

    return "\n\n---\n\n".join(
        formatted_results
    )


# ============================================================
# Calculator Tool
# ============================================================

@tool
def calculator(expression: str) -> str:
    """
    Calculate a mathematical expression.

    Use this tool when arithmetic or numerical
    calculation is required.
    """

    try:

        result = eval(expression)

        return str(result)

    except Exception:

        return (
            "Unable to calculate the expression."
        )


# ============================================================
# Tools
# ============================================================

tools = [
    search_knowledge_base,
    calculator,
]


# ============================================================
# LLM with Tools
# ============================================================

llm_with_tools = llm.bind_tools(
    tools
)


# ============================================================
# Agent State
# ============================================================

class AgentState(MessagesState):
    pass


# ============================================================
# Model Node
# ============================================================

def call_model(state: AgentState):

    system_message = SystemMessage(
        content=(
            "You are a helpful company assistant.\n\n"

            "You have access to the following tools:\n\n"

            "1. search_knowledge_base:\n"
            "Searches internal company documents.\n\n"

            "2. calculator:\n"
            "Performs mathematical calculations.\n\n"

            "Use search_knowledge_base when the user "
            "asks about company-specific information.\n\n"

            "Use calculator when arithmetic is required.\n\n"

            "Do not use tools unnecessarily.\n\n"

            "When answering company-specific questions, "
            "use information returned by the knowledge "
            "base and do not invent facts."
        )
    )

    messages = [
        system_message,
        *state["messages"],
    ]

    response = llm_with_tools.invoke(
        messages
    )

    return {
        "messages": [response]
    }


# ============================================================
# Build Graph
# ============================================================

graph = StateGraph(
    AgentState
)


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

graph.add_node(
    "model",
    call_model,
)


# ------------------------------------------------------------
# Tools
# ------------------------------------------------------------

graph.add_node(
    "tools",
    ToolNode(tools),
)


# ------------------------------------------------------------
# START → MODEL
# ------------------------------------------------------------

graph.add_edge(
    START,
    "model",
)


# ------------------------------------------------------------
# MODEL → TOOLS or END
# ------------------------------------------------------------

graph.add_conditional_edges(
    "model",
    tools_condition,
    {
        "tools": "tools",
        END: END,
    },
)


# ------------------------------------------------------------
# TOOLS → MODEL
# ------------------------------------------------------------

graph.add_edge(
    "tools",
    "model",
)


# ============================================================
# Checkpointing
# ============================================================

checkpointer = InMemorySaver()


# ============================================================
# Compile Agent
# ============================================================

agent = graph.compile(
    checkpointer=checkpointer
)


# ============================================================
# Public Function
# ============================================================

def ask_agent(
    question: str,
    thread_id: str = "default",
):

    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content=question
                )
            ]
        },
        {
            "configurable": {
                "thread_id": thread_id
            }
        },
    )

    return result["messages"][-1].content