# AI Agent for Company Information

A production-oriented AI agent that answers company-related questions using
Retrieval-Augmented Generation (RAG), LangChain, and LangGraph.

The system combines a persistent FAISS-based knowledge base with an
LLM-powered agent that can dynamically choose between document retrieval
and other tools such as a calculator.

---

## Overview

Traditional RAG systems follow a fixed pipeline:

```text
User Question
      ↓
Retrieve Documents
      ↓
Generate Answer