company-ai-agent/
│
├── configs/
│   └── config.yaml
│
├── src/
│   ├── api/
│   │   └── routes.py
│   │
│   ├── agent/
│   │   ├── agent.py
│   │   ├── state.py
│   │   └── router.py
│   │
│   ├── tools/
│   │   ├── company_search.py
│   │   ├── web_search.py
│   │   ├── news_search.py
│   │   └── rag_retriever.py
│   │
│   ├── llm/
│   │   └── client.py
│   │
│   ├── services/
│   │   └── ...
│   │
│   └── utils/
│       └── ...
│
├── tests/
│
├── scripts/
│
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md