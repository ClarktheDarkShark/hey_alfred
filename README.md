# Alfred: Retrieval-Augmented Generation (RAG) AI Agent

**Alfred** is a Python-based AI agent framework designed for long-term memory management, document retrieval, and dynamic query resolution. This project integrates Pinecone and LangChain technologies to enhance RAG capabilities, enabling intelligent workflows and seamless interaction with various data sources.

## Features

- **Memory Management**: Core and recall memory system using Pinecone for semantic search and metadata filtering.
- **RAG Workflow**: Supports PDF, CSV, Excel, and Word document retrieval and similarity-based searches. (BASIC RAG WORKFLOW STABLE AND FUNCTIONAL. ADVANCED RAG IN DEVELOPMENT. See git logs.)
- **Dynamic Tooling**: Modular tools for METAR/TAF aviation weather, current news, arithmetic, date handling, and unit conversion.
- **Flexible Architecture**: Built with LangGraph for workflow orchestration and easily extendable via submodules.
- **Automation**: Supports automated document ingestion for efficient data updates.

## Project Structure

# Open Source repos: Lang-MemGPT and Langgraph-course.

1. Alfred was built on the Lang-MemGPT repo --> https://github.com/langchain-ai/lang-memgpt
2. The RAG portion of Alfred is from the following repo --> https://github.com/emarco177/langgaph-course. 

Inspired by papers like [MemGPT](https://memgpt.ai/), the graph
extracts memories from chat interactions and persists them to a database. This information can later be read or queried semantically
to provide personalized context when your bot is responding to a particular user.

The memory graph handles thread process deduplication and supports continuous updates to a single "memory schema" as well as "event-based" memories that can be queried semantically.


#### Initial Project Structure prior to Alfred Structure:

```bash
├── langgraph.json # LangGraph Cloud Configuration
├── lang_memgpt
│   ├── __init__.py
│   └── graph.py # Define the agent w/ memory
├── poetry.lock
├── pyproject.toml # Project dependencies
└── tests # Add testing + evaluation logic
    └── evals
        └── test_memories.py
```


#### Prerequisites

This example defaults to using Pinecone for its memory database, and OPENAI's GPT family of LLMs.

Before starting, make sure your resources are created.

1. [Create an index](https://docs.pinecone.io/reference/api/control-plane/create_index) with a dimension size of `768`. Note down your Pinecone API key, index name, and namespac for the next step.
2. [Create an API Key] with OPENAI to use for the LLM & embeddings models.

#### LangChain Academy (I highly recommend you go through the Academy course to better understand LangGraph and Langsmith!)
1. https://academy.langchain.com/
2. LangGraph Tutorial/Overview: https://www.youtube.com/watch?v=29XE10U6ooc

#### Download LangGraph Studio
1. Frontend UI: https://studio.langchain.com/.
2. LangGraph Series: https://www.youtube.com/playlist?list=PLfaIDFEXuae16n2TWUkKq5PgJ0w6Pkwtg

####  Langsmith 
1. Frontend UI: https://www.langchain.com/langsmith. 
2. Tutorial/Overview: https://www.youtube.com/watch?v=KcGoJPK-0tY&t=1s

```bash
# .env
PINECONE_API_KEY=<your-pinecone-api-key> # The baseline API is free but has limited usage. Get API key at —> https://www.pinecone.io/
PINECONE_INDEX_NAME=alfred-index # You can change when you setup your pinecone account if desired. 
PINECONE_NAMESPACE=default # You can change whne you setup your pinecone account if desired. 
PINECONE_ENVIRONMENT=eastus2 # Those on the west coast may want to select a different enviorment closer to you all. 

OPENAI_API_KEY=<your-openai-api-key> # You will have to pay for this key. I have spent around $5.00 over the last few weeks experimenting. It’s worth it!! Get API key at —> https://platform.openai.com/docs/overview

TAVILY_API_KEY=<your-tavily-api-key> # Free API key. This allows the agent to search the internet. Get API key at —> https://tavily.com/

# LangChain / LangSmith Configuration
LANGCHAIN_API_KEY=<your-langchain-api-key> # Free API key. Get API key at —> https://docs.smith.langchain.com/administration/how_to_guides/organization_management/create_account_api_key
LANGCHAIN_TRACING_V2=true  # Enables LangSmith tracing. This is how you debug your agent and figure out what is going on with the memory state. 
LANGCHAIN_PROJECT=Alfredv2 # Sets project name in LangSmith


NEWSDATA_API_KEY=<your-newsdata-api-key> # Free key up to 200 calls a day. Searches the internet but with a focus on newsites only. Get API key at —>  https://newsdata.io/****
