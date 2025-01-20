# graph.py
"""Lang-MemGPT: A Long-Term Memory Agent.

This module implements an agent with long-term memory capabilities using LangGraph.
The agent can store, retrieve, and use memories to enhance its interactions with users.

Key Components:
1. Memory Types: Core (always available) and Recall (contextual/semantic).
2. Tools: For saving and retrieving memories + performing other tasks (METAR, TAF, etc.).
3. Memory Vector Database: Uses Pinecone for both core and recall memories.
4. RAG Vector Database: For querying PDF and CSV documents. Uses Chroma or other vector DB.

Configuration: Requires Pinecone and OpenAI API keys (see README for setup).

This code integrates the original multi-step RAG pipeline with a simplified UI-facing
`process_chat` function, ensuring both the TAF tool and the entire RAG flow are available.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional, Tuple, List, Dict, Any
import sys

import langsmith
import tiktoken
from langchain.chat_models import init_chat_model
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_core.messages.utils import get_buffer_string
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.config import (
    RunnableConfig,
    ensure_config,
    get_executor_for_config,
)
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from typing_extensions import Literal

# -- Local modules
from lang_memgpt import _constants as constants
from lang_memgpt import _schemas as schemas
from lang_memgpt import _settings as settings
from lang_memgpt import _utils as utils

# -- RAG pipeline logic
# Instead of importing "generate" from __init__.py, we import it directly:
from lang_memgpt.RAG_Structure.decision_logic import decide_to_generate
from lang_memgpt.RAG_Structure.nodes.generate import generate
# route_question, retrieve, etc. are also imported directly from their modules:
from lang_memgpt.RAG_Structure.nodes.ingestion import ingest_data as _ingest_data
from lang_memgpt.RAG_Structure.nodes.retrieve import retrieve as _retrieve
from lang_memgpt.RAG_Structure.nodes.grade_documents import grade_documents as _grade_documents
from lang_memgpt.RAG_Structure.nodes.web_search import web_search as _web_search
from lang_memgpt.RAG_Structure.route_question import route_question as _route_question
from lang_memgpt.RAG_Structure.grade_generation import (
    grade_generation_grounded_in_documents_and_question as _grade_generation
)

# -- Tools from lang_memgpt/tools
from lang_memgpt.tools import (
    get_metar_data,
    get_taf_data,
    calculate,
    unit_converter,
    date_time_tool,
    fetch_latest_news
)

logging.basicConfig(
    level=logging.DEBUG,  # Use DEBUG to capture all logs
    stream=sys.stdout,    # Ensure logs are sent to stdout
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)

logger = logging.getLogger("memory")
# logging.basicConfig(level=logging.DEBUG)


# ------------------------------------------------------------------------------
# Fix #1: Provide docstrings (or "description") for all RAG-related tools
# ------------------------------------------------------------------------------
@tool
def wrapped_ingest_data(*args, **kwargs):
    """
    Ingest user-supplied documents for analysis and storage.
    This docstring ensures no 'Function must have a docstring' error occurs.
    """
    return ingest_data(*args, **kwargs)

@tool
def wrapped_route_question(*args, **kwargs):
    """
    Route a user query to the correct RAG branch (retrieve or web search).
    """
    return route_question(*args, **kwargs)

@tool
def wrapped_retrieve(*args, **kwargs):
    """
    Retrieve relevant documents from the vector DB based on a query.
    """
    return retrieve(*args, **kwargs)

@tool
def wrapped_web_search(*args, **kwargs):
    """
    Perform a web search for additional info on the query.
    """
    return web_search(*args, **kwargs)

@tool
def wrapped_grade_documents(*args, **kwargs):
    """
    Grade candidate documents for relevance and correctness.
    """
    return grade_documents(*args, **kwargs)

@tool
def wrapped_grade_generation(*args, **kwargs):
    """
    Grade the AI-generated answer for factual grounding given the question and relevant documents.
    """
    return grade_generation_grounded_in_documents_and_question(*args, **kwargs)


# ------------------------------------------------------------------------------
# A small non-zero vector to avoid certain DB issues (workaround).
_EMPTY_VEC = [0.00001] * 1536

# Initialize the TavilySearchResults tool
search_tool = TavilySearchResults(max_results=1)
tools = [search_tool]


#
# -----------------------------
#  Memory & Core Tools
# -----------------------------
#
@tool
async def save_recall_memory(memory: str) -> str:
    """
    Save a memory to the database for later semantic retrieval.

    Args:
        memory: The memory to be saved.

    Returns:
        The saved memory.
    """
    config = ensure_config()
    configurable = utils.ensure_configurable(config)
    embeddings = utils.get_embeddings()
    vector = await embeddings.aembed_query(memory)
    current_time = datetime.now(tz=timezone.utc)
    path = constants.INSERT_PATH.format(
        user_id=configurable["user_id"],
        event_id=str(uuid.uuid4()),
    )
    documents = [
        {
            "id": path,
            "values": vector,
            "metadata": {
                constants.PAYLOAD_KEY: memory,
                constants.PATH_KEY: path,
                constants.TIMESTAMP_KEY: current_time,
                constants.TYPE_KEY: "recall",
                "user_id": configurable["user_id"],
            },
        }
    ]
    utils.get_index().upsert(
        vectors=documents,
        namespace=settings.SETTINGS.pinecone_namespace,
    )
    return memory


@tool
def search_memory(query: str, top_k: int = 5) -> List[str]:
    """
    Search for memories in the database based on semantic similarity.

    Args:
        query: The search query.
        top_k: Number of results to return.

    Returns:
        A list of relevant memories.
    """
    config = ensure_config()
    configurable = utils.ensure_configurable(config)
    embeddings = utils.get_embeddings()
    vector = embeddings.embed_query(query)

    with langsmith.trace("query", inputs={"query": query, "top_k": top_k}) as rt:
        response = utils.get_index().query(
            vector=vector,
            filter={
                "user_id": {"$eq": configurable["user_id"]},
                constants.TYPE_KEY: {"$eq": "recall"},
            },
            namespace=settings.SETTINGS.pinecone_namespace,
            include_metadata=True,
            top_k=top_k,
        )
        rt.end(outputs={"response": response})
    memories = []
    if matches := response.get("matches"):
        memories = [m["metadata"][constants.PAYLOAD_KEY] for m in matches]
    return memories


@langsmith.traceable
def fetch_core_memories(user_id: str) -> Tuple[str, List[str]]:
    """
    Fetch core memories for a specific user.

    Args:
        user_id: The ID of the user.

    Returns:
        A tuple of (path, list of core memories).
    """
    path = constants.PATCH_PATH.format(user_id=user_id)
    response = utils.get_index().fetch(
        # "core_memories" replaced settings.SETTINGS.pinecone_namespace
        ids=[path], namespace="core_memories"
    )
    memories = []
    if vectors := response.get("vectors"):
        document = vectors[path]
        payload = document["metadata"][constants.PAYLOAD_KEY]
        memories = json.loads(payload)["memories"]
    return path, memories


@tool
def store_core_memory(memory: str, index: Optional[int] = None) -> str:
    """
    Store a core memory in the database.

    Args:
        memory: The memory to store.
        index: The index at which to store the memory (optional).

    Returns:
        A confirmation message.
    """
    config = ensure_config()
    configurable = utils.ensure_configurable(config)
    path, memories = fetch_core_memories(configurable["user_id"])
    if index is not None:
        if index < 0 or index >= len(memories):
            return "Error: Index out of bounds."
        memories[index] = memory
    else:
        memories.insert(0, memory)
    documents = [
        {
            "id": path,
            "values": _EMPTY_VEC,
            "metadata": {
                constants.PAYLOAD_KEY: json.dumps({"memories": memories}),
                constants.PATH_KEY: path,
                constants.TIMESTAMP_KEY: datetime.now(tz=timezone.utc),
                constants.TYPE_KEY: "core",
                "user_id": configurable["user_id"],
            },
        }
    ]
    utils.get_index().upsert(
        vectors=documents,
        # "core_memories" replaced settings.SETTINGS.pinecone_namespace
        namespace="core_memories"
    )
    return "Memory stored."


#
# -----------------------------
#  Combine Tools & Prompt
# -----------------------------
#
# Now re-define a single all_tools with guaranteed docstrings:
all_tools = tools + [
    save_recall_memory,
    search_memory,
    store_core_memory,
    get_metar_data,
    get_taf_data,
    calculate,
    unit_converter,
    date_time_tool,
    fetch_latest_news,

    # These wrapped versions each have a description/docstring:
    wrapped_ingest_data,
    wrapped_route_question,
    wrapped_retrieve,
    wrapped_web_search,
    wrapped_grade_documents,
    wrapped_grade_generation,
]


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant called Alfred with advanced long-term memory"
            " capabilities. Powered by a stateless LLM, you must rely on"
            " external memory to store information between conversations."
            " Utilize the available memory tools to store and retrieve"
            " important details that will help you better attend to the user's"
            " needs and understand their context.\n\n"
            "Memory Usage Guidelines:\n"
            "1. Actively use memory tools (save_core_memory, save_recall_memory)"
            " to build a comprehensive understanding of the user.\n"
            "2. Make informed suppositions and extrapolations based on stored"
            " memories.\n"
            "3. Regularly reflect on past interactions to identify patterns and"
            " preferences.\n"
            "4. Update your mental model of the user with each new piece of"
            " information.\n"
            "5. Cross-reference new information with existing memories for"
            " consistency.\n"
            "6. Prioritize storing emotional context and personal values"
            " alongside facts.\n"
            "7. Use memory to anticipate needs and tailor responses to the"
            " user's style.\n"
            "8. Recognize and acknowledge changes in the user's situation or"
            " perspectives over time.\n"
            "9. Leverage memories to provide personalized examples and"
            " analogies.\n"
            "10. Recall past challenges or successes to inform current"
            " problem-solving.\n"
            "11. Alfred can also process (i.e., ingest documents), index, and query PDFs, CSVs,"
            " Excel, or Word docs stored in a dedicated document database for research.\n"
            "12. Use document processing tools to analyze uploaded PDFs or Excel files and"
            " summarize key insights.\n"
            "13. For long or complex queries, break them into smaller parts and retrieve"
            " relevant document sections incrementally.\n"
            "14. Use the 'RETRIEVE' tool only when queries explicitly reference external documents"
            " such as PDFs, reports, or data analysis. Avoid using 'RETRIEVE' for general queries.\n"
            "15. Before deciding to retrieve document data, evaluate whether the query explicitly mentions"
            " document analysis, files, or content extraction. Otherwise, default to responding directly or"
            " leveraging memory tools.\n"
            "16. Use the newsdata_tool for any requests regarding current events.\n\n"
            "## Core Memories\n"
            "Core memories are fundamental to understanding the user and are"
            " always available:\n{core_memories}\n\n"
            "## Recall Memories\n"
            "Recall memories are contextually retrieved based on the current"
            " conversation:\n{recall_memories}\n\n"
            "## Instructions\n"
            "Engage with the user naturally, as a trusted colleague or friend."
            " There's no need to explicitly mention your memory capabilities."
            " Instead, seamlessly incorporate your understanding of the user"
            " into your responses. Be attentive to subtle cues and underlying"
            " emotions. Adapt your communication style to match the user's"
            " preferences and current emotional state. Use tools to persist"
            " information you want to retain in the next conversation. If you"
            " do call tools, all text preceding the tool call is an internal"
            " message. Respond AFTER calling the tool, once you have"
            " confirmation that the tool completed successfully.\n\n"
            "Current system time: {current_time}\n\n",
        ),
        ("placeholder", "{messages}"),
    ]
)


#
# -----------------------------
#  Agent Function
# -----------------------------
#
async def agent(state: schemas.State, config: RunnableConfig) -> schemas.State:
    """
    Process the current state and generate a response using the LLM.

    Args:
        state: The current state containing messages/memories.
        config: The runtime configuration for the agent.

    Returns:
        The updated state with the agent's response.
    """
    logger.debug("Entering agent function")
    print("Entering agent function", flush=True)
    configurable = utils.ensure_configurable(config)
    llm = init_chat_model(configurable["model"])
    bound = prompt | llm.bind_tools(all_tools)

    # Pull data from the state
    messages = state.get("messages", [])
    core_memories = state.get("core_memories", [])
    recall_memories = state.get("recall_memories", [])
    current_time = datetime.now(tz=timezone.utc).isoformat()

    logger.debug(f"Messages: {messages}")
    logger.debug(f"Core memories: {core_memories}")
    logger.debug(f"Recall memories: {recall_memories}")
    logger.debug(f"Current time: {current_time}")
    print(f"Messages: {messages}", flush=True)
    print(f"Core memories: {core_memories}", flush=True)
    print(f"Recall memories: {recall_memories}", flush=True)
    print(f"Current time: {current_time}", flush=True)

    # Invoke the prompt + model
    prediction = await bound.ainvoke({
        "messages": messages,
        "core_memories": "\n".join(core_memories),
        "recall_memories": "\n".join(recall_memories),
        "current_time": current_time,
    })

    logger.debug(f"Prediction: {prediction}")
    print(f"Prediction: {prediction}", flush=True)

    # Return the new state
    return {
        "messages": prediction,
        "core_memories": core_memories,
        "recall_memories": recall_memories
    }


#
# -----------------------------
#  Memory Loader
# -----------------------------
#
def load_memories(state: schemas.State, config: RunnableConfig) -> schemas.State:
    """
    Load core and recall memories for the current conversation.

    Args:
        state: The current conversation state (messages, etc.).
        config: The runtime configuration.

    Returns:
        The updated state with loaded memories.
    """
    configurable = utils.ensure_configurable(config)
    user_id = configurable["user_id"]

    # Convert conversation to string (truncated) for memory lookup
    tokenizer = tiktoken.encoding_for_model("gpt-4o")
    convo_str = get_buffer_string(state.get("messages", []))
    convo_str = tokenizer.decode(tokenizer.encode(convo_str)[:2048])

    with get_executor_for_config(config) as executor:
        futures = [
            executor.submit(fetch_core_memories, user_id),
            executor.submit(search_memory.invoke, convo_str),
        ]
        # fetch_core_memories returns (path, core_mem_list)
        _, core_memories = futures[0].result()
        recall_memories = futures[1].result()

    # Merge the newly retrieved memories into the state
    return {
        "core_memories": core_memories,
        "recall_memories": recall_memories,
    }


#
# -----------------------------
#  Router / Tools
# -----------------------------
#
def route_tools(state: schemas.State) -> Literal["tools", "ROUTER", "__end__"]:
    """
    Decide whether to invoke tools, RAG routing, or end conversation.

    Args:
        state: The current state of the conversation.

    Returns:
        "tools" if a general tool should be called,
        "ROUTER" if the route_question node should be triggered,
        or "__end__" if no tool usage is requested.
    """
    msg = state["messages"][-1]

    # Check if the query triggers any tool calls
    if getattr(msg, "tool_calls", None):
        calls = [tc["name"] for tc in msg.tool_calls]
        if "route_question" in calls:
            return "ROUTER"
        if "retrieve" in calls:
            return "RETRIEVE"
        return "tools"

    return END


#
# -----------------------------
#  Build the Graph
# -----------------------------
#
from langgraph.graph import START

builder = StateGraph(schemas.State, schemas.GraphConfig)

# Add nodes
builder.add_node("load_memories", load_memories)
builder.add_node("agent", agent)
builder.add_node("tools", ToolNode(all_tools))
builder.add_node("ROUTER", wrapped_route_question)
builder.add_node("RETRIEVE", wrapped_retrieve)
builder.add_node("GRADE_DOCUMENTS", wrapped_grade_documents)
builder.add_node("GENERATE", generate)  # 'generate' already has docstrings
builder.add_node("WEBSEARCH", wrapped_web_search)

# Edges
builder.add_edge(START, "load_memories")
builder.add_edge("load_memories", "agent")
builder.add_edge("tools", "agent")

builder.add_conditional_edges(
    "agent",
    route_tools,
    {
        "tools": "tools",
        "ROUTER": "ROUTER",
        "RETRIEVE": "RETRIEVE",
        "__end__": END,
    }
)

# RAG flow after "ROUTER"
builder.add_conditional_edges(
    "ROUTER",
    _route_question,  # returns "RETRIEVE" or "WEBSEARCH"
    {
        "RETRIEVE": "RETRIEVE",
        "WEBSEARCH": "WEBSEARCH",
        "__end__": END,
    }
)

builder.add_edge("RETRIEVE", "GRADE_DOCUMENTS")
builder.add_conditional_edges(
    "GRADE_DOCUMENTS",
    decide_to_generate,
    {
        "WEBSEARCH": "WEBSEARCH",
        "GENERATE": "GENERATE",
    },
)
builder.add_conditional_edges(
    "GENERATE",
    _grade_generation,
    {
        "not supported": "GENERATE",
        "useful": "agent",
        "not useful": "WEBSEARCH",
    },
)
builder.add_edge("WEBSEARCH", "GENERATE")
builder.add_edge("GENERATE", "agent")

# Compile the graph
memgraph = builder.compile()


#
# -----------------------------
#  Main UI-facing Function
# -----------------------------
#
async def process_chat(messages: List[Dict[str, str]], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process chat messages through the memory graph.

    Args:
        messages: A list of message dicts with role/content. E.g.:
                  [{"role": "user", "content": "Hello!"}, {...}, ...]
        config: A dictionary of runtime configuration (e.g., {"user_id": "123", "model": "gpt-4"})

    Returns:
        A dict with {"response": "..."} containing the final assistant text.
    """
    logger.debug("Entering process_chat function")
    print("Entering process_chat function", flush=True)
    logger.debug(f"Messages: {messages}")
    print("Messages: {messages}", flush=True)
    logger.debug(f"Config: {config}")
    print(f"Config: {config}", flush=True)

    # Convert raw "role"/"content" messages into typed messages
    formatted_messages = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            formatted_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            formatted_messages.append(AIMessage(content=content))
        elif role == "system":
            formatted_messages.append(SystemMessage(content=content))
        else:
            # Fallback if some custom role is provided.
            formatted_messages.append(HumanMessage(content=content))

    logger.debug(f"Formatted messages: {formatted_messages}")
    print(f"Formatted messages: {formatted_messages}", flush=True)

    # Build the initial state
    input_state = {
        "messages": formatted_messages,
    }

    # Invoke the graph
    result = await memgraph.ainvoke(input=input_state, config=config)
    logger.debug(f"Result: {result}")
    print(f"Result: {result}", flush=True)
    
    # Extract the final assistant text from result["messages"]
    final_ai_content = ""
    if "messages" in result and result["messages"]:
        messages_val = result["messages"]
        # If messages is a list, pick the last element; otherwise, use it directly.
        if isinstance(messages_val, list):
            last_msg = messages_val[-1]
        else:
            last_msg = messages_val
        # Try to get the 'content' attribute; if not available, try the 'content' key (if it is a dict).
        final_ai_content = getattr(last_msg, "content", None) or (
            last_msg.get("content") if isinstance(last_msg, dict) else ""
        )

    # Return the "response" field so the frontend can do data.response
    return {
    "messages": [
        {
            "role": "assistant",
            "content": final_ai_content
        }
    ]
}


#
# -----------------------------
#  Docstring Issue Detector
# -----------------------------
#
import inspect

def ensure_docstring(func):
    """Inject a default docstring if one is missing."""
    if not func.__doc__ or func.__doc__.strip() == "":
        func.__doc__ = "No description provided."
    return func

def detect_and_fix_docstring_issues():
    """
    Detect and fix issues with function docstrings throughout the codebase.
    """
    logger = logging.getLogger(__name__)
    
    # Only include the base functions that are actually used
    functions = [
        save_recall_memory,
        search_memory,
        fetch_core_memories,
        store_core_memory,
        agent,
        load_memories,
        process_chat
    ]

    for func in functions:
        try:
            # If it's decorated/wrapped, get the real function
            while hasattr(func, '__wrapped__'):
                func = func.__wrapped__

            docstring = inspect.getdoc(func)
            if not docstring:
                continue

            # Basic check for 'Args:' or 'Returns:'
            if 'Args:' not in docstring:
                continue

            # Try to parse out the Args: and Returns: sections
            args_section = docstring.split('Args:')[1]
            if 'Returns:' in args_section:
                params_section = args_section.split('Returns:')[0]
            else:
                params_section = args_section

            docstring_params = []
            for line in params_section.split('\n'):
                line = line.strip()
                if ':' in line:
                    param_name = line.split(':')[0].strip()
                    docstring_params.append(param_name)

            if not docstring_params:
                continue

            # Check function signature for parameter mismatch
            try:
                sig = inspect.signature(func)
            except ValueError:
                continue

            # Example fix: if docstring says 'state' but signature uses 'input_state'
            for param in docstring_params:
                if param not in sig.parameters and param == "state":
                    fixed_docstring = docstring.replace(
                        f"{param}:", "input_state:"
                    ).replace(
                        f"{param} (dict)", "input_state (dict)"
                    )
                    func.__doc__ = fixed_docstring
                    logger.info(f"Fixed docstring for function: replaced '{param}' with 'input_state'")

        except Exception as e:
            logger.warning(f"Skipping docstring check for function: {e}")
            continue

# Now re-run ensure_docstring on everything in all_tools, just in case:
all_tools = [ensure_docstring(t) for t in all_tools]

__all__ = ["memgraph", "process_chat"]
