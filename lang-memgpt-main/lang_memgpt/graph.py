"""
Lang-MemGPT: A Long-Term Memory Agent.

This module implements an agent with long-term memory capabilities using LangGraph.
It supports memory storage/retrieval and a RAG pipeline (e.g. RETRIEVE on document queries)
and exposes a simple UI-facing `process_chat` function.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional, Tuple, List, Dict, Any
import sys

import langsmith
import tiktoken
from langchain.chat_models import init_chat_model
from lang_memgpt.RAG_Structure.route_question import route_question
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
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode
from typing_extensions import Literal

from lang_memgpt import _constants as constants
from lang_memgpt import _schemas as schemas
from lang_memgpt import _settings as settings
from lang_memgpt import _utils as utils

# RAG pipeline imports
from lang_memgpt.RAG_Structure.decision_logic import decide_to_generate
from lang_memgpt.RAG_Structure.nodes.generate import generate
from lang_memgpt.RAG_Structure.nodes.ingestion import ingest_data
from lang_memgpt.RAG_Structure.nodes.retrieve import retrieve
from lang_memgpt.RAG_Structure.nodes.grade_documents import grade_documents
from lang_memgpt.RAG_Structure.nodes.web_search import web_search
from lang_memgpt.RAG_Structure.chains.router import question_router
from lang_memgpt.RAG_Structure.chains import answer_grader, generation, hallucination_grader, retrieval_grader, router
from lang_memgpt.RAG_Structure.grade_generation import grade_generation_grounded_in_documents_and_question as grade_generation

# Tools from lang_memgpt/tools
from lang_memgpt.tools import (
    get_metar_data,
    get_taf_data,
    calculate,
    unit_converter,
    date_time_tool,
    fetch_latest_news
)

# Configure logging to stdout
logging.basicConfig(
    level=logging.DEBUG,
    stream=sys.stdout,
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger("memory")

# A small non-zero vector workaround
_EMPTY_VEC = [0.00001] * 1536

# Initialize the search tool
search_tool = TavilySearchResults(max_results=1)
tools = [search_tool]

@tool
async def save_recall_memory(memory: str) -> str:
    """
    Save a memory to the database for later semantic retrieval.
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
    documents = [{
        "id": path,
        "values": vector,
        "metadata": {
            constants.PAYLOAD_KEY: memory,
            constants.PATH_KEY: path,
            constants.TIMESTAMP_KEY: current_time,
            constants.TYPE_KEY: "recall",
            "user_id": configurable["user_id"],
        },
    }]
    utils.get_index().upsert(
        vectors=documents,
        namespace=settings.SETTINGS.pinecone_namespace,
    )
    return memory

@tool
def search_memory(query: str, top_k: int = 5) -> List[str]:
    """
    Search for memories in the database based on semantic similarity.
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
    """
    path = constants.PATCH_PATH.format(user_id=user_id)
    response = utils.get_index().fetch(
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
    documents = [{
        "id": path,
        "values": _EMPTY_VEC,
        "metadata": {
            constants.PAYLOAD_KEY: json.dumps({"memories": memories}),
            constants.PATH_KEY: path,
            constants.TIMESTAMP_KEY: datetime.now(tz=timezone.utc),
            constants.TYPE_KEY: "core",
            "user_id": configurable["user_id"],
        },
    }]
    utils.get_index().upsert(
        vectors=documents,
        namespace="core_memories"
    )
    return "Memory stored."

# Combine all tools into one list (ensuring every tool has a docstring)
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
    ingest_data,
    route_question,
    retrieve,
    web_search,
    grade_documents,
    grade_generation,
]

def ensure_docstring(func):
    if not func.__doc__:
        print(f"[TOOL DEBUG] Adding docstring to {func.__name__}", flush=True)
        func.__doc__ = "No description provided."
    if "(dict)" in func.__doc__:
        print(f"Tool with (dict) in docstring: {func.__name__}", flush=True)
    return func

all_tools = [ensure_docstring(t) for t in all_tools]


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
            "11. Alfred can also process (i.e., ingest documents), index, and query PDF, CSV, Excel, and Word documents "
            "stored in a dedicated document database for research and analysis tasks.\n"
            "12. Use document processing tools to analyze uploaded PDFs or Excel files and"
            " summarize key insights.\n"
            "13. For long or complex queries, break them into smaller parts and retrieve"
            " relevant document sections incrementally.\n"
            "14. Use the retrieve tool when queries explicitly reference external documents"
            " such as PDFs, reports, or data analysis. Avoid using retrieve for general queries.\n"
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
            "When handling document queries:"
            "- Use 'retrieve' with both filename and query parameters"
            "- Example: retrieve(filename=\"doc.pdf\", query=\"what is this about?\")"
            "- Always include the document name in retrieval requests"
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
            " confirmation that the tool completed successfully. Always provide responses in markdown format, using headers, bullets, and other formatting as approrpriate.\n\n"
            # "Always provide the URL, if you use the web_search tool. The URL in the format `[{url}]({url})`.\n\n"
            "Current system time: {current_time}\n\n",
            
        ),
        ("placeholder", "{messages}"),
    ]
)

def prepare_tool_args(tool_name: str, raw_args: Dict[str, Any], last_human_message: str = None) -> Dict[str, Any]:
    """Prepare and validate tool arguments."""
    try:
        print(f"[TOOL DEBUG] prepare_tool_args called for: {tool_name}", flush=True)
        print(f"[TOOL DEBUG] Raw args: {raw_args}", flush=True)
        print(f"[TOOL DEBUG] Last human message: {last_human_message}", flush=True)

        if tool_name == "retrieve":
            try:
                print("[TOOL DEBUG] Creating retrieve state", flush=True)
                state = {"state": {"question": last_human_message or "Please provide information about the document"}}
                print(f"[TOOL DEBUG] Created retrieve state: {state}", flush=True)
                return state
            except Exception as e:
                print(f"[TOOL ERROR] Error creating retrieve state: {str(e)}", flush=True)
                raise
        return raw_args
    except Exception as e:
        print(f"[TOOL ERROR] Error in prepare_tool_args: {str(e)}", flush=True)
        raise

async def agent(state: schemas.State, config: RunnableConfig) -> schemas.State:
    """
    Process the current state and generate a response using the LLM.

    Args:
        state: The current state containing messages and memories.
        config: The runtime configuration for the agent.

    Returns:
        The updated state with the agent's response.
    """
    logger.debug("Entering agent function")
    print(f"[GRAPH] Entering agent function with state {state}", flush=True)
    configurable = utils.ensure_configurable(config)
    llm = init_chat_model(configurable["model"])
    
    # Ensure all tools have descriptions and check for "(dict)" in the docstring
    for tool in all_tools:
        if not tool.__doc__:
            tool.__doc__ = "No description provided."
        if "(dict)" in tool.__doc__:
            print(f"Tool with (dict) in docstring: {tool.__name__}", flush=True)
            
    bound = prompt | llm.bind_tools(all_tools)
    # print(f"[GRAPH] Bound: {bound}", flush=True)

    messages = state.get("messages", [])
    core_memories = state.get("core_memories", [])
    recall_memories = state.get("recall_memories", [])
    current_time = datetime.now(tz=timezone.utc).isoformat()

    prediction = await bound.ainvoke({
        "messages": messages,
        "core_memories": "\n".join(core_memories),
        "recall_memories": "\n".join(recall_memories),
        "current_time": current_time,
    })

    print(f"[GRAPH] Prediction: {prediction}", flush=True)

    # Log any tool calls if they are provided in the prediction.
    tool_calls = []
    if isinstance(prediction, AIMessage):
        tool_calls = prediction.additional_kwargs.get("tool_calls", [])
    elif isinstance(prediction, dict):
        tool_calls = prediction.get("additional_kwargs", {}).get("tool_calls", [])
    else:
        print("[GRAPH] Unknown prediction structure; unable to extract tool calls.", flush=True)

    for tc in tool_calls:
        print(f"[GRAPH] [TOOL CALL] Name: {tc['function']['name']} Arguments: {tc['function']['arguments']}", flush=True)

    return {
        "messages": prediction,
        "core_memories": core_memories,
        "recall_memories": recall_memories
    }

def load_memories(state: schemas.State, config: RunnableConfig) -> schemas.State:
    """
    Load core and recall memories for the current conversation.

    Args:
        state: The current state containing messages.
        config: The runtime configuration.

    Returns:
        An updated state with loaded core and recall memories.
    """
    configurable = utils.ensure_configurable(config)
    user_id = configurable["user_id"]

    tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")
    convo_str = get_buffer_string(state.get("messages", []))
    convo_str = tokenizer.decode(tokenizer.encode(convo_str)[:2048])

    with get_executor_for_config(config) as executor:
        futures = [
            executor.submit(fetch_core_memories, user_id),
            executor.submit(search_memory.invoke, convo_str),
        ]
        _, core_memories = futures[0].result()
        recall_memories = futures[1].result()

    return {
        "core_memories": core_memories,
        "recall_memories": recall_memories,
    }

def route_tools(state: schemas.State) -> Literal["tools", "__end__"]:
    """Route queries to general tools, retrieve, or end the conversation.

    Args:
        state (schemas.State): The current state of the conversation.

    Returns:
        Literal["tools", "__end__"]: The next step in the graph.
    """
    msg = state["messages"][-1]

    # Check if the query triggers any tool calls
    if msg.tool_calls:
        # Handle the ingest_data tool specifically
        # if any(tool["name"] == "retrieve" for tool in msg.tool_calls):
        #     return "RETRIEVE"
        # All other tools
        return "tools"

    # Default to ending the conversation
    return END

# Build the graph.
builder = StateGraph(schemas.State, schemas.GraphConfig)

builder.add_node("load_memories", load_memories)
builder.add_node("agent", agent)
builder.add_node("tools", ToolNode(all_tools))
builder.add_node("RETRIEVE", retrieve)

# Define the main flow: start -> load memories -> agent.
builder.add_edge(START, "load_memories")
builder.add_edge("load_memories", "agent")
# After tool or RETRIEVE execution, flow returns to the agent.
builder.add_edge("tools", "agent")
builder.add_edge("RETRIEVE", "agent")

# Conditionally route from the agent node.
builder.add_conditional_edges(
    "agent",
    route_tools,
    {
        "tools": "tools",
        "__end__": END,
    }
)

memgraph = builder.compile()

async def process_chat(messages: List[Dict[str, str]], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process chat messages through the memory graph.

    This function converts raw messages into typed messages, then invokes the graph.
    If the graph returns a tool call (finish_reason == 'tool_calls'), it executes the tool,
    appends the tool result to the conversation state, and re-invokes the graph until a final
    assistant message is produced.

    Args:
        messages: A list of message dictionaries with keys "role" and "content".
        config: A dictionary of runtime configuration.
            (Optionally, include "already_ingested": True to force retrieval instead of ingestion
             for files that are already ingested.)

    Returns:
        A dict with {"messages": [{"role": "assistant", "content": final_text}]}.
    """
    # Convert raw messages into LangChain messages.
    try:
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
                formatted_messages.append(HumanMessage(content=content))
        print(f"Formatted messages: {formatted_messages}", flush=True)

        # Build initial state.
        state = {"messages": formatted_messages}
        _config = config  # pass through config as received

        # Loop until the agent produces a final answer (i.e. no pending tool call).
        while True:
            try:
                print("[PROCESS DEBUG] Starting new iteration", flush=True)
                result = await memgraph.ainvoke(input=state, config=_config)
                
                try:
                    tool_calls = []
                    print(f"[PROCESS DEBUG] Result type: {result}", flush=True)
                    last_msg = result.get("messages", [])[-1] if isinstance(result.get("messages"), list) else result.get("messages")
                    print(f"[PROCESS DEBUG] Last message type: {type(last_msg)}", flush=True)
                    print(f"[PROCESS DEBUG] Last message: {last_msg}", flush=True)
                    
                    if hasattr(last_msg, "additional_kwargs"):
                        print("[PROCESS DEBUG] Extracting tool_calls from additional_kwargs", flush=True)
                        tool_calls = last_msg.additional_kwargs.get("tool_calls", [])
                    elif isinstance(last_msg, dict):
                        print("[PROCESS DEBUG] Extracting tool_calls from dict", flush=True)
                        tool_calls = last_msg.get("additional_kwargs", {}).get("tool_calls", [])
                    
                    print(f"[PROCESS DEBUG] Found tool_calls: {tool_calls}", flush=True)
                except Exception as e:
                    print(f"[PROCESS ERROR] Error extracting tool calls: {str(e)}", flush=True)
                    raise

                if tool_calls:
                    tc = tool_calls[0]
                    tool_name = tc["function"]["name"]
                    
                    try:
                        print(f"[PROCESS DEBUG] Processing tool: {tool_name}", flush=True)
                        tool_args = json.loads(tc["function"]["arguments"])
                        print(f"[PROCESS DEBUG] Parsed tool arguments: {tool_args}", flush=True)
                    except Exception as e:
                        print(f"[PROCESS ERROR] Error parsing tool arguments: {str(e)}", flush=True)
                        tool_args = {}

                    try:
                        print("[PROCESS DEBUG] Looking for last human message", flush=True)
                        last_human = next((
                            (m.content if hasattr(m, "content") else m.get("content"))
                            for m in reversed(state["messages"])
                            if (hasattr(m, "role") and m.role == "user") or 
                               (isinstance(m, dict) and m.get("role") == "user")
                        ), "No message found")
                        print(f"[PROCESS DEBUG] Found last human message: {last_human}", flush=True)
                    except Exception as e:
                        print(f"[PROCESS ERROR] Error finding last human message: {str(e)}", flush=True)
                        last_human = ""

                    if tool_name == "retrieve":
                        try:
                            print("[PROCESS DEBUG] Preparing retrieve tool arguments", flush=True)
                            tool_args = prepare_tool_args(tool_name, tool_args, last_human)
                            print(f"[PROCESS DEBUG] Prepared retrieve arguments: {tool_args}", flush=True)
                        except Exception as e:
                            print(f"[PROCESS ERROR] Error preparing retrieve arguments: {str(e)}", flush=True)
                            raise

                    try:
                        print(f"[PROCESS DEBUG] Looking for tool function: {tool_name}", flush=True)
                        tool_func = next((t for t in all_tools if t.__name__ == tool_name), None)
                        if tool_func is None:
                            raise ValueError(f"Tool {tool_name} not found")
                        
                        print(f"[PROCESS DEBUG] Executing tool with args: {tool_args}", flush=True)
                        if asyncio.iscoroutinefunction(tool_func):
                            tool_result = await tool_func(**tool_args)
                        else:
                            tool_result = tool_func(**tool_args)
                        print(f"[PROCESS DEBUG] Tool execution successful: {str(tool_result)[:100]}...", flush=True)
                    except Exception as e:
                        print(f"[PROCESS ERROR] Error executing tool: {str(e)}", flush=True)
                        raise

                    # Append tool result as a system message to the conversation.
                    state["messages"].append(SystemMessage(content=f"[Tool:{tool_name}] {tool_result}"))
                    # Continue the loop so that the graph re-processes the updated state.
                    continue
                else:
                    # No pending tool call. Update state messages and break.
                    state["messages"] = result.get("messages")
                    break

            except Exception as e:
                print(f"[PROCESS ERROR] Critical error in process_chat: {str(e)}", flush=True)
                return {"messages": [{"role": "assistant", "content": f"I encountered an error: {str(e)}"}]}

        # Extract final assistant text.
        final_ai_content = ""
        final_messages = state.get("messages", [])
        if final_messages:
            if isinstance(final_messages, list):
                last_msg = final_messages[-1]
            else:
                last_msg = final_messages
            final_ai_content = getattr(last_msg, "content", None) or (last_msg.get("content") if isinstance(last_msg, dict) else "")
        
        return {"messages": [{"role": "assistant", "content": final_ai_content}]}
    except Exception as e:
        print(f"[PROCESS_CHAT] Error: {e}", flush=True)
        return {"messages": [{"role": "assistant", "content": f"An error occurred: {e}"}]}

__all__ = ["memgraph", "process_chat"]
