from lang_memgpt._schemas import State
from lang_memgpt.RAG_Structure.chains.router import question_router, RouteQuery
from lang_memgpt.RAG_Structure.consts import WEBSEARCH, RETRIEVE
from langchain_core.tools import tool
from pydantic import ValidationError
import os

import logging

# Ensure the directory exists
log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)  # Create the directory if it doesn't exist

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Log all INFO-level and above messages
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename=os.path.join(log_dir, "route_question.log"),  # File path
    filemode="a"  # Append logs if the file exists; create if it doesn't
)


@tool
def route_question(state: State) -> str:
    """
    Default docstring: Routes a question to websearch or retriever.

    Returns:
        Literal["WEBSEARCH", "RETRIEVE", "__end__"]: The next step in the graph.
    """

    try:
        # Log the start of processing
        logging.info("Starting route_question function.")

        # Log input state
        logging.info(f"Received state: {state}")

        # Ensure 'state' contains a nested dictionary or unwrap if needed
        if "state" in state:
            state = state["state"]  # Unwrap if state is nested

        # Input validation to ensure 'question' exists in the state
        if "question" not in state:
            raise ValueError(
                "The 'state' parameter must contain a 'question' field.")

        print("---ROUTE QUESTION---")
        question = state["question"]

        source: RouteQuery = question_router.invoke({"question": question})
        print(f"Routing source: {source}")
        logging.info(f"Routing source identified: {source['datasource']}")

        if source['datasource'] == WEBSEARCH:
            print("---ROUTE QUESTION TO WEB SEARCH---")
            logging.info("Routing question to WEBSEARCH.")
            return WEBSEARCH

        elif source['datasource'] == "vectorstore":
            print("---ROUTE QUESTION TO RAG---")
            logging.info("Routing question to RETRIEVE (RAG).")
            return RETRIEVE

        else:
            # Handle unexpected sources gracefully
            logging.warning(f"Unexpected routing source: {source['datasource']}")
            return "__end__"

    except ValidationError as ve:
        # Handle Pydantic validation errors
        logging.error(f"Validation Error: {str(ve)}")
        return "Validation failed. Check input format."

    except Exception as e:
        # Catch any unexpected errors
        logging.error(f"Unexpected error occurred: {str(e)}", exc_info=True)
        return "An unexpected error occurred. Please check the logs for details."

    finally:
        # Always log function exit
        logging.info("Exiting route_question function.")
