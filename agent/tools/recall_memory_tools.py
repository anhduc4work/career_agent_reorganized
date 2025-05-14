from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import InjectedState, InjectedStore
from langchain_core.tools import tool
from typing_extensions import Annotated
from langgraph.store.base import BaseStore
from typing import Literal

from langgraph.graph import MessagesState
@tool
def recall_history_chat(query: str, config: RunnableConfig, store: Annotated[BaseStore, InjectedStore]):
    """
    Retrieve contextually relevant content, enables retrieval based on meaning, not just exact keywords, useful for:
    - Precisely recall of previous user queries or instructions
    - Searching past discussions by topic or intent

    Args:
        query (str): The semantic query to search for.

    Returns:
        A list of top matching messages, sorted by semantic similarity to the query.
    """
    print("--tool8: recall--")
    user_id = config["configurable"].get("user_id","")
    if user_id:
        namespace = ("chat_history", user_id)
        related_messages = store.search(namespace, query=query, limit=3)
        threshold = 0.5
        info = "\n".join([d.value["data"] for d in related_messages if d.score > threshold])
        print(info)
        if info:
            return info
        else:
            return "No relevant data"
    return "User have not provide/submit id so that this memory recall tool not work for them"


@tool
def recall_state(query: Literal[
    'cv', 
    'new_cv', 
    'cv_reviews', 
    'chat_history_summary', 
    'extractor_insights', 
    'analyst_insights', 
    'suggestor_insights'
], config: RunnableConfig):
    """
    Retrieve a specific value from the agent's persistent state.

    This tool is part of the LangGraph agent framework and allows querying previously 
    stored information from the shared state (AgentState). It is commonly used to fetch 
    user-related data or intermediate results within a multi-step conversational workflow.

    Args:
        query (str): The name of the field to retrieve from the state. Supported fields include:
            - "cv": The original Curriculum Vitae submitted by the user.
            - "new_cv": The revised CV after receiving feedback.
            - "cv_reviews": Structured evaluation of the CV against a job description.
            - "chat_history_summary": A summarized version of prior conversation history.
            - "extractor_insights": Parsed key requirements extracted from the job description.
            - "analyst_insights": Feedback on how well the CV matches each JD criterion.
            - "suggestor_insights": Recommendations and keywords for improving the CV.

    Returns:
        Any: The value associated with the specified query key from the state, or "Empty" if not found.
    """
    print("--tool9: get state--")
    from app_function.backend_function import graph
    return graph.get_state({"configurable": {"thread_id": config['metadata']['thread_id']}}).values.get(query, "Not available")
