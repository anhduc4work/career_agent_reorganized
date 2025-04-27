from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import InjectedState, InjectedStore
from langchain_core.tools import tool
from typing_extensions import Annotated
from langgraph.store.base import BaseStore


@tool
def recall_memory(query: str, config: RunnableConfig, store: Annotated[BaseStore, InjectedStore]):
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