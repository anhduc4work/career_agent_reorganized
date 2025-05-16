from langchain_core.messages import ToolMessage, SystemMessage, AIMessage, HumanMessage
from typing import Optional, Literal, List, Union, Dict, get_args
from typing import Annotated, Optional, Union, Literal

from operator import add
from langgraph.graph import MessagesState, StateGraph, END
from pydantic import BaseModel, Field




class AgentState(MessagesState):
    sender: str
    cv: Optional[str] 
    jds: Annotated[list, add]
    sender: Optional[str]
    new_cv: Optional[str]
    chat_history_summary: str 
    last_index: int = 0
    jd: Optional[str] 
    extractor_insights: Optional[dict] 
    analyst_insights: Optional[dict] 
    suggestor_insights: Optional[dict] 
    goto: str
    content_reviewer_insights: str | dict 
    format_reviewer_insights: str | dict
    
    
