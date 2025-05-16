from pydantic import BaseModel, Field
from typing import Annotated, Optional, Union, Literal
from langgraph.graph import MessagesState, StateGraph, END
from agent.llm_provider import get_llm_structured, get_llm
from agent.tools.retrieve_pg_tools import vector_store
from langgraph.constants import Send
from langgraph.types import Command
from langchain_core.messages import ToolMessage, SystemMessage, AIMessage, HumanMessage
from operator import add
from pydantic import BaseModel, Field
from typing import Literal
from typing import Optional, Literal, List, Union, Dict, get_args
from pydantic import BaseModel, Field
from typing_extensions import TypedDict, Annotated
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage, AnyMessage
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import InjectedState
from langchain_core.tools.base import InjectedToolCallId
from langchain_core.tools import tool
from langgraph.types import Command
from agent.llm_provider import get_llm_structured, get_llm
from agent.tools.retrieve_pg_tools import vector_store
from langgraph.constants import Send
from typing import Optional, Literal, List, Union, Dict, get_args
import os 
import json

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
    message_from_sender: AnyMessage

COORDINATOR_SYSTEM_PROMPT = """
You are CareerFlow, an intelligent AI coordinator that helps users navigate their career journey. 
You specialize in routing all other tasks to the correct next step using the `next_step` field. 

# 🎯 Your Responsibilities
- Route all task-specific or complex inputs to the proper next agent using the `next_step` field

# 🧭 Request Classification

## 1. Handle Directly
Use `next_step = "__end__"` when:
- The input is a simple greeting or small talk
  e.g. “hi”, “hello”, “how are you?”, “what’s your name?”
- The user asks what you can do or your role

## 2. Route to Next Agent
Use `next_step = "<AGENT>"` when:
- The input contains a specific task or request that matches an agent’s domain:
  
| Request Type                                | Set `next_step`          |
|--------------------------------------------|---------------------------|
| Searching for jobs                         | "job_searcher_agent"      |
| Scoring CV against job descriptions        | "jd_agent"                |
| Ranking or comparing job descriptions      | "jd_agent"                |
| Synthesizing job insights or market trends | "jd_agent"                |
| Analyzing job market, domain market        | "jd_agent"                |
| Reviewing, editing, or aligning a CV format| "cv_agent"                |
| Reviewing, aligning a CV content to a JD   | "cv_agent"                |

📌 **Important**: When setting `next_step` to a specific agent, always include a `message_to_next_agent` that summarizes the user's intent or provides context for the next agent. This ensures a smooth handoff and better user experience.

# ⚙️ Execution Rules
- Never attempt to perform the task yourself (e.g., scoring, rewriting CVs)
- Only classify the request and assign the appropriate agent
- Never fabricate job listings or data
- Respond clearly, concisely, and kindly

# Notes
- Do not respond to research or technical questions outside career scope (e.g., LLM theory, system internals)
"""

class CoordinatorOutput(BaseModel):
    
    next_step: Literal['__end__', 'job_searcher_agent', 'cv_agent', 'jd_agent'] = Field(...,
        description=(
            "Next step in the graph flow. Determines which specialized agent will handle the request:\n"
            "- '__end__': End the interaction or continue small talk with CareerFlow.\n"
            "- 'job_searcher_agent': Hand off to Job Search Agent (e.g., search, filter, role exploration).\n"
            "- 'cv_agent': Hand off to CV Agent (e.g., CV review, job alignment, rewrite suggestions).\n"
            "- 'jd_agent': Hand off to JD Agent (e.g., scoring JD relevance, ranking jobs, synthesizing job trends, market analysis)."
        )
    )
    message_to_next_agent: str = Field(..., description=(
            "A summary or instruction to pass to the next agent. "
            "This should clearly explain the user's intent, request, or any necessary context. "
            "Leave empty if `next_step` is '__end__'."
        )
    )
    message_to_user: str = Field(..., description="A friendly message/ notification to send to the user.")
       
def coordinator_node(state: AgentState) -> Command:
    print('---coord --- state:------', state.keys())
    print('---------', state.get('message_from_sender', ''))
    if isinstance(state.get('message_from_sender', ''), AIMessage):
        
        print('yes',state['sender'], state['message_from_sender'])
        return Command( goto = "__end__",
                    update= {"messages": [state['message_from_sender']],'message_from_sender': ''}
            )
    # Command(update={"messages": [ToolMessage(state['messages'][-1].content, tool_call_id=tool_call_id)]})
    llm = get_llm_structured(CoordinatorOutput)
    messages = state['messages']
    cv = state.get('cv', '')
    if cv:
        print('have cv')
        add_in = f"\n **Note: Here is the full Curriculum Vitae (CV) of user (To let you know he already upload it): {cv[:200]}"
    else:
        add_in = '\n **Note: User havent upload cv yet'
    
    
    response = llm.invoke(
        [SystemMessage(COORDINATOR_SYSTEM_PROMPT+add_in)] + messages
    )
    print('res:------', response)
    
    if response.next_step == "__end__":
        # return Command(
        #     goto = response.next_step,
        #     update=,'sender': 'coordinator'},
        # )
        return Command( goto = "__end__",
                    update= {"messages": [AIMessage(response.message_to_user)]}
            )
    else:
        return Send(response.next_step, {"messages":  response.message_to_next_agent+ "(user said: "+ messages[-1].content+ ' /no_think', 
                                'sender': 'coordinator',
                                'cv': state.get('cv', ''),
                                'content_reviewer_insights': state.get('content_reviewer_insights', ''),
                                'format_reviewer_insights': state.get('format_reviewer_insights', ''),
                                })