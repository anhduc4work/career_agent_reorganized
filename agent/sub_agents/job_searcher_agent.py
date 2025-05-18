from pydantic import BaseModel, Field
from typing import Annotated, Optional, Union, Literal
from langgraph.graph import MessagesState, StateGraph, END
from agent.llm_provider import get_llm_structured, get_llm
from agent.tools.retrieve_pg_tools import vector_store
from langgraph.constants import Send
from langgraph.types import Command
from langchain_core.messages import ToolMessage, SystemMessage, AIMessage, HumanMessage, AnyMessage
from operator import add
from pydantic import BaseModel, Field
from typing import Literal
from typing import Optional, Literal, List, Union, Dict, get_args
from pydantic import BaseModel, Field
from typing_extensions import TypedDict, Annotated
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage
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
from .schema import AgentState
from agent.tools import job_search_by_cv as search_by_cv
from agent.tools import job_search_by_query as search_by_keyword
from langgraph.prebuilt import tools_condition, ToolNode

JOB_SEARCHER_SYSTEM_PROMPT = """
You are an expert job search assistant. Your job is to help users find relevant job postings either by using specific keywords or by analyzing their CV.

Details

Your primary responsibilities are:
	•	Always responding with job index
	•	You must not attempt to generate or fabricate job listings yourself
	•	You should keep your response brief, friendly, and informative after the tool runs

Execution Rules
	•	Only call one of the two tools depending on the request
	•	After the tool executes, provide a clear and friendly message with the results

Notes
	•	Keep responses concise and focused
	•	Your only goal is to route the request to search_by_keyword() or search_by_cv() appropriately
	•	You should not handle any career advice, market analysis, or CV review — those belong to other agents
	•	Never doubting 
MUST MUST return including job index/ jd id index of specific job (for example 1338 and 5646 in 1. Job A (1338), 2. Job B (5646), ...) and following by postion/hyperlink,... information
"""


def job_agent_node(state: AgentState):
    print("--job searcher--")
    print("--state: ", state)
    # print("--message form sender", message_from_sender)
    llm = get_llm().bind_tools([search_by_cv, search_by_keyword], )
    messages = state["messages"]
    cv = state.get('cv', '')
    if cv:
        add_in = '\n Note: You already have user cv'
    else:
        add_in = '\n Note: User havent upload cv yet'
    
    if isinstance(messages[-1], ToolMessage):
        messages.append(HumanMessage('/no_think'))
        
    if isinstance(state.get('message_from_sender', ''), HumanMessage):
        response = llm.invoke([SystemMessage(JOB_SEARCHER_SYSTEM_PROMPT + add_in)] + messages + [state['message_from_sender']])
        
        return {"messages": [state['message_from_sender'], response], 
                'message_from_sender': '',
                'sender': state['sender'], 
                'jds': state['jds']}
    else:
        response = llm.invoke([SystemMessage(JOB_SEARCHER_SYSTEM_PROMPT + add_in)] + messages)
        
        return {"messages": [response], 'sender': state['sender'], 'jds': state['jds']}

def router(state):
    print("--router--")
    print("----",state)
    sender = state['sender']
    if sender == 'coordinator':
        return Command(
            goto = sender,
            graph=Command.PARENT,
            update={'sender': 'job_searcher',"message_from_sender": state["messages"][-1], 'jds': state['jds']},)
    else:
        return Command(
            goto = sender,
            graph=Command.PARENT,
            update={'sender': 'job_searcher',"message_from_sender": HumanMessage(state["messages"][-1].content), 'jds': state['jds']},)

builder = StateGraph(AgentState)
builder.add_node("job_searcher", job_agent_node)
builder.add_node("router", router)

builder.add_node("tools", ToolNode([search_by_cv, search_by_keyword]))
builder.add_edge(START, "job_searcher")
builder.add_conditional_edges(
    "job_searcher",
    tools_condition,
    {'tools': 'tools',END: "router" }
)
builder.add_edge("tools", "job_searcher")
job_searcher = builder.compile()