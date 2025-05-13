from typing import List, Annotated
from langgraph.graph import StateGraph, MessagesState
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langgraph.constants import Send
from langchain_core.tools import tool
from pydantic import BaseModel, Field
import operator
from agent.tools.retrieve_pg_tools import vector_store
from langchain_core.tools.base import InjectedToolCallId
from langgraph.types import Command
 

from agent.llm_provider import get_llm_structured, get_llm

# ---------------------------- PROMPTS ----------------------------
analyze_instruction = """You are an AI assistant helping users analyze job descriptions.

Your task is to extract and structure the main components of the job from the following description:

{jd}

Extract using the structured format.
"""

summarize_instruction = """You are a hiring analyst AI assistant. Your task is to summarize and synthesize multiple job descriptions (JDs) that share the same job title.

Each JD has been analyzed based on a set of common criteria such as responsibilities, required skills, experience, education, and soft skills.

Your goal is to:
1. Identify common patterns across the JDs.
2. Highlight differences or variations.
3. Note any unique features in any JD.
4. Optionally, categorize JDs into types.
5. Provide a final summary insight about the market for this role.

Use markdown formatting and bullet points/tables if appropriate.
"""

# ---------------------------- SCHEMA ----------------------------
class JobCriteriaComparison(BaseModel):
    job_responsibilities: str = Field(..., description="Key responsibilities listed in the job")
    technical_skills_tools: str = Field(..., description="Required technical skills or tools")
    years_of_experience: str = Field(..., description="Years of experience required")
    education_certifications: str = Field(..., description="Required education or certifications")
    soft_skills: str = Field(..., description="Required soft skills or personality traits")
    industry_sector: str = Field(..., description="Industry or sector the job belongs to")
    location_mode: str = Field(..., description="Location and work mode: Remote / Hybrid / On-site")
    salary_range: str | None = Field(None, description="Salary range if mentioned")
    career_growth: str | None = Field(None, description="Mention of career growth or advancement opportunities")
    unique_aspects: str | None = Field(None, description="Any unique benefits or characteristics of the job")

class AnalyzeState(MessagesState):
    jd: str
    jds: List[str]
    jd_analysis: Annotated[list, operator.add]
    jd_indices: list
    summary: str

# ---------------------------- AGENT LOGIC ----------------------------
def get_jd(state):
    jds = vector_store.get_by_ids([str(i) for i in state["jd_indices"]])
    jds = [jd.page_content for jd in jds]
    return {"jds": jds}


def router(state):
    """Route each JD into the extraction node"""
    return [Send("extract", {"jd": jd}) for jd in state.get("jds", [])]

def extract_agent(state): 
    jd = state.get("jd", "")
    llm = get_llm_structured(JobCriteriaComparison)
    response = llm.invoke([
        SystemMessage(analyze_instruction.format(jd=jd)),
        HumanMessage("Conduct extraction")
    ])
    
    return Command(update = {"jd_analysis": [response]})

def summarize_agent(state):
    jd_analysis = state['jd_analysis']
    llm = get_llm()
    response = llm.invoke([
        SystemMessage(summarize_instruction),
        HumanMessage(f"Here are the analyses: {jd_analysis}. /no_think")
    ])
    # print("--response--", response)
    return Command(update = {"summary": response.content})

# ---------------------------- GRAPH ----------------------------
analyze_graph = StateGraph(AnalyzeState)
analyze_graph.add_node("get_jd", get_jd)
analyze_graph.add_node("extract", extract_agent)
analyze_graph.add_node("summarize", summarize_agent)

analyze_graph.set_entry_point("get_jd")
analyze_graph.add_conditional_edges("get_jd", router, ["extract"])

analyze_graph.add_edge("extract", "summarize")
analyze_graph.set_finish_point("summarize")

analyze_agent = analyze_graph.compile()

# ---------------------------- TOOL ----------------------------
@tool
def job_market_analysis(jd_indices: list[str], tool_call_id: Annotated[str, InjectedToolCallId]):
    """
    Analyze and summarize a set of job descriptions (JDs) for a given job title.

    This tool extracts patterns, skill requirements, and trends from multiple job descriptions,
    helping users understand the current market landscape for a role.

    Note:
        For effective market analysis, provide at least 5 JD indices. If not, use tool to find more.

    Args:
        jd_indices (list[str]): List of job description IDs (in same domain/sector) to analyze. Minimum 5 recommended.

    """
        # You should call `job_search_by_query` first to retrieve a list of related JD IDs.
    print("--tool: compare_jobs--")
    response = analyze_agent.invoke({"jd_indices": jd_indices})
    
    
    return Command(
        update={
            "messages": [ToolMessage(response['summary'], tool_call_id=tool_call_id)],
            # "job_analysis": response.jd_analysis
            # "new_cv": response.get("new_cv", ""),
            # "cv_reviews": response.get("review", "")
        }
    )
