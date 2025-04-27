from typing import List, Annotated
from langgraph.graph import StateGraph, MessagesState
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.constants import Send
from langchain_core.tools import tool
from pydantic import BaseModel, Field
import operator

from agent.llm_provider import get_llm_structured

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

# ---------------------------- AGENT LOGIC ----------------------------
def do_nothing(state):
    """No-op for initializing state"""
    pass

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
    return {"jd_analysis": [response]}

def summarize_agent(state):
    jd_analysis = state.get("jd_analysis", [])
    llm = get_llm_structured(str)
    response = llm.invoke([
        SystemMessage(summarize_instruction),
        HumanMessage(f"Here are the analyses: {jd_analysis}")
    ])
    return {"messages": [response]}

# ---------------------------- GRAPH ----------------------------
analyze_graph = StateGraph(AnalyzeState)
analyze_graph.add_node("_init", do_nothing)
analyze_graph.add_node("extract", extract_agent)
analyze_graph.add_node("summarize", summarize_agent)

analyze_graph.set_entry_point("_init")
analyze_graph.add_conditional_edges("_init", router, ["extract"])
analyze_graph.add_edge("extract", "summarize")
analyze_graph.set_finish_point("summarize")

analyze_agent = analyze_graph.compile()

# ---------------------------- TOOL ----------------------------
@tool
def compare_jobs_tool(jds: list[str]):
    """
    Analyze and summarize a list of job descriptions (JDs) for the same job title.

    Returns:
        A structured summary highlighting patterns, differences, and trends from the provided JDs.
    """
    print("--tool: compare_jobs--")
    return analyze_agent.invoke({"jds": jds})
