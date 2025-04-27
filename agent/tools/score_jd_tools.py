from typing import List, Annotated
from langgraph.graph import StateGraph, MessagesState
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
import operator
from langgraph.constants import Send
from pydantic import BaseModel, Field, model_validator

# Import LLM model and prompts from external module
from agent.llm_provider import get_llm_structured

# ---------------------------- PROMPT ----------------------------
# Prompt dùng để chấm điểm CV theo từng JD
score_instruction = """You are an AI assistant helping HR evaluate job candidates.

Your task is to evaluate a candidate’s fit based on the provided **Job Description (JD)** and **Curriculum Vitae (CV)**.

Please score the candidate from 0 to 10 across the following criteria:
1. Job title relevance
2. Years of experience
3. Required skills match
4. Education & certifications
5. Project & work history relevance
6. Soft skills & language

Then, write an **overall comment** explaining the fit in 1–3 sentences.

### Job Description:
{jd}

### Candidate CV:
{cv}

Please return the scores and comment in the expected structured format.
"""

# Prompt dùng để tổng hợp các đánh giá thành 1 đoạn summary
summary_instruction = """You are an AI assistant summarizing HR evaluations.

You are given a list of evaluation objects that scored one CV against multiple job descriptions.

Summarize the following:
- Patterns or trends across evaluations
- Overall candidate fit across all jobs
- Any notable strengths or weaknesses
- If applicable, recommend the best-fitting job(s)

Only refer to information available in the analysis. Be concise and insightful.
"""

# ---------------------------- SCHEMA ----------------------------
class CVJDMatchFeedback(BaseModel):
    job_title_relevance:     int = Field(..., ge=0, le=10, description="Score (0-10): How well does the candidate's experience align with the job title?")
    years_of_experience:     int = Field(..., ge=0, le=10, description="Score (0-10): Does the candidate have sufficient experience for the position?")
    required_skills_match:   int = Field(..., ge=0, le=10, description="Score (0-10): To what extent does the candidate possess the skills listed in the JD?")
    education_certification: int = Field(..., ge=0, le=10, description="Score (0-10): Does the candidate's academic background fit the job requirements?")
    project_work_history:    int = Field(..., ge=0, le=10, description="Score (0-10): Are the candidate’s past projects or roles relevant to this position?")
    softskills_language:     int = Field(..., ge=0, le=10, description="Score (0-10): Does the candidate show relevant communication, leadership, or other soft skills?")
    overall_comment: str = Field(..., description="One overall comment about the candidate’s fit for the job.")
    overall_fit_score:       float = Field(0, description="Average score (0-10) calculated from all score fields.")

    @model_validator(mode="after")
    def compute_overall_score(self):
        self.overall_fit_score = round((
            self.job_title_relevance +
            self.years_of_experience +
            self.required_skills_match +
            self.education_certification +
            self.project_work_history +
            self.softskills_language
        ) / 6, 2)
        return self

class ScoreSummary(BaseModel):
    summary: str = Field(..., description="A concise summary of the JD scoring analysis.")

# ---------------------------- GRAPH STATE ----------------------------
class ScoreState(MessagesState):
    cv: str
    jd: str
    jds: List[str]
    jd_analysis: Annotated[list, operator.add]


# ---------------------------- AGENT LOGIC ----------------------------
def do_nothing(state):
    """Do nothing but setup for router"""
    pass

def router(state):
    print("--router--")
    print(type(state.get("jds", [])),state.get("jds", []))
    return [Send("score", {"jd": jd}) for jd in state.get("jds", [])]

def score_agent(state):
    print("--score--")
    jd = state.get("jd", "")
    cv = state.get("cv", "")
    llm = get_llm_structured(CVJDMatchFeedback)
    response = llm.invoke([
        SystemMessage(score_instruction.format(cv=cv, jd=jd)),
        HumanMessage("Conduct scoring")
    ])
    return {"jd_analysis": [response]}

def summarize_score_agent(state):
    print("--summa--")
    jd_analysis = state.get("jd_analysis", [])
    llm = get_llm_structured(ScoreSummary)
    response = llm.invoke([
        SystemMessage(summary_instruction),
        HumanMessage(f"Here are the analyses of jobs: {jd_analysis}")
    ])
    return response

# ---------------------------- GRAPH ----------------------------

def build_score_graph() -> StateGraph:
    score_graph = StateGraph(ScoreState)
    score_graph.add_node("init", do_nothing)
    score_graph.add_node("score", score_agent)
    score_graph.add_node("summarize", summarize_score_agent)

    score_graph.set_entry_point("init")
    score_graph.add_conditional_edges("init", router, ["score"])
    score_graph.add_edge("score", "summarize")
    score_graph.set_finish_point("summarize")

    return score_graph.compile()

score_agent = build_score_graph()

# ---------------------------- TOOL ----------------------------
from langchain_core.tools import tool
@tool
def score_jobs(jds: list[str], cv: str):
    """
    Compare a list of job descriptions (JDs) against a single CV and evaluate the candidate's fit for each job.

    Returns:
        A structured summary containing score evaluations and comments for each JD compared to the CV.
    """
    print("--tool6: score--")
    response = score_agent.invoke({"jds": jds, "cv": cv})
    return response
