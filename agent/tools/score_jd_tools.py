from typing import List, Annotated
from langgraph.graph import StateGraph, MessagesState
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
import operator
from langgraph.constants import Send
from pydantic import BaseModel, Field, model_validator
from agent.tools.retrieve_pg_tools import vector_store
from langchain_core.tools.base import InjectedToolCallId
from langchain_core.tools import tool
from langgraph.types import Command
from langgraph.prebuilt import InjectedState

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
    id: int = Field(..., description="Job index")
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

def format_cvjd_feedback_list(feedback_list: list[CVJDMatchFeedback]) -> str:
    output = []
    for fb in feedback_list:
        text = f"""\
        Job Index {fb.id}:
        - 🧠 Job Title Relevance:       {fb.job_title_relevance}/10
        - 📆 Years of Experience:       {fb.years_of_experience}/10
        - 🛠️ Required Skills Match:     {fb.required_skills_match}/10
        - 🎓 Education & Certification: {fb.education_certification}/10
        - 📂 Project Work History:      {fb.project_work_history}/10
        - 💬 Soft Skills & Language:    {fb.softskills_language}/10
        - ⭐ Overall Fit Score:          {fb.overall_fit_score}/10

        📝 Comment: {fb.overall_comment}
        """
        
        output.append(text)
    return "\n".join(output)

class ScoreSummary(BaseModel):
    summary: str = Field(..., description="A concise summary of the JD scoring analysis.")

# ---------------------------- GRAPH STATE ----------------------------
class ScoreState(MessagesState):
    cv: str
    jd_index: str
    jds: List[str]
    scored_jds: Annotated[list | list[CVJDMatchFeedback], operator.add]
    jd_indices: List[str]
    


# ---------------------------- AGENT LOGIC ----------------------------
def do_nothing(state):
    pass

def router(state):
    print("--router--")
    return [Send("score", {"jd_index": id, "cv": state["cv"]}) for id in state.get("jd_indices", [])]

def score_agent(state): #: Annotated[ScoreState, InjectedState]):
    print("--score--")
    
    jd = vector_store.get_by_ids([state["jd_index"]])[0].page_content
    cv = state.get("cv", "")
    
    llm = get_llm_structured(CVJDMatchFeedback)
    response = llm.invoke([
        SystemMessage(score_instruction.format(cv=cv, jd=jd)),
        HumanMessage(f"Conduct scoring job {state['jd_index']}. /no_think")
    ])
    print(type(response), "response from score",  response)
    print("state", state)
    return {"scored_jds": [response]}

def summarize_score_agent(state):
    print("--summa--")
    # jd_analysis = state.get("scored_jds", [])
    
    # llm = get_llm_structured(ScoreSummary)
    # response = llm.invoke([
    #     SystemMessage(summary_instruction),
    #     HumanMessage(f"Here are the analyses of jobs: {jd_analysis}. /no_think")
    # ])
    pass

# ---------------------------- GRAPH ----------------------------

def build_score_graph() -> StateGraph:
    score_graph = StateGraph(ScoreState)
    score_graph.add_node("do_nothing", do_nothing)
    score_graph.add_node("score", score_agent)
    score_graph.add_node("summarize", summarize_score_agent)

    score_graph.set_entry_point("do_nothing")
    score_graph.add_conditional_edges("do_nothing", router, ["score"])    
    score_graph.add_edge("score", "summarize")
    score_graph.set_finish_point("summarize")

    return score_graph.compile()

score_agent = build_score_graph()




# ---------------------------- TOOL ----------------------------
from langgraph.prebuilt import InjectedState
from langchain_core.tools import tool
@tool
def score_jobs(jd_index: list[str], cv: Annotated[str, InjectedState("cv")], tool_call_id: Annotated[str, InjectedToolCallId]):
    """
    Evaluate how well a given CV matches a list of job descriptions (JDs) by scoring each JD individually.

    Args:
        jd_index (list[str]): List of indices identifying the job descriptions to compare against the CV.

    Returns:
        dict: A structured summary that includes evaluation scores and comments highlighting the candidate's fit across all selected JDs.
    """
    print("--tool6: score--")
    # jd_index = [str(i) for i in jd_index]
    if not cv:
        raise "This tool can be executed because curriculum vitae is not uploaded yet"
    response = score_agent.invoke({"jd_indices": jd_index, "cv": cv})
    # return response
    
    print("length: ", len(response['scored_jds']))
    formated_response = format_cvjd_feedback_list(response['scored_jds'])
    print("formated:", format)
    return Command(
        update={
            "messages": [ToolMessage(formated_response, tool_call_id=tool_call_id)],
            # "new_cv": response.get("new_cv", ""),
            # "cv_reviews": response.get("review", "")
        }
    )
