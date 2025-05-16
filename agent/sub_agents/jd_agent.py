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







SYNTHESIZE_SYSTEM_PROMPT = """You are an AI assistant helping users analyze job descriptions.

Your task is to extract and structure the main components of the job from the following description:
Extract using the structured format.
"""
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
    jd_analysis: Annotated[list, add]
    jd_indices: list
    summary: str

# ---------------------------- AGENT LOGIC ----------------------------
def get_jd(state):
    print('--get_jd--', state)
    jds = vector_store.get_by_ids([str(i) for i in state["jd_indices"]])
    print('1------', jds)
    jds = [jd.page_content for jd in jds]
    
    print('2------', jds)
    if jds==[]:
        print('stop')
        return Command(goto = 'jd_expert',
        graph = Command.PARENT,
                   
        update = {"messages": [AIMessage('fail to route')]})
    else:
        return {"jds": jds}

def router(state):
    """Route each JD into the extraction node"""
    print('--router--', state)
    
    if state.get("jds"):
        print('1------', )
        
        return [Send("parser", {"jd": jd}) for jd in state['jds']]
    else:
        print('2------', )
        # return 
        return Command(goto = 'jd_expert',
            graph = Command.PARENT,
                   
        update = {"messages": [AIMessage('fail to route')]})

def parser_agent(state): 
    print('--parser--')
    
    jd = state.get("jd", "")
    llm = get_llm_structured(JobCriteriaComparison)
    response = llm.invoke([
        SystemMessage(SYNTHESIZE_SYSTEM_PROMPT),
        HumanMessage(f"Conduct extraction this jd :{jd}")
    ])
    
    return Command(update = {"jd_analysis": [response]})

SUMMARIZE_SYSTEM_PROMPT = """You are a hiring analyst AI assistant. Your task is to summarize and synthesize multiple job descriptions (JDs) that share the same job title.

Each JD has been analyzed based on a set of common criteria such as responsibilities, required skills, experience, education, and soft skills.

Your goal is to:
1. Identify common patterns across the JDs.
2. Highlight differences or variations.
3. Note any unique features in any JD.
4. Optionally, categorize JDs into types.
5. Provide a final summary insight about the market for this role.

Use markdown formatting and bullet points/tables if appropriate.
"""
def summarize_agent(state):
    print('--summary--')
    jd_analysis = state['jd_analysis']
    llm = get_llm()
    response = llm.invoke([
        SystemMessage(SUMMARIZE_SYSTEM_PROMPT),
        HumanMessage(f"Here are the analyses: {jd_analysis}. /no_think")
    ])
    # print("--response--", response)
    return Command(goto = 'router',
        graph = Command.PARENT,
                   
        update = {"messages": [response]})

# ---------------------------- GRAPH ----------------------------
graph = StateGraph(AnalyzeState)
graph.add_node("get_jd", get_jd)
graph.add_node("parser", parser_agent)
graph.add_node("summarize", summarize_agent)

graph.set_entry_point("get_jd")
graph.add_conditional_edges("get_jd", router, ["parser"])

graph.add_edge("parser", "summarize")
graph.set_finish_point("summarize")

synthesize_agent = graph.compile()



# ---------------------------- PROMPT ----------------------------
# Prompt dùng để chấm điểm CV theo từng JD
SCORE_PROMPT_SYSTEM = """
You are an expert in evaluating the relevance between a candidate's CV and a specific Job Description (JD).

Your task is to score how well the candidate matches the job across different criteria, and assign **a weight for each criterion** based on the **importance of that factor in the given JD**.

Please follow the rules below:

1. For each evaluation criterion, give:
   - A score from 0 to 10 based on the candidate's CV.
   - A weight from 0.0 to 1.0 representing how important that criterion is in the JD.

2. Then, write **one brief overall comment** summarizing the candidate’s fit.

3. Do not calculate the final average score. It will be computed automatically.

### Fields to fill:

- `id`: Index or ID of the Job Description being evaluated.
- `job_title_relevance`: Does the candidate's experience match the job title?
- `years_of_experience`: Is the experience duration sufficient?
- `required_skills_match`: Does the candidate possess the required technical skills?
- `education_certification`: Is the academic background suitable?
- `project_work_history`: Are the past projects relevant?
- `softskills_language`: Does the candidate demonstrate useful soft skills?
- `*_weight`: For each of the above criteria, specify how important it is (0.0 to 1.0) for this job.
- `overall_comment`: Give one short paragraph with your summary evaluation.
- Leave `overall_fit_score` as 0; it will be computed after scoring.

### Example:
If the JD is for a senior backend engineer with a focus on Golang microservices, then:
- `required_skills_match_weight` might be 0.9 (very important),
- `education_certification_weight` might be 0.3 (less important).

Return the result strictly in the schema format.
"""

# Prompt dùng để tổng hợp các đánh giá thành 1 đoạn summary
summary_instruction = """You are an AI assistant summarizing HR evaluations.

You are given a list of structured evaluation objects comparing one CV against multiple job descriptions (JDs), each with an overall fit score and component scores.

Summarize in the following format:

1. 🏆 **Rank the JDs** by overall_fit_score (from highest to lowest), include score next to each:
   - JD #3 – 8.4
   - JD #1 – 7.8
   - JD #2 – 6.9

2. 📌 **For each JD**, add 1–2 bullet points noting key strengths or weaknesses (based on score components), such as:
   - Strong skill match, but low education fit
   - Good experience, lacks soft skills

Be concise, use clear bullet points. Do not repeat similar strengths across all JDs unless relevant for comparison.
"""

# ---------------------------- SCHEMA ----------------------------
from pydantic import BaseModel, Field
from typing import Literal
from pydantic import BaseModel, Field, model_validator

class CVJDMatchFeedback(BaseModel):
    id: int = Field(..., description="Job index")

    job_title_relevance: int = Field(..., ge=0, le=10, description="Score (0-10): How well does the candidate's experience align with the job title?")
    job_title_weight: float = Field(..., ge=0.0, le=1.0, description="Weight (0-1): Importance of job title relevance for this role.")
    
    years_of_experience: int = Field(..., ge=0, le=10, description="Score (0-10): Does the candidate have sufficient experience for the position?")
    years_of_experience_weight: float = Field(..., ge=0.0, le=1.0, description="Weight (0-1): Importance of experience duration.")

    required_skills_match: int = Field(..., ge=0, le=10, description="Score (0-10): To what extent does the candidate possess the skills listed in the JD?")
    required_skills_weight: float = Field(..., ge=0.0, le=1.0, description="Weight (0-1): Importance of matching required skills.")

    education_certification: int = Field(..., ge=0, le=10, description="Score (0-10): Does the candidate's academic background fit the job requirements?")
    education_certification_weight: float = Field(..., ge=0.0, le=1.0, description="Weight (0-1): Importance of educational background.")

    project_work_history: int = Field(..., ge=0, le=10, description="Score (0-10): Are the candidate’s past projects or roles relevant to this position?")
    project_work_history_weight: float = Field(..., ge=0.0, le=1.0, description="Weight (0-1): Importance of work/project experience relevance.")

    softskills_language: int = Field(..., ge=0, le=10, description="Score (0-10): Does the candidate show relevant communication, leadership, or other soft skills?")
    softskills_language_weight: float = Field(..., ge=0.0, le=1.0, description="Weight (0-1): Importance of soft skills and communication.")

    overall_comment: str = Field(..., description="One overall comment about the candidate’s fit for the job.")
    overall_fit_score:       float = Field(0, description="Average score (0-10) calculated from all score fields.")
    
    
    @model_validator(mode="after")
    def compute_overall_score(self):
        self.overall_fit_score = round(
            self.job_title_relevance * self.job_title_weight +
            self.years_of_experience * self.years_of_experience_weight +
            self.required_skills_match * self.required_skills_weight +
            self.education_certification * self.education_certification_weight +
            self.project_work_history * self.project_work_history_weight +
            self.softskills_language * self.softskills_language_weight
        )
        return self

def format_cvjd_feedback_list(feedback_list: list[CVJDMatchFeedback]) -> str:
    output = []
    for fb in feedback_list:
        text = f"""\
        Job Index {fb.id}:
        - Job Title Relevance:       {fb.job_title_relevance}/10
        - Years of Experience:       {fb.years_of_experience}/10
        - Required Skills Match:     {fb.required_skills_match}/10
        - Education & Certification: {fb.education_certification}/10
        - Project Work History:      {fb.project_work_history}/10
        - Soft Skills & Language:    {fb.softskills_language}/10
        - Overall Fit Score:          {fb.overall_fit_score}/10

        Comment: {fb.overall_comment}
        """
        
        output.append(text)
    return "\n".join(output)

# ---------------------------- GRAPH STATE ----------------------------
class ScoreState(MessagesState):
    cv: str
    jd_index: str
    jds: List[str]
    scored_jds: Annotated[list | list[CVJDMatchFeedback], add]
    jd_indices: List[str]
    


# ---------------------------- AGENT LOGIC ----------------------------

def get_jd(state):
    print('--get_jd--', state)
    jds = vector_store.get_by_ids([str(i) for i in state["jd_indices"]])
    jds = [jd.page_content for jd in jds]
    
    return {"jds": jds}

def router(state):
    """Route each JD into the extraction node"""
    print('--router--', state)
    
    if state.get("jds", []):
        return [Send("score", {"jd": jd, "cv": state["cv"], 'jd_index': id}) for jd, id in zip(state["jds"], state["jd_indices"])]
    else:
        return Command(goto = 'jd_expert',
        graph = Command.PARENT,
                   
        update = {"messages": [AIMessage('fail to route')]})
        
        
def score_agent(state): #: Annotated[ScoreState, InjectedState]):
    print("--score--")
    
    jd = state.get("jd", "")
    cv = state.get("cv", "")
    
    llm = get_llm_structured(CVJDMatchFeedback)
    response = llm.invoke([
        SystemMessage(SCORE_PROMPT_SYSTEM),
        HumanMessage(f"Conduct scoring job {state['jd_index']}: {jd} with cv: {cv} . /no_think")
    ])
    print(type(response), "response from score",  response)
    print("state", state)
    return {"scored_jds": [response]}

def summarize_score_agent(state):
    print("--summa--")
    jd_analysis = state.get("scored_jds", [])
    print(state)
    llm = get_llm()
    response = llm.invoke([
        SystemMessage(summary_instruction),
        HumanMessage(f"Here are the analyses of jobs to compare: {jd_analysis}. /no_think")
    ])
    print(response)
    return Command(goto = 'router',
        graph = Command.PARENT,
                   
        update = {"message": [response]})

# ---------------------------- GRAPH ----------------------------

def build_score_graph() -> StateGraph:
    score_graph = StateGraph(ScoreState)
    score_graph.add_node("get_jd", get_jd)
    score_graph.add_node("score", score_agent)
    score_graph.add_node("summarize", summarize_score_agent)

    score_graph.set_entry_point("get_jd")
    score_graph.add_conditional_edges("get_jd", router, ["score"])    
    score_graph.add_edge("score", "summarize")
    score_graph.set_finish_point("summarize")

    return score_graph.compile()

score_agent = build_score_graph()



from langgraph.prebuilt import tools_condition, ToolNode



JD_SYSTEM_PROMPT = """
You are a coordination agent designed to route user job-related requests to the correct tools. 
You cannot fabricate data or provide market analysis beyond what is in the job descriptions (JDs). 
Your job is to understand the user's intent and invoke one of the available tools to help them.

How to Act:
1. If the user asks to explore or analyze a field, a market like "marketing" or "UI/UX jobs", call `call_job_searcher` first, alway call job_searcher first.
    • After that, you may use scoring or synthesis tools as appropriate.
    
2. If the user asks how well their CV fits a role or job, and JD indices are already available, use `call_score_jds()`.

3. If a tool cannot be used due to missing information (e.g., no JD indices yet), call `call_job_searcher`
    • Never fabricate data or assume.

Never:
• Make up job descriptions or scores.
• Give CV advice or generate new documents — other agents handle those.
"""

 
@tool
def call_score_jds(jd_indices: List[int]):
    """
    Scores the user's CV against one or more job descriptions (JDs) to evaluate fit. You can also
    think of this as ranking JDs based on how well they match the CV.

    Args:
        jd_indices (List[int]): One or more job description indices to compare against the user's CV.

    """
    return

@tool
def call_synthesize_jds(jd_indices: List[int]):
    """
     Summarizes and compares multiple job descriptions to extract insights, conduct market analysis about the job market, 
    key skill demands, or overlapping patterns across roles.

    Args:
        jd_indices (List[int]): The job description indices to synthesize.

    Returns:
        A narrative synthesis that describes:
        - Similarities and differences among the JDs
        - Common required skills, tools, or qualifications
        - Trends or focus areas across the roles
        - Any anomalies or outliers

    """
    return

@tool
def call_job_searcher(task_title):
    """
    Initiates a job search for a given field or role. This is the *first* tool you must call 
    when the user mentions a domain but hasn't provided any job indices yet.

    Args:
        task_title (str): The job title or field of interest. The more detail the better. Alway include number of 3 or more. For example: 3 job data analyst in marketing domain, ...
    """
    # This tool is not returning anything: we're just using it
    # as a way for LLM to signal that it needs to hand off to planner agent
    return

from .schema import AgentState

def jd_agent_node(state: AgentState):
    print('--- jds expert ---')
    llm = get_llm().bind_tools([call_score_jds, call_synthesize_jds, call_job_searcher])
    print('state:  ', state)
    response = llm.invoke([SystemMessage(JD_SYSTEM_PROMPT)] + state["messages"])
    print(response)
    if len(response.tool_calls) > 0:
        if response.tool_calls[0]['name'] == 'call_job_searcher':
            print("---------goi job")
            return Command(
				goto = 'job_searcher_agent',
                graph=Command.PARENT,
				update={"messages": [AIMessage(response.tool_calls[0]['args']['task_title'])],'sender': 'jd_agent'},
			)
        elif response.tool_calls[0]['name'] == 'call_score_jds':

            return Send(
                # 'score_jds', {"jd_indices": response.tool_calls[0]['args'].get('jd_indices', [4394, 7276]), 'cv': state['cv']}
                'score_jds', {"jd_indices": response.tool_calls[0]['args'].get('jd_indices', [4942, 7363]), 'cv': state['cv']}
            )
        elif response.tool_calls[0]['name'] == 'call_synthesize_jds':
            return Send(
                'synthesize_jds', {"jd_indices": [4942, 7363], 'cv': state['cv']}
            )

        else:
            pass
    return response



# def score_jds(jd_index):
#     """"""
#     return

# def synthesize_jds(jd_index):
#     """"""
#     return

def router(state):
    return Command(goto= 'coordinator',graph=Command.PARENT, update={'message_from_sender': [state['messages'][-1]]})

from .schema import AgentState
builder = StateGraph(AgentState)
builder.add_node("jd_expert", jd_agent_node)
builder.add_node("router", router)

builder.add_node("tools", ToolNode([call_job_searcher,call_synthesize_jds,call_score_jds]))
builder.add_edge(START, "jd_expert")
builder.add_conditional_edges(
    "jd_expert",
    tools_condition,
    {'tools': 'tools',END: "router" }
)
from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()
builder.add_node("score_jds", score_agent)
builder.add_node("synthesize_jds", synthesize_agent)
JDExpert = builder.compile(checkpointer=memory)
