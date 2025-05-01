from pydantic import BaseModel, Field
from typing_extensions import TypedDict, Annotated
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import InjectedState
from langchain_core.tools.base import InjectedToolCallId
from langchain_core.tools import tool
from langgraph.types import Command
from agent.llm_provider import get_llm_structured
from agent.tools.retrieve_pg_tools import vector_store

# ----------------------------- SCHEMAS -----------------------------
class Feedback(BaseModel):
    issue: str = Field(description="The issue identified by the human reviewer")
    solution: str = Field(description="The solution to the issue identified by the human reviewer")
    criteria: str = Field(description="The criteria used to evaluate the CV")

    def __str__(self):
        return (
            f"Criteria: {self.criteria}\n"
            f"Issue: {self.issue}\n"
            f"Solution: {self.solution}\n"
        )
        
class Feedbacks(BaseModel):
    feedbacks: list[Feedback] = Field(description="The feedbacks from the human reviewers")
    
    def __str__(self):
        return "\n---\n".join(str(fb) for fb in self.feedbacks)

class ReviewedCV(BaseModel):
    new_cv: str = Field(description="The new suitable CV after reviewing the candidate's current CV")

class SuggestChangeState(TypedDict):
    candidate_cv: str
    job_description: str
    review: list[Feedback]
    new_cv: str
    human_feedback: str


# ------------------------- PROMPT TEMPLATES -------------------------
review_instruction = """
You are an experienced CV Reviewer & Recruitment Specialist responsible for evaluating a candidate’s CV 
based on a Job Description (JD) and 10 key hiring criteria.

Your review must include at least 5 pieces of feedback across all criteria.

Context:
Job Description (JD):
{job_description}

Candidate’s CV:
{candidate_cv}

Review the CV based on the following criteria:
1. Job Fit
2. Work Experience
3. Technical & Soft Skills
4. Achievements & Impact
5. Education & Certifications
6. Consistency & Accuracy
7. CV Formatting & Readability
8. Projects & Contributions
9. Cultural Fit
10. Growth Potential & Initiative

Instructions:
- Be objective and structured.
- Identify weaknesses and provide specific solutions.
- Generate at least 5 key feedback points.
- Avoid generic responses.
- Highlight missing info.
- Use bullet points if needed.
"""

adjust_instruction = """
You are an AI-powered CV Editing Specialist. Improve the candidate’s CV based on reviewer feedback to match the JD and industry standards.

Context:
Job Description (JD):
{job_description}

Original Candidate’s CV:
{candidate_cv}

Carefully adjust the CV following {n_keys} key criteria:
{criteria}

Instructions:
- Apply all feedback and ensure clear improvements.
- Keep the CV concise, impactful, and professional.
- Ensure at least 5 major improvements.
- Never fabricate info—enhance existing content only.
- Return output in markdown form, remember to highlight changes by green color markdown
"""

# --------------------------- AGENT NODES ---------------------------
def suggest_cv(state: SuggestChangeState):
    print('--suggest--')
    candidate_cv = state["candidate_cv"]
    job_description = state["job_description"]

    system_message = review_instruction.format(
        candidate_cv=candidate_cv,
        job_description=job_description
    )

    structured_llm = get_llm_structured(Feedbacks)
    feedbacks = structured_llm.invoke(
        [SystemMessage(system_message), HumanMessage('Let start the review process')]
    )

    return {'review': feedbacks.feedbacks}

def adjust_cv(state: SuggestChangeState) -> ReviewedCV:
    print('--adjust--')
    candidate_cv = state["candidate_cv"]
    job_description = state["job_description"]
    feedbacks = state['review']

    criteria = "\n".join([
        f"{i+1}. {fb.criteria}: {fb.issue}\n\tSolution: {fb.solution}"
        for i, fb in enumerate(feedbacks)
    ])

    system_message = adjust_instruction.format(
        job_description=job_description,
        candidate_cv=candidate_cv,
        criteria=criteria,
        n_keys=len(feedbacks)
    )

    structured_llm = get_llm_structured(ReviewedCV)
    new_cv = structured_llm.invoke(
        [SystemMessage(system_message), HumanMessage('Let start the adjust process')]
    )

    return {'new_cv': new_cv.new_cv}

# ---------------------------- GRAPH NODES ----------------------------
builder_1 = StateGraph(SuggestChangeState)
builder_1.add_node('suggest_cv', suggest_cv)
builder_1.set_entry_point('suggest_cv')
builder_1.set_finish_point('suggest_cv')
suggest_agent = builder_1.compile()

builder_2 = StateGraph(SuggestChangeState)
builder_2.add_node('adjust_cv', adjust_cv)
builder_2.add_edge(START, 'adjust_cv')
builder_2.add_edge('adjust_cv', END)
adjust_agent = builder_2.compile()

# ---------------------------- COMBINED FLOW ----------------------------
workflow = StateGraph(SuggestChangeState)
workflow.add_node('suggest', suggest_agent)
workflow.add_node('change', adjust_agent)
workflow.add_edge("suggest", "change")
workflow.set_entry_point("suggest")
workflow.set_finish_point('change')
review_agent = workflow.compile()

# ----------------------------- TOOL WRAPPER -----------------------------
@tool
def review_cv(job_index: str, cv: Annotated[str, InjectedState("cv")], tool_call_id: Annotated[str, InjectedToolCallId]):
    """Review and improve a CV by comparing it against a specific job description.

    Args:
        job_index (str): The identifier of the job description to compare against."""
    print("--tool: review_cv--")

    if not cv:
        raise FileExistsError('CV is not uploaded yet.')

    jd = vector_store.get_by_ids([job_index])[0]
    
    if not jd:
        raise FileExistsError('JD is not available.')
        
    jd = jd.page_content

    result = review_agent.invoke({
        "job_description": jd,
        "candidate_cv": cv,
    })

    return Command(
        update={
            "messages": [ToolMessage("Succesfully review", tool_call_id=tool_call_id)],
            "new_cv": result.get("new_cv", ""),
            "cv_reviews": result.get("review", "")
        }
    )
