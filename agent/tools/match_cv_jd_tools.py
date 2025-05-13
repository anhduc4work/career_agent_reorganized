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
    new_cv: str = Field(description="The new suitable curriculum vitae after reviewing the candidate's current CV, not the old version")

class SuggestChangeState(TypedDict):
    candidate_cv: str
    job_description: str
    review: list[Feedback]
    new_cv: str
    human_feedback: str


# ------------------------- PROMPT TEMPLATES -------------------------
review_instruction = """
You are an experienced HR reviewer. Your job is to analyze a candidate’s CV in comparison with a specific job description (JD) and provide a professional, structured, and actionable assessment.
You do not rewrite or edit the CV. Your task is to point out how well the CV matches the JD, what is missing and what should be improved.
Please follow this structure in your review:

	1.	Understand the JD:
	•	Identify key required skills, qualifications, and responsibilities.
	•	Highlight critical keywords (e.g., tools, technologies, methods).
	•	Understand the expected role, domain, and tone of the JD.
	2.	Compare the CV against the JD:
	•	Analyze whether the candidate’s CV reflects the required skills and experience.
	•	Point out missing keywords or competencies.
	•	Identify parts that do not align with the job’s focus.
	3.	Evaluate major sections of the CV:
	•	Career Objective (if present): Is it relevant to the job? Is it specific enough?
	•	Skills Section: Are the listed skills relevant and do they match the JD?
	•	Work Experience: Are responsibilities and achievements aligned with the JD?
	•	Education & Certifications: Are they sufficient for the role?
	•	Language and tone: Is it professional and aligned with the target role?
	4.	Give structured feedback:
	•	Weaknesses: List up to 5 areas where the CV does not meet the JD.
	•	Suggestions: Provide professional advice on how to improve alignment (without rewriting).

Your feedback must be:
	•	Honest and objective
	•	Actionable and specific
	•	Structured in paragraphs or bullet points

Do not assume missing details — only evaluate based on the given content.

Context:
Job Description (JD):
{job_description}

Candidate’s CV:
{candidate_cv}
"""

adjust_instruction = """
You are an AI-powered CV Editing Specialist. Improve the candidate’s CV based on reviewer feedback to match the JD and industry standards.

Context:

Original Candidate’s CV:
{candidate_cv}

Carefully adjust the CV following {n_keys} key criteria:
{criteria}

Instructions:
- Apply all feedback and ensure clear improvements.
- Keep the CV concise, impactful, and professional.
- Ensure at least 5 major improvements.
- Never fabricate info—enhance existing content only.
- Return output in markdown form, remember to highlight changes by green color markdown.
- Must return nothing but new, reviewed, changed Curriculum Vitae.
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
        [SystemMessage(system_message), HumanMessage('Let start the review process /no_think')]
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

    structured_llm = get_llm()
    response = structured_llm.invoke(
        [SystemMessage(system_message), HumanMessage('Let start the adjust process /no_think')]
    )
    print("------   ",response)
    
    extractor = get_llm_structured(ReviewedCV)
    extracted = extractor.invoke([
                SystemMessage("You are a curriculum vitae extractor that helps parser raw cv with markdown."),
                HumanMessage(f"Here is a messages contain cv content \n {response.content} /no_think")
            ]
        )

    return {'new_cv': extracted.new_cv}

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
def match_cv_jd(job_index: str, cv: Annotated[str, InjectedState("cv")], tool_call_id: Annotated[str, InjectedToolCallId]):
    """Analyze and tailor a CV to match a specific job description.

    This tool compares the candidate's CV with the provided JD and rewrites or enhances relevant
    sections to increase alignment, keyword relevance, and chances of passing ATS screening.

    Key functions include:
        - Matching skills and experience to job requirements
        - Rewriting bullet points to reflect results and impact
        - Adding relevant keywords from the JD
        - Highlighting transferable skills or accomplishments
        - Suggesting structural changes for better flow and focus
        
    Args:
        job_index (str): The identifier of the job description to compare against."""
    print("--tool: match_cv--")

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
            "messages": [ToolMessage("Succesfully matching", tool_call_id=tool_call_id)],
            "new_cv": result.get("new_cv", ""),
            "cv_reviews": result.get("review", "")
        }
    )
