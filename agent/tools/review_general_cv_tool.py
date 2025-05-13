from pydantic import BaseModel, Field
from typing_extensions import TypedDict, Annotated
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import InjectedState
from langchain_core.tools.base import InjectedToolCallId
from langchain_core.tools import tool
from langgraph.types import Command
from agent.llm_provider import get_llm_structured, get_llm

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
You are an HR expert whose task is to review and evaluate a candidate’s CV based on the following detailed criteria. Carefully read and analyze the CV, then provide clear, actionable feedback. Highlight specific errors and offer direct suggestions for improvement. Avoid vague or generic comments. Follow these evaluation points:
	1.	Relevance to the job:

	•	Does the CV show a clear connection to the target job?
	•	Are relevant skills, education, and keywords included?

	2.	Timeline structure:

	•	Are all time periods for education and work fully listed?
	•	Is the timeline sorted in reverse chronological order?
	•	Are there unexplained time gaps?

	3.	Personal information:

	•	Is the candidate’s name clearly and prominently displayed?
	•	Is the email professional (ideally containing the name)?
	•	Is the phone number formatted for easy reading (e.g., 0123 456 789)?
	•	Are social media links appropriate and professional (prefer LinkedIn over Facebook)?
	•	Is the address overly detailed? (Only district/city is needed)
	•	Are there unnecessary details like ID number, gender, age, or marital status?
	•	Is the personal info section concise (less than 4 lines)?
	•	Is the photo professional (no selfies, clear face, plain background)?

	4.	Career objective (optional):

	•	Is the career objective clear and concise (maximum 4 lines)?
	•	Is it personalized, specific, and not generic?

	5.	Education:

	•	Does it include institution name, major, and start/end dates?
	•	Is the order correct (most recent first)?
	•	Are relevant courses or certifications beyond formal degrees listed?

	6.	Skills:

	•	Is there a skills section?
	•	Are the skills relevant to the job?
	•	Are soft skills and technical skills clearly differentiated?
	•	Are rating bars used without clear criteria? If yes, remove or replace with measurable standards (e.g., IELTS 6.5).
	•	Are there 4–8 skills listed? Too few or too many may be ineffective.
	•	Are action verbs used to describe the skills? (e.g., “Manage a team” instead of “Leadership”)

	7.	Work experience & activities:

	•	Is the timeline sorted from most recent to oldest?
	•	Are there unexplained gaps?
	•	Does each job include a short description of the company (size, product, industry)?
	•	Are there measurable results (numbers, KPIs, achievements)?
	•	Does each role have at least 3 bullet points describing tasks and outcomes?

	8.	References:

	•	Are references only included when requested?
	•	If listed, do they contain full name, position, company, email, and phone number?

	9.	Hobbies:

	•	Do the listed hobbies help the candidate stand out?
	•	Are they specific (e.g., reading psychology books, listening to classical music)?

	10.	Overall layout and formatting:

	•	Is the structure logical? (Personal Info > Skills > Experience > Education)
	•	Is the length appropriate (between 40–80 lines)?
	•	Are personal pronouns like “I”, “My”, or “Me” avoided?

After reviewing, return your feedback in the form of a checklist or structured list, pointing out both the problems and suggested improvements.

Context:
Candidate’s CV:
{candidate_cv}"""

adjust_instruction = """
You are an AI-powered CV Editing Specialist. Improve the candidate’s CV based on reviewer feedback and industry standards.

Candidate’s CV:
{candidate_cv}

Carefully adjust the CV following key criteria:
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

    system_message = review_instruction.format(
        candidate_cv=candidate_cv,
    )

    structured_llm = get_llm_structured(Feedbacks)
    feedbacks = structured_llm.invoke(
        [SystemMessage(system_message), HumanMessage('Let start the review process /no_think')]
    )

    return {'review': feedbacks.feedbacks}

def adjust_cv(state: SuggestChangeState) -> ReviewedCV:
    print('--adjust--')
    candidate_cv = state["candidate_cv"]
    feedbacks = state['review']

    criteria = "\n".join([
        f"{i+1}. {fb.criteria}: {fb.issue}\n\tSolution: {fb.solution}"
        for i, fb in enumerate(feedbacks)
    ])

    system_message = adjust_instruction.format(
        candidate_cv=candidate_cv,
        criteria=criteria,
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
def review_cv(cv: Annotated[str, InjectedState("cv")], tool_call_id: Annotated[str, InjectedToolCallId]):
    """Analyze and edit a CV without referencing any specific job description.

    This tool not only reviews the overall structure and content of the CV,
    but also rewrites or reformats key sections for better clarity, impact, and professionalism.

    Key functions include:
        - Identifying missing or redundant sections
        - Improving formatting and layout for ATS readability
        - Enhancing clarity of career objectives, skills, and experiences
        - Removing unprofessional or unnecessary personal details
        - Correcting grammar, tone, and stylistic issues"""
    print("--tool 2.2: review_cv--")

    if not cv:
        raise FileExistsError('CV is not uploaded yet.')

    result = review_agent.invoke({
        "candidate_cv": cv,
    })

    return Command(
        update={
            "messages": [ToolMessage("Succesfully review", tool_call_id=tool_call_id)],
            "new_cv": result.get("new_cv", ""),
            "cv_reviews": result.get("review", "")
        }
    )
