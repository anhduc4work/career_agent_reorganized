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


EXTRACTOR_INSTRUCTION = """
# Role and Objective  
You are a Job Description (JD) analysis expert. Your mission is to deeply analyze the provided JD and extract key hiring signals that reveal the employer’s true expectations and priorities for the ideal candidate. You are not just listing what’s mentioned — you must infer deeper intent and structure your output clearly.

# Task  
Analyze the given job description and extract the most important requirements and signals. Your output must be paraphrased and organized into **five categories**:

## Output Format (Use exact structure):
1. **Technical Skills**  
List relevant tools, technologies, programming languages, frameworks, platforms, or systems explicitly or implicitly required.

2. **Soft Skills**  
List essential non-technical traits or behaviors (e.g. communication, teamwork, leadership, problem-solving).

3. **Experience Requirements**  
Summarize the years of experience, industry background, project types, or job roles needed.

4. **Education & Certifications**  
List any degrees, majors, educational levels, or certifications that are required or preferred.

5. **Hidden Insights (Recruiter Intent)**  
Reflect on the overall tone, emphasis, and structure of the JD to infer the *real* goals behind the role.  
- Why is this role open now?  
- What kind of personality or archetype fits best?  
- What challenges might this role be solving?

# Instructions  

- **Paraphrase only** — Do **not** copy-paste content from the JD.  
- **Think critically** — Focus on what matters most, not just what’s explicitly stated.  
- **Be concise** — Use clean, bullet-pointed lists that are easy to scan.  
- **Infer meaning** — Capture both overt and subtle signals from the JD.  

# Reasoning Steps  
1. Read the full JD carefully.  
2. Identify explicit requirements and implied expectations.  
3. Categorize the findings under the 5 output sections.  
4. Paraphrase everything professionally and clearly.  

# Input Context  
**Job Description (JD):**  
```text
{job_description}

Final Note

You are acting as a specialist whose job is to extract and interpret hiring intent behind job postings. Respond with only the structured output. Do not include any explanations, introductions, or summaries outside of the 5 required categories.
"""

from pydantic import BaseModel, Field
from typing import List

class ExtractorOutput(BaseModel):
    technical_skills: List[str] = Field(
        ..., 
        description="List of tools, technologies, programming languages, frameworks, platforms, or systems required or implied."
    )
    soft_skills: List[str] = Field(
        ..., 
        description="List of essential non-technical traits or behaviors expected for the role."
    )
    experience_requirements: List[str] = Field(
        ..., 
        description="Summarized expectations for years of experience, industries, project types, or job roles."
    )
    education_certifications: List[str] = Field(
        ..., 
        description="Degrees, majors, education levels, or certifications that are required or preferred."
    )
    hidden_insights: List[str] = Field(
        ..., 
        description="Inferred recruiter intent such as purpose of the role, ideal personality, timing, or underlying organizational needs."
    )

class ExtractorInput(TypedDict):
    job_description: str
    curriculum_vitae: str
    
    
def extract_jd(state):
    print('--extract--')
    print(type(state), state)
    
    job_description = state["job_description"]
    
    system_message = EXTRACTOR_INSTRUCTION.format(
        job_description=job_description,
    )

    structured_llm = get_llm_structured(ExtractorOutput)
    response = structured_llm.invoke(
        [SystemMessage(system_message), HumanMessage('Start extracting')]
    )
    print("--done extract--")
    # return Send('analyze_cv', {"extractor_insights": response, "curriculum_vitae": state["curriculum_vitae"]})
    return {"extractor_insights": response}




# ---------------------------------------------------------

ANALYST_INSTRUCTION = """# Role and Objective
You are a career analysis expert. Your mission is to evaluate how well a candidate's CV aligns with the key hiring criteria extracted from a job description (JD).

You will be given:
- A list of **JD Insights** (divided into 5 categories).
- The candidate’s **CV** (in text form).

# Task
For each of the 5 JD insight categories listed below, assess the CV and provide structured feedback.

## Output Format (Use exact structure):

Return a list of 5 feedback items, each containing:

1. **name**  
One of the following exact values:
   - `technical_skills`
   - `soft_skills`
   - `experience_requirements`
   - `education_certifications`
   - `hidden_insights`

2. **score**  
Rate from 0 to 10 how well the CV fulfills the expectations for that category (higher is better).

3. **comment**  
Give a concise, professional evaluation. Explain how well the CV meets this requirement. Mention any gaps or strong points.

## Scoring Guide
- **9–10**: Fully satisfies the requirement; matches perfectly or exceeds expectations.
- **7–8**: Mostly meets expectations; minor gaps.
- **4–6**: Partially meets expectations; some important aspects missing.
- **1–3**: Barely relevant; major issues.
- **0**: Not addressed at all in the CV.

# Categories to Evaluate
You must review the CV according to the following categories:

1. **Technical Skills**  
Compare listed tools, technologies, platforms, or programming languages with what the JD expects.

2. **Soft Skills**  
Evaluate evidence of interpersonal traits like communication, teamwork, leadership, adaptability, etc.

3. **Experience Requirements**  
Check for years of experience, relevant industries, types of projects, and similar past roles.

4. **Education & Certifications**  
Look for relevant degrees, fields of study, or professional certifications.

5. **Hidden Insights (Recruiter Intent)**  
Assess whether the candidate seems to match the deeper intent behind the role (e.g. personality, purpose of the role, cultural fit).

# Instructions
- Be objective and professional.
- Justify the score with a short but insightful comment.
- Think critically. Don’t just match keywords — understand fit.
- Respond ONLY with the structured output, nothing more.

# Input Context

## JD Insights
{insights}

## CV
{curriculum_vitae}"""

class Feedback(BaseModel):
    name: Literal[
        'technical_skills',
        'soft_skills',
        'experience_requirements',
        'education_certifications',
        'hidden_insights'
    ] = Field(..., description="The criteria being evaluated.")
    score: int = Field(..., ge=0, le=10, description="Score from 0 to 10 reflecting how well the CV meets this requirement.")
    comment: str = Field(..., description="Brief comment analyzing how well the CV fulfills this requirement.")

class AnalystOutput(BaseModel):
    feedbacks: List[Feedback] = Field(..., description="List of feedback items for each key requirement category.")

class AnalystInput(TypedDict):
    insights: str
    curriculum_vitae: str

def analyze_cv(state):
    print('--analyze--')
    print(type(state), state)
    
    curriculum_vitae = state["curriculum_vitae"]
    insights = state["extractor_insights"]

    system_message = ANALYST_INSTRUCTION.format(
        curriculum_vitae = curriculum_vitae,
        insights = insights
    )

    structured_llm = get_llm_structured(AnalystOutput)
    response = structured_llm.invoke(
        [SystemMessage(system_message), HumanMessage('Start analyzing')]
    )
    print("--done analyze--")

    # return Send('suggest_cv', {"insights": response, "curriculum_vitae": state["curriculum_vitae"]})
    return {"analyst_insights": response}


# ____________________________________________________-


from pydantic import BaseModel, Field
from typing import List, Literal

SUGGESTOR_INSTRUCTION = """# Role and Objective
You are a professional CV improvement advisor. Your task is to review structured feedback about how well a CV matches a job description, and determine whether and how the CV can be improved.

Your job is to:
- Assess each requirement category based on the feedback
- Decide whether improvements are possible through rewriting or restructuring the CV
- Suggest how to improve it if possible
- Identify relevant keywords or phrases that could be added to enhance alignment with the JD

# Input
You will receive a list of 5 feedback entries. Each entry includes:
- `name`: the requirement category (e.g., "technical_skills")
- `score`: numeric score (0–10) reflecting how well the CV meets that requirement
- `comment`: a brief explanation of the score

# For each category, return the following:
1. `name`: the category name (keep unchanged)
2. `action_needed`: one of:
   - `"yes"` → Improvements are possible and recommended
   - `"no"` → Already strong, no change needed
   - `"cannot_be_improved"` → Gaps reflect real limitations that cannot be fixed through CV editing
3. `recommendation`: 
   - If `"yes"` → Suggest what to rephrase, highlight, or clarify in the CV
   - If `"no"` → Briefly note the strength
   - If `"cannot_be_improved"` → Explain why improvement is not possible through writing
4. `suggested_keywords`: 
   - A list of relevant terms (technologies, behaviors, job titles, etc.) that could be added to improve alignment
   - Leave empty (`[]`) if `action_needed` is `"no"` or `"cannot_be_improved"`

# Guidelines
- Be realistic: do not suggest faking credentials or experience.
- Think critically about what can actually be added, emphasized, or reworded in a CV.
- Use bullet points and keep output concise and professional.

# Input Context
**Feedback List:**
{insights}"""

class ImprovementSuggestion(BaseModel):
    name: Literal[
        'technical_skills',
        'soft_skills',
        'experience_requirements',
        'education_certifications',
        'hidden_insights'
    ] = Field(..., description="The requirement category being evaluated.")
    
    action_needed: Literal[
        'yes',                 # CV can and should be improved for this aspect
        'no',                  # This aspect is already strong, no changes needed
        'cannot_be_improved'   # This gap reflects something that cannot realistically be improved through CV edits (e.g., lack of years of experience)
    ] = Field(..., description="Whether the CV can be improved in this area.")
    
    current_expression: str = Field(..., description="How the candidate has currently demonstrated or expressed this aspect in the CV. Paraphrase or quote relevant parts.")
    
    recommendation: str = Field(..., description="Suggestion on what to change, emphasize, or reword in the CV. If no action is needed or not possible, explain why.")
    
    suggested_keywords: List[str] = Field(..., description="List of relevant keywords (technologies, soft skills, etc.) to consider adding to improve alignment with the JD.")


class SuggestorOutput(BaseModel):
    suggestions: List[ImprovementSuggestion] = Field(..., description="List of improvement suggestions for each requirement category.")


class SuggestorInput(TypedDict):
    insights: str
    curriculum_vitae: str

def suggest_cv(state):
    print('--suggest--')
    print(type(state), state)
    
    # curriculum_vitae = state["curriculum_vitae"]
    insights = state["analyst_insights"]

    system_message = SUGGESTOR_INSTRUCTION.format(
        insights = insights
    )

    structured_llm = get_llm_structured(SuggestorOutput)
    response = structured_llm.invoke(
        [SystemMessage(system_message), HumanMessage('Start analyzing')]
    )
    print('--done suggest--')

    # return Send('write_cv', {"insights": response, "curriculum_vitae": state["curriculum_vitae"]})
    return {"suggestor_insights": response}


# _____________________


WRITER_INSTRUCTION = """
# Role and Objective  
You are a professional CV editing assistant. Your job is to improve the candidate’s CV to better align with a job description (JD), based on a structured improvement plan generated by a JD-CV analysis agent.

Your edits should make the CV more relevant, professional, and compelling — while staying truthful to the candidate’s background.

# Task  
Use the improvement suggestions provided to revise the CV. You may:
- Emphasize or reword existing experience
- Add missing but plausible details based on the candidate's background
- Inject relevant keywords or terminology from the job description
- Reorganize or clarify content for better readability

# Editing Rules  
- **Do not fabricate** experience that doesn’t exist (e.g., fake years of experience or degrees).
- You may reframe or emphasize relevant work to match JD expectations.
- Do not add certifications, education, or job titles the candidate does not have.
- Add relevant **keywords** where appropriate, based on suggestion.
- Be clear, concise, and use a professional tone.
- You can update or add bullet points, rephrase summaries, and enhance skill sections.

# Input  
You will receive the following:
1. The original CV text.
2. A list of improvement suggestions in structured format, each with:
   - The category (e.g. technical skills, soft skills)
   - Whether an action is needed
   - A recommendation
   - Suggested keywords

# Output  
Return the **revised CV text only** — do not include any commentary or explanation. The CV should reflect the edits and improvements based on the provided plan.

# Context  
## Original CV:
```text
{curriculum_vitae}

Improvement Plan:

{insights}

Final Note

Do not mention that this is an edited CV. Just return the improved CV text.
"""


class WriterOutput(BaseModel):
    new_cv: str = Field(..., description="The fully rewritten and improved CV based on the provided suggestions. The output should be clean, professional, and align closely with the job description.")

class WriterInput(TypedDict):
    insights: str
    curriculum_vitae: str

def writer_cv(state):
    print('--writer--')
    print(type(state), state)
    curriculum_vitae = state["curriculum_vitae"]
    insights = state["suggestor_insights"]

    system_message = WRITER_INSTRUCTION.format(
        curriculum_vitae = curriculum_vitae,
        insights = insights
    )
    llm = get_llm()
    response = llm.invoke(
        [SystemMessage(system_message), HumanMessage('Start rewriting /no_think')]
    )
    extractor = get_llm_structured(WriterOutput)
    final = extractor.invoke([HumanMessage(f"""The following message contains a rewritten CV. Please extract **only the full CV text** from it.
    Message:
    {response.content}""")])
    print('--done write--')
    
    return final

# ---------------------------- COMBINED FLOW ----------------------------
class AgentState(TypedDict):
    job_description: str
    curriculum_vitae: str
    extractor_insights: dict
    analyst_insights: dict
    suggestor_insights: dict
    new_cv: str
    
    


workflow = StateGraph(AgentState)
workflow.add_node('extract_jd', extract_jd)
workflow.add_node('analyze_cv', analyze_cv)
workflow.add_node('suggest_cv', suggest_cv)
workflow.add_node('write_cv', writer_cv)
workflow.add_edge('extract_jd', 'analyze_cv')
workflow.add_edge('analyze_cv', 'suggest_cv')
workflow.add_edge('suggest_cv', 'write_cv')
workflow.set_entry_point("extract_jd")
workflow.set_finish_point('write_cv')
match_cv_jd_agent = workflow.compile()



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
    print("--tool: match_cv_jd--")

    if not cv:
        raise FileExistsError('CV is not uploaded yet.')

    jd = vector_store.get_by_ids([job_index])[0]
    
    if not jd:
        raise FileExistsError('JD is not available.')
        
    jd = jd.page_content

    result = match_cv_jd_agent.invoke({
        "job_description": jd,
        "curriculum_vitae": cv,
    })

    print(result)
    result = result.dict()
    
    return Command(
        update={
            "messages": [ToolMessage(json.dumps({
                "new_cv": result.get("new_cv", ""),
                "analyst_insights": result.get("analyst_insights", {})
            }), tool_call_id=tool_call_id)],
            "new_cv": result.get("new_cv", ""),
            "extractor_insights": result.get("extractor_insights", {}),
            "analyst_insights": result.get("analyst_insights", {}),
            "suggestor_insights": result.get("suggestor_insights", {}),
        }
    )
